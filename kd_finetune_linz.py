# ============================================================
# KNOWLEDGE-DISTILLATION 5-FOLD CV FINE-TUNE OF A BASE STUDENT ON LINZ
# ------------------------------------------------------------
# Lifts the weaker base models (ResNet / DenseNet) toward the stronger
# TorchGeo foundational models by distilling from a LINZ-trained TorchGeo
# model (the TEACHER) while the student is LoRA/adapter fine-tuned on LINZ.


import csv
import json
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# student side (torchgeo-FREE helpers + base-model builders)
from base_models.xbd_crops_common import (
    build_resnet6ch, inject_lora_resnet, build_densenet,
    pair_to_6ch, build_labelled_pairs, augment_6ch,
    RESNET_SIZE, RESNET_ARCHS, OUTPUT_SIZE, ADAPTER_DIM, NUM_CLASSES,
    stratified_folds, threshold_sweep, fbeta, roc_auc,
    best_threshold_youden, average_precision, build_criterion,
    plot_training_curves,
)
# teacher side (needs torchgeo -> envft only)
from torchGeo.torchgeo_common_linz import build_model as build_tg_model, model_input_size


# ============================================================
# CONFIG
# ============================================================
DATA_DIRS      = ["/home/c/cpehesar/research/CG_250m/train",
                  "/home/c/cpehesar/research/CG_250m/hold"]   # pooled for CV
BASE_CKPT_DIR  = "/home/c/cpehesar/research/base_models/checkpoints"      # student xBD weights
TEACHER_CKPT_DIR = "/home/c/cpehesar/research/torchGeo/checkpoints"      # teacher benchmark ckpts
OUT_CKPT_DIR   = "/home/c/cpehesar/research/checkpoints/kd_ckpts"        # distilled students
RESULTS_CSV    = "/home/c/cpehesar/research/distill_results.csv"          # appended per run
PLOTS_DIR      = "/home/c/cpehesar/research/bm_plots"                    # loss curves per run



STUDENT         = "resnet101"                 # resnet18|resnet50|resnet101|densenet121
STUDENT_WEIGHTS = "./base_models/checkpoints/resnet101_xbd_lora_best.pth"                   # default derived in main
TEACHER         = "swin_v2_b_satlas_naip"    # any TorchGeo registry name
TEACHER_CKPT    = "./torchGeo/checkpoints/swin_v2_b_satlas_naip_bestFocal.pth"                       # default derived in main
MODEL_TAG       = None                       # set in main

# distillation
KD_ALPHA      = 0.5     # weight on the soft (teacher) term
KD_T          = 4.0     # softmax temperature

# LoRA (ResNet student)
LORA_R        = 8
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.05

N_FOLDS       = 5
EPOCHS        = 25       # per fold
AUG_FACTOR    = 5
BATCH_SIZE    = 12
LEARNING_RATE = 1e-4
FBETA         = 2.0
SEED          = 42
NUM_WORKERS   = 4

# hard-label loss: cross_entropy | focal | dice | focal_dice
LOSS_CFG = {"type": "cross_entropy", "class_weighting": True,
            "focal": {"gamma": 2.0}, "dice": {"smooth": 1.0},
            "combo": {"focal_weight": 1.0, "dice_weight": 1.0}}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# TEACHER / STUDENT BUILDERS
# ============================================================
def build_teacher():
    """LINZ-trained TorchGeo model, loaded and FULLY FROZEN (eval mode)."""
    if not os.path.exists(TEACHER_CKPT):
        raise FileNotFoundError(
            f"Teacher checkpoint '{TEACHER_CKPT}' not found. Point --teacher-ckpt "
            f"at the LINZ-trained benchmark checkpoint for '{TEACHER}'.")
    model = build_tg_model(TEACHER, num_classes=NUM_CLASSES, freeze_backbone=True)
    ck = torch.load(TEACHER_CKPT, map_location=DEVICE, weights_only=False)
    state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model.to(DEVICE), model_input_size(TEACHER)


def build_student():
    """Fresh student from its xBD weights + PEFT. Returns (model, size, meta).
    The xBD backbone is frozen; only PEFT deltas + stem + head train."""
    if STUDENT in RESNET_ARCHS:
        model = build_resnet6ch(STUDENT, num_classes=NUM_CLASSES,
                                freeze_backbone=True, pretrained=False)
        ck = torch.load(STUDENT_WEIGHTS, map_location=DEVICE, weights_only=False)
        state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        model.load_state_dict(state)
        inject_lora_resnet(model, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
        meta = {"arch": STUDENT, "input_size": RESNET_SIZE, "finetune": "lora",
                "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT}}
        return model.to(DEVICE), RESNET_SIZE, meta
    else:   # densenet121
        model = build_densenet(num_classes=NUM_CLASSES, adapter_dim=ADAPTER_DIM,
                               pretrained=False)
        ck = torch.load(STUDENT_WEIGHTS, map_location=DEVICE, weights_only=False)
        state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        model.load_state_dict(state)
        meta = {"arch": "densenet121", "input_size": OUTPUT_SIZE,
                "finetune": "adapter", "adapter_dim": ADAPTER_DIM}
        return model.to(DEVICE), OUTPUT_SIZE, meta


# ============================================================
# DATA  (cache one base tensor at WORK_SIZE; resize per-model in the loop)
# ============================================================
class FoldDataset(Dataset):
    def __init__(self, base, labels, indices, aug_factor=1):
        self.base, self.labels = base, labels
        self.indices = list(indices)
        self.aug_factor = aug_factor

    def __len__(self):
        return len(self.indices) * self.aug_factor

    def __getitem__(self, i):
        pair_idx = self.indices[i // self.aug_factor]
        variant  = i % self.aug_factor
        t = self.base[pair_idx]
        if variant != 0:
            t = augment_6ch(t)
        return t, torch.tensor(self.labels[pair_idx], dtype=torch.long)


def class_weights(labels, indices):
    lab = np.asarray(labels)[list(indices)]
    n0 = int((lab == 0).sum()); n1 = int((lab == 1).sum()); tot = n0 + n1
    w0 = tot / (2 * max(n0, 1)); w1 = tot / (2 * max(n1, 1))
    return torch.tensor([w0, w1], dtype=torch.float32)


def _resize(x, size):
    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


# ============================================================
# DISTILLATION LOSS + TRAIN / EVAL ONE FOLD
# ============================================================
def kd_loss(student_logits, teacher_logits, labels, hard_criterion):
    T = KD_T
    soft = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction="batchmean") * (T * T)
    hard = hard_criterion(student_logits, labels)
    return KD_ALPHA * soft + (1.0 - KD_ALPHA) * hard


@torch.no_grad()
def eval_hard_loss(student, base, labels, indices, student_size, criterion):
    was_training = student.training
    student.eval()
    total, n = 0.0, 0
    for start in range(0, len(indices), BATCH_SIZE):
        batch = indices[start:start + BATCH_SIZE]
        x = _resize(torch.stack([base[i] for i in batch], dim=0).to(DEVICE), student_size)
        y = torch.tensor([labels[i] for i in batch], dtype=torch.long, device=DEVICE)
        total += criterion(student(x), y).item() * len(batch)
        n += len(batch)
    if was_training:
        student.train()
    return total / max(n, 1)


def train_fold(base, labels, train_idx, teacher, teacher_size, student_size, val_idx=None):
    student, size, meta = build_student()
    loader = DataLoader(FoldDataset(base, labels, train_idx, AUG_FACTOR),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    hard_criterion = build_criterion(LOSS_CFG, class_weights(labels, train_idx).to(DEVICE))
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, student.parameters()),
                            lr=LEARNING_RATE)
    val_idx = list(val_idx) if val_idx is not None else None
    history = {"train_loss": [], "val_loss": []}
    student.train()
    for _ in range(EPOCHS):
        running, n = 0.0, 0
        for images, lab in loader:
            images, lab = images.to(DEVICE), lab.to(DEVICE)
            x_s = _resize(images, student_size)
            x_t = _resize(images, teacher_size)
            with torch.no_grad():
                t_logits = teacher(x_t)
            s_logits = student(x_s)
            loss = kd_loss(s_logits, t_logits, lab, hard_criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0); n += images.size(0)
        history["train_loss"].append(running / max(n, 1))
        if val_idx is not None:
            history["val_loss"].append(
                eval_hard_loss(student, base, labels, val_idx, student_size, hard_criterion))
    return student, history, meta


@torch.no_grad()
def predict(student, base, indices, student_size):
    student.eval()
    probs = []
    for start in range(0, len(indices), BATCH_SIZE):
        batch = indices[start:start + BATCH_SIZE]
        x = _resize(torch.stack([base[i] for i in batch], dim=0).to(DEVICE), student_size)
        probs.extend(F.softmax(student(x), dim=1)[:, 1].cpu().numpy().tolist())
    return probs


# ============================================================
# RUN 5-FOLD CV  (mirrors resnet_finetune_linz_cv.py, with the KD loss)
# ============================================================
def run_cv(pairs, labels, teacher, teacher_size, student_size, work_size):
    print(f"\n=== {MODEL_TAG}  (student {student_size}px <- teacher {TEACHER} {teacher_size}px, "
          f"cache {work_size}px) ===")
    print("  caching base tensors...", flush=True)
    base = [pair_to_6ch(p[0], p[1], size=work_size) for p in pairs]
    os.makedirs(OUT_CKPT_DIR, exist_ok=True)

    folds = stratified_folds(labels, n_folds=N_FOLDS, seed=SEED)
    oof_prob = np.full(len(pairs), np.nan)
    per_fold = []
    fold_accuracies = []
    histories = []
    n_params = None
    t0 = time.time()
    best_fold_acc = -1.0
    best_fold_state = None
    best_fold_idx = None
    meta = None

    for k, test_idx in enumerate(folds):
        train_idx = np.array([i for i in range(len(pairs)) if i not in set(test_idx.tolist())])
        student, history, meta = train_fold(base, labels, train_idx, teacher,
                                            teacher_size, student_size,
                                            val_idx=test_idx.tolist())
        histories.append(history)
        if n_params is None:
            n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
        probs = predict(student, base, test_idx.tolist(), student_size)
        oof_prob[test_idx] = probs

        y = np.asarray(labels)[test_idx]
        sweep = threshold_sweep(y, np.array(probs))
        best = max(sweep, key=lambda r: fbeta(r["precision"], r["recall"], FBETA))
        per_fold.append(best)

        thr_y, _, _, _ = best_threshold_youden(y, probs)
        fold_pred = (np.array(probs) >= thr_y).astype(int)
        fold_acc = float((fold_pred == y).mean())
        fold_accuracies.append(fold_acc)
        if fold_acc >= best_fold_acc:
            best_fold_acc = fold_acc
            best_fold_idx = k + 1
            best_fold_state = {
                "model_name": MODEL_TAG, **meta,
                "fold": best_fold_idx,
                "teacher": TEACHER,
                "kd": {"alpha": KD_ALPHA, "temp": KD_T},
                "operating_threshold": thr_y,
                "threshold_rule": "youden_j",
                "validation_accuracy": best_fold_acc,
                "state_dict": student.state_dict(),
            }
        print(f"  fold {k+1}/{N_FOLDS}: AUC={roc_auc(y, probs):.3f} "
              f"acc={fold_acc:.3f}@thr{thr_y:.2f} "
              f"F{int(FBETA)}-best thr={best['threshold']:.2f} "
              f"recall={best['recall']:.3f} prec={best['precision']:.3f}", flush=True)
        del student
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    if best_fold_state is not None:
        best_fold_state["train_histories"] = histories
        ckpt = os.path.join(OUT_CKPT_DIR, f"{MODEL_TAG}_best.pth")
        torch.save(best_fold_state, ckpt)
        print(f"  saved best-fold checkpoint -> {ckpt}")

    plot_training_curves(histories, MODEL_TAG, PLOTS_DIR)

    y_all = np.asarray(labels)
    pooled_auc = roc_auc(y_all, oof_prob)
    pooled_ap = average_precision(y_all, oof_prob)
    pooled_sweep = threshold_sweep(y_all, oof_prob)
    pooled_best = max(pooled_sweep, key=lambda r: fbeta(r["precision"], r["recall"], FBETA))
    youden_thr, _, _, _ = best_threshold_youden(y_all, oof_prob)

    def ms(key):
        vals = np.array([f[key] for f in per_fold], dtype=float)
        return float(vals.mean()), float(vals.std())

    rec_m, rec_s = ms("recall")
    prec_m, prec_s = ms("precision")
    f1_m, f1_s = ms("f1")
    acc_m, acc_s = float(np.mean(fold_accuracies)), float(np.std(fold_accuracies))
    elapsed = (time.time() - t0) / 60.0

    row = {
        "model": MODEL_TAG,
        "arch": meta["arch"] if meta else STUDENT,
        "teacher": TEACHER,
        "student_weights": os.path.basename(STUDENT_WEIGHTS),
        "finetune": meta["finetune"] if meta else "?",
        "kd_alpha": KD_ALPHA, "kd_temp": KD_T,
        "loss": LOSS_CFG.get("type", "cross_entropy"),
        "input_size": student_size,
        "trainable_params": n_params,
        "pooled_auc": round(float(pooled_auc), 4),
        "pooled_ap": round(float(pooled_ap), 4),
        "pooled_best_threshold": pooled_best["threshold"],
        "youden_threshold": round(float(youden_thr), 4),
        "pooled_recall": round(pooled_best["recall"], 4),
        "pooled_precision": round(pooled_best["precision"], 4),
        "pooled_f1": round(pooled_best["f1"], 4),
        "cv_recall_mean": round(rec_m, 4), "cv_recall_std": round(rec_s, 4),
        "cv_precision_mean": round(prec_m, 4), "cv_precision_std": round(prec_s, 4),
        "cv_f1_mean": round(f1_m, 4), "cv_f1_std": round(f1_s, 4),
        "cv_accuracy_mean": round(acc_m, 4), "cv_accuracy_std": round(acc_s, 4),
        "minutes": round(elapsed, 1),
    }
    print(f"  >> pooled AUC={row['pooled_auc']}  AP={row['pooled_ap']}  "
          f"acc={acc_m:.2f}+/-{acc_s:.2f}  "
          f"recall={row['pooled_recall']} (cv {rec_m:.2f}+/-{rec_s:.2f})  "
          f"prec={row['pooled_precision']}  [{elapsed:.1f} min]")
    return row


def append_row(row, csv_path):
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


# ============================================================
# MAIN
# ============================================================
def _default_student_weights(student):
    if student in RESNET_ARCHS:
        return os.path.join(BASE_CKPT_DIR, f"{student}_xbd.pth")
    return "/home/c/cpehesar/research/base_models/checkpoints/densenet121_xbd.pth"   # xBD-pretrained densenet


def main():
    start_time = time.time()
    global STUDENT, STUDENT_WEIGHTS, TEACHER, TEACHER_CKPT, MODEL_TAG
    global KD_ALPHA, KD_T, EPOCHS, LORA_R, LORA_ALPHA
    parser = argparse.ArgumentParser(
        description="KD 5-fold CV fine-tune of a base student from a TorchGeo teacher")
    parser.add_argument("--student", default=STUDENT,
                        choices=list(RESNET_ARCHS) + ["densenet121"],
                        help="base student model")
    parser.add_argument("--student-weights", default=None,
                        help="xBD-pretrained student weights (default derived)")
    parser.add_argument("--teacher", default=TEACHER,
                        help="TorchGeo teacher registry name (LINZ-trained)")
    parser.add_argument("--teacher-ckpt", default=None,
                        help="teacher LINZ checkpoint (default checkpoints/ra_models/<t>_bestFocal.pth)")
    parser.add_argument("--kd-alpha", type=float, default=None, help="soft-loss weight (0..1)")
    parser.add_argument("--kd-temp", type=float, default=None, help="distillation temperature")
    parser.add_argument("--loss", default=None,
                        choices=["cross_entropy", "focal", "dice", "focal_dice"],
                        help="hard-label loss (default cross_entropy)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    args = parser.parse_args()

    STUDENT = args.student
    STUDENT_WEIGHTS = args.student_weights or _default_student_weights(STUDENT)
    TEACHER = args.teacher
    TEACHER_CKPT = args.teacher_ckpt or os.path.join(
        TEACHER_CKPT_DIR, f"{TEACHER}_bestFocal.pth")
    MODEL_TAG = (f"{STUDENT}_xbd_lora_kd" if STUDENT in RESNET_ARCHS
                 else f"{STUDENT}_xbd_kd")
    if args.kd_alpha is not None: KD_ALPHA = args.kd_alpha
    if args.kd_temp is not None:  KD_T = args.kd_temp
    if args.loss:                 LOSS_CFG["type"] = args.loss
    if args.epochs:               EPOCHS = args.epochs
    if args.lora_r:               LORA_R = args.lora_r
    if args.lora_alpha:           LORA_ALPHA = args.lora_alpha

    if not os.path.exists(STUDENT_WEIGHTS):
        raise SystemExit(f"Student xBD weights not found: {STUDENT_WEIGHTS}")

    torch.manual_seed(SEED); np.random.seed(SEED)

    pairs = []
    for d in DATA_DIRS:
        pairs.extend(build_labelled_pairs(d))
    labels = [p[2] for p in pairs]
    n0 = labels.count(0); n1 = labels.count(1)
    print(f"Pooled dataset: {len(pairs)} pairs (no_damage: {n0}, damaged: {n1})")
    if len(pairs) == 0:
        raise SystemExit("No labelled pairs found under DATA_DIRS.")

    teacher, teacher_size = build_teacher()
    student_size = RESNET_SIZE if STUDENT in RESNET_ARCHS else OUTPUT_SIZE
    work_size = max(teacher_size, student_size)
    print(f"{MODEL_TAG} | teacher={TEACHER} (frozen) | KD(alpha={KD_ALPHA}, T={KD_T}) | "
          f"{N_FOLDS}-fold CV | {EPOCHS} epochs/fold | aug x{AUG_FACTOR} | "
          f"batch {BATCH_SIZE} | loss={LOSS_CFG['type']} | device={DEVICE}")

    row = run_cv(pairs, labels, teacher, teacher_size, student_size, work_size)

    json_path = os.path.join(os.path.dirname(RESULTS_CSV), f"{MODEL_TAG}_results.json")
    with open(json_path, "w") as f:
        json.dump(row, f, indent=2)
    append_row(row, RESULTS_CSV)

    print("\n" + "=" * 70)
    print(f"  {MODEL_TAG} — KD 5-FOLD CV RESULT (teacher: {TEACHER})")
    print("=" * 70)
    print(f"  pooled AUC={row['pooled_auc']}  AP={row['pooled_ap']}  "
          f"acc={row['cv_accuracy_mean']:.2f}+/-{row['cv_accuracy_std']:.2f}  "
          f"recall={row['pooled_recall']}  prec={row['pooled_precision']}  F1={row['pooled_f1']}")
    print("=" * 70)
    print(f"Saved {json_path}, appended {RESULTS_CSV}, best-fold checkpoint in {OUT_CKPT_DIR}/")
    end_time = time.time()
    print(f"Elapsed time: {(end_time - start_time) / 60.0:.1f} minutes")



if __name__ == "__main__":
    main()
