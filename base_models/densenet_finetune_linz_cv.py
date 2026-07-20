# ============================================================
# 5-FOLD CV FINE-TUNE OF DENSENET121 (xBD-pretrained) ON LINZ
# ------------------------------------------------------------
# The DenseNet counterpart to torchGeo_benchmark_cv.py. It pools
# CG_250m/train + CG_250m/hold and runs stratified 5-fold CV with the
# SAME protocol as the TorchGeo foundational-model benchmark:
#   * AUG_FACTOR x geometry+photometric augmentation per crop
#   * class-weighted (or focal / dice / focal_dice) loss
#   * pooled out-of-fold ROC-AUC + per-fold mean±std F1/recall/precision
#     at a recall-weighted (F-beta) operating point
#   * ROC-optimal (Youden) threshold stored in the checkpoint
#   * the same ROC / PR / loss / threshold diagnostic plots
#
# The one difference vs the TorchGeo backbones: every fold starts FROM the
# xBD-pretrained densenet121 weights (domain adaptation) rather than from
# remote-sensing pretrain, and uses the xBD 512px 6-channel preprocessing
# so the inputs match how those weights were trained.
#
# Run (in envft, with the NeSI Python module loaded):
#   module load Python/3.11.6-foss-2023a
#   ./envft/bin/python densenet_finetune_linz_cv.py              # cross_entropy
#   ./envft/bin/python densenet_finetune_linz_cv.py --loss focal
# ============================================================
import os
import csv
import json
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Everything comes from xbd_crops_common (torchgeo-FREE): the DenseNet runs in
# an environment WITHOUT torchgeo, so this script must never import
# torchgeo_common_linz / torchGeo_benchmark_cv. Plotting lives in the eval
# script (densenetEval_linz_cv.py); here we just save the loss histories.
from xbd_crops_common import (
    build_densenet, OUTPUT_SIZE, pair_to_6ch, build_labelled_pairs, augment_6ch,
    stratified_folds, threshold_sweep, fbeta, roc_auc, best_threshold_youden,
    average_precision, build_criterion, plot_training_curves,
)


# ============================================================
# CONFIG
# ============================================================
DATA_DIRS       = ["/nesi/nobackup/massey04767/CG_250m/train",
                   "/nesi/nobackup/massey04767/CG_250m/hold"]   # pooled for CV
PRETRAINED_PATH = "/nesi/nobackup/massey04767/densenet121.pth"  # xBD-pretrained start
MODEL_TAG       = "densenet121_xbd"
RESULTS_CSV     = "densenet_cv_results.csv"
RESULTS_JSON    = "densenet_cv_results.json"
CHECKPOINT_DIR  = "checkpoints"
PLOTS_DIR       = "plots"

N_FOLDS       = 5
EPOCHS        = 30       # per fold
AUG_FACTOR    = 5
BATCH_SIZE    = 8        # 512x512 6-channel inputs are memory-heavy
LEARNING_RATE = 5e-5     # nudging the xBD weights, not retraining
NUM_CLASSES   = 2
ADAPTER_DIM   = 128
FBETA         = 2.0      # recall-weighted operating point (missed damage costs more)
SEED          = 42
NUM_WORKERS   = 4

# loss: cross_entropy | focal | dice | focal_dice  (override with --loss)
LOSS_CFG = {"type": "cross_entropy", "class_weighting": True,
            "focal": {"gamma": 2.0}, "dice": {"smooth": 1.0},
            "combo": {"focal_weight": 1.0, "dice_weight": 1.0}}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# DATA  (pair_to_6ch / augment_6ch come from xbd_crops_common — the xBD-style
# 512px 6-channel preprocessing the DenseNet was trained on)
# ============================================================
class FoldDataset(Dataset):
    """Cached base tensors expanded AUG_FACTOR x; variant 0 is the clean crop,
    variants 1.. are re-randomised augmentations (same scheme as the benchmark)."""
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


# ============================================================
# TRAIN / EVAL ONE FOLD  (each fold restarts from the xBD weights)
# ============================================================
def build_fold_model():
    """Fresh adapter-DenseNet121 loaded with the xBD-pretrained weights."""
    if not os.path.exists(PRETRAINED_PATH):
        raise FileNotFoundError(
            f"xBD-pretrained weights '{PRETRAINED_PATH}' not found — this script "
            f"fine-tunes FROM them, it does not train from scratch.")
    model = build_densenet(num_classes=NUM_CLASSES, adapter_dim=ADAPTER_DIM,
                           pretrained=False).to(DEVICE)
    state = torch.load(PRETRAINED_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    return model


@torch.no_grad()
def eval_loss(model, base, labels, indices, criterion):
    """Mean loss over a held-out (validation) fold, no augmentation."""
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for start in range(0, len(indices), BATCH_SIZE):
        batch = indices[start:start + BATCH_SIZE]
        x = torch.stack([base[i] for i in batch], dim=0).to(DEVICE)
        y = torch.tensor([labels[i] for i in batch], dtype=torch.long, device=DEVICE)
        total += criterion(model(x), y).item() * len(batch)
        n += len(batch)
    if was_training:
        model.train()
    return total / max(n, 1)


def train_fold(base, labels, train_idx, val_idx=None):
    """Train one fold from the xBD weights. Returns (model, history) with
    per-epoch mean train loss and held-out validation loss."""
    model = build_fold_model()
    loader = DataLoader(FoldDataset(base, labels, train_idx, AUG_FACTOR),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    criterion = build_criterion(LOSS_CFG, class_weights(labels, train_idx).to(DEVICE))
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LEARNING_RATE)
    val_idx = list(val_idx) if val_idx is not None else None
    history = {"train_loss": [], "val_loss": []}
    model.train()
    for _ in range(EPOCHS):
        running, n = 0.0, 0
        for images, lab in loader:
            images, lab = images.to(DEVICE), lab.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), lab)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0); n += images.size(0)
        history["train_loss"].append(running / max(n, 1))
        if val_idx is not None:
            history["val_loss"].append(eval_loss(model, base, labels, val_idx, criterion))
    return model, history


@torch.no_grad()
def predict(model, base, indices):
    model.eval()
    probs = []
    for start in range(0, len(indices), BATCH_SIZE):
        batch = indices[start:start + BATCH_SIZE]
        x = torch.stack([base[i] for i in batch], dim=0).to(DEVICE)
        probs.extend(F.softmax(model(x), dim=1)[:, 1].cpu().numpy().tolist())
    return probs


# ============================================================
# RUN 5-FOLD CV  (mirrors benchmark_model in torchGeo_benchmark_cv.py)
# ============================================================
def run_cv(pairs, labels):
    print(f"\n=== {MODEL_TAG}  (input {OUTPUT_SIZE}px, from {os.path.basename(PRETRAINED_PATH)}) ===")
    print("  caching base tensors...", flush=True)
    base = [pair_to_6ch(p[0], p[1]) for p in pairs]
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    folds = stratified_folds(labels, n_folds=N_FOLDS, seed=SEED)
    oof_prob = np.full(len(pairs), np.nan)        # out-of-fold P(damaged)
    per_fold = []                                  # per-fold F-beta-opt metrics
    fold_accuracies = []                           # per-fold validation accuracy
    histories = []                                 # per-fold train/val loss curves
    n_params = None
    t0 = time.time()
    best_fold_acc = -1.0
    best_fold_state = None
    best_fold_idx = None

    for k, test_idx in enumerate(folds):
        train_idx = np.array([i for i in range(len(pairs)) if i not in set(test_idx.tolist())])
        model, history = train_fold(base, labels, train_idx, val_idx=test_idx.tolist())
        histories.append(history)
        if n_params is None:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        probs = predict(model, base, test_idx.tolist())
        oof_prob[test_idx] = probs

        # per-fold operating point (F-beta best on this fold)
        y = np.asarray(labels)[test_idx]
        sweep = threshold_sweep(y, np.array(probs))
        best = max(sweep, key=lambda r: fbeta(r["precision"], r["recall"], FBETA))
        per_fold.append(best)

        # accuracy at the ROC-optimal threshold (Youden's J), not a fixed 0.5
        thr_y, _, _, _ = best_threshold_youden(y, probs)
        fold_pred = (np.array(probs) >= thr_y).astype(int)
        fold_acc = float((fold_pred == y).mean())
        fold_accuracies.append(fold_acc)
        if fold_acc >= best_fold_acc:
            best_fold_acc = fold_acc
            best_fold_idx = k + 1
            best_fold_state = {
                "model_name": MODEL_TAG,
                "fold": best_fold_idx,
                "input_size": OUTPUT_SIZE,
                "adapter_dim": ADAPTER_DIM,
                "operating_threshold": thr_y,     # ROC-optimal cutoff for inference
                "threshold_rule": "youden_j",
                "validation_accuracy": best_fold_acc,
                "state_dict": model.state_dict(),
            }
        print(f"  fold {k+1}/{N_FOLDS}: AUC={roc_auc(y, probs):.3f} "
              f"acc={fold_acc:.3f}@thr{thr_y:.2f} "
              f"F{int(FBETA)}-best thr={best['threshold']:.2f} "
              f"recall={best['recall']:.3f} prec={best['precision']:.3f}", flush=True)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    if best_fold_state is not None:
        # stash the per-fold loss histories so the eval script can plot the
        # training/validation loss curves (no plotting happens here)
        best_fold_state["train_histories"] = histories
        ckpt = os.path.join(CHECKPOINT_DIR, f"{MODEL_TAG}_best.pth")
        torch.save(best_fold_state, ckpt)
        print(f"  saved best-fold checkpoint -> {ckpt}")

    plot_training_curves(histories, MODEL_TAG, PLOTS_DIR)

    # pooled out-of-fold metrics (every crop predicted once, unseen)
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
        "weights": os.path.basename(PRETRAINED_PATH),
        "loss": LOSS_CFG.get("type", "cross_entropy"),
        "input_size": OUTPUT_SIZE,
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


# ============================================================
# MAIN
# ============================================================
def main():
    global EPOCHS
    parser = argparse.ArgumentParser(
        description="5-fold CV fine-tune of the xBD-pretrained DenseNet121 on LINZ")
    parser.add_argument("--loss", default=None,
                        choices=["cross_entropy", "focal", "dice", "focal_dice"],
                        help="override loss type (default: cross_entropy)")
    parser.add_argument("--epochs", type=int, default=None, help="override epochs/fold")
    args = parser.parse_args()
    if args.loss:
        LOSS_CFG["type"] = args.loss
    if args.epochs:
        EPOCHS = args.epochs

    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    pairs = []
    for d in DATA_DIRS:
        pairs.extend(build_labelled_pairs(d))
    labels = [p[2] for p in pairs]
    n0 = labels.count(0); n1 = labels.count(1)
    print(f"Pooled dataset: {len(pairs)} pairs (no_damage: {n0}, damaged: {n1})")
    print(f"{N_FOLDS}-fold CV | {EPOCHS} epochs/fold | aug x{AUG_FACTOR} | "
          f"batch {BATCH_SIZE} | lr {LEARNING_RATE} | loss={LOSS_CFG['type']} | device={DEVICE}")
    if len(pairs) == 0:
        raise SystemExit("No labelled pairs found under DATA_DIRS.")

    row = run_cv(pairs, labels)

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader(); w.writerow(row)
    with open(RESULTS_JSON, "w") as f:
        json.dump(row, f, indent=2)

    print("\n" + "=" * 70)
    print("  DENSENET121 (xBD) — 5-FOLD CV RESULT")
    print("=" * 70)
    print(f"  pooled AUC={row['pooled_auc']}  AP={row['pooled_ap']}  "
          f"acc={row['cv_accuracy_mean']:.2f}+/-{row['cv_accuracy_std']:.2f}  "
          f"recall={row['pooled_recall']}  prec={row['pooled_precision']}  F1={row['pooled_f1']}")
    print("=" * 70)
    print(f"Saved {RESULTS_CSV}, {RESULTS_JSON}, and best-fold checkpoint in "
          f"{CHECKPOINT_DIR}/ (loss histories embedded for the eval script's plots)")
    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
