# ============================================================
# 5-FOLD CROSS-VALIDATED BENCHMARK OF TORCHGEO BACKBONES (LINZ)
# ------------------------------------------------------------
# Fine-tunes EACH model in MODELS_TO_RUN with identical data / aug /
# loss / split, using stratified 5-fold CV over ALL labelled LINZ pairs
# (CG_250m/train + CG_250m/hold pooled). For each model it reports the
# pooled out-of-fold ROC-AUC plus per-fold mean +/- std of F1 / recall /
# precision at a recall-weighted (F2) operating point, and appends a row
# to benchmark_results.csv so the leaderboard builds itself.
#
# Why pooled out-of-fold: every crop is predicted exactly once, by a
# model that never saw it in training -> an honest, leakage-free score
# on the full dataset, which matters when you only have ~170 crops.
#
# Run (in envft, with the NeSI Python module loaded):
#   module load Python/3.11.6-foss-2023a
#   ./envft/bin/python torchGeo_benchmark_cv.py
# ============================================================
import os
import csv
import json
import time
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from torchgeo_common_linz import (
    MODELS, build_model, model_input_size, build_labelled_pairs, pair_to_6ch,
    augment_6ch, NUM_CLASSES, compute_metrics, roc_auc, threshold_sweep, fbeta,
    stratified_folds, roc_curve, pr_curve, best_threshold_youden, average_precision,
    build_criterion,
)

# headless matplotlib (no display on NeSI). The common import above already
# points MPLCONFIGDIR at nobackup, so importing matplotlib here is safe.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG  (built-in defaults; overridden by config.yaml via apply_config)
# ============================================================
CONFIG_PATH = os.environ.get(
    "BENCHMARK_CONFIG",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))

DATA_DIRS   = ["/nesi/nobackup/massey04767/CG_250m/train",
               "/nesi/nobackup/massey04767/CG_250m/hold"]   # pooled for CV
RESULTS_CSV = "benchmark_results.csv"
RESULTS_JSON = "benchmark_results.json"
CHECKPOINT_DIR = "checkpoints"
PLOTS_DIR   = "plots"

MODELS_TO_RUN = list(MODELS.keys())   # or a subset, e.g. ["resnet50_fmow_rgb", ...]

N_FOLDS       = 5
EPOCHS        = 30      # per fold
AUG_FACTOR    = 5
BATCH_SIZE    = 16
LEARNING_RATE = 1e-4
FREEZE_BACKBONE = True  # benchmark protocol: train only the 6-ch stem + head
FBETA         = 2.0     # recall-weighted operating point (missed damage costs more)
SEED          = 42
NUM_WORKERS   = 4

# loss selection (see config.yaml). cross_entropy | focal | dice | focal_dice
LOSS_CFG = {"type": "cross_entropy", "class_weighting": True,
            "focal": {"gamma": 2.0}, "dice": {"smooth": 1.0},
            "combo": {"focal_weight": 1.0, "dice_weight": 1.0}}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def apply_config(cfg):
    """Overlay a parsed config.yaml onto the module-level settings. Missing
    keys keep their built-in default, so partial configs are fine."""
    global DATA_DIRS, RESULTS_CSV, RESULTS_JSON, CHECKPOINT_DIR, PLOTS_DIR
    global MODELS_TO_RUN, N_FOLDS, EPOCHS, AUG_FACTOR, BATCH_SIZE, LEARNING_RATE
    global FREEZE_BACKBONE, FBETA, SEED, NUM_WORKERS, LOSS_CFG, DEVICE

    data = cfg.get("data", {}) or {}
    DATA_DIRS = data.get("dirs", DATA_DIRS)

    out = cfg.get("output", {}) or {}
    RESULTS_CSV    = out.get("results_csv", RESULTS_CSV)
    RESULTS_JSON   = out.get("results_json", RESULTS_JSON)
    CHECKPOINT_DIR = out.get("checkpoint_dir", CHECKPOINT_DIR)
    PLOTS_DIR      = out.get("plots_dir", PLOTS_DIR)

    models = cfg.get("models", "all")
    MODELS_TO_RUN = list(MODELS.keys()) if models in (None, "all", []) else list(models)

    tr = cfg.get("training", {}) or {}
    N_FOLDS         = int(tr.get("n_folds", N_FOLDS))
    EPOCHS          = int(tr.get("epochs", EPOCHS))
    AUG_FACTOR      = int(tr.get("aug_factor", AUG_FACTOR))
    BATCH_SIZE      = int(tr.get("batch_size", BATCH_SIZE))
    LEARNING_RATE   = float(tr.get("learning_rate", LEARNING_RATE))
    FREEZE_BACKBONE = bool(tr.get("freeze_backbone", FREEZE_BACKBONE))
    SEED            = int(tr.get("seed", SEED))
    NUM_WORKERS     = int(tr.get("num_workers", NUM_WORKERS))

    ev = cfg.get("evaluation", {}) or {}
    FBETA = float(ev.get("fbeta", FBETA))

    LOSS_CFG = cfg.get("loss", LOSS_CFG) or LOSS_CFG

    dev = cfg.get("device", "auto")
    DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if dev == "auto" else dev


# ============================================================
# DATASET over cached base tensors + an index list
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


def train_fold(model_name, base, labels, train_idx, val_idx=None):
    """Train one fold. Returns (model, history) where history holds per-epoch
    mean train loss and (if val_idx given) held-out validation loss."""
    model = build_model(model_name, num_classes=NUM_CLASSES,
                        freeze_backbone=FREEZE_BACKBONE).to(DEVICE)
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
# PLOTTING  (ROC / Precision-Recall / loss / threshold sweep)
# ============================================================
def plot_model_curves(model_name, y, oof_prob, histories, out_dir):
    """One figure per model: ROC, PR, train/val loss, and metric-vs-threshold,
    all from the pooled out-of-fold predictions (+ per-fold loss histories)."""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) ROC curve with the ROC-optimal (Youden) operating point marked
    fpr, tpr, _ = roc_curve(y, oof_prob)
    auc = roc_auc(y, oof_prob)
    thr_y, tpr_y, fpr_y, _ = best_threshold_youden(y, oof_prob)
    ax = axes[0, 0]
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.scatter([fpr_y], [tpr_y], color="red", zorder=5, label=f"Youden thr={thr_y:.2f}")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (pooled out-of-fold)"); ax.legend(loc="lower right")

    # (b) Precision-Recall curve with no-skill baseline (positive prevalence)
    rec, prec, _ = pr_curve(y, oof_prob)
    ap = average_precision(y, oof_prob)
    base_rate = float(np.mean(y))
    ax = axes[0, 1]
    ax.plot(rec, prec, lw=2, label=f"PR (AP={ap:.3f})")
    ax.axhline(base_rate, ls="--", color="gray", lw=1, label=f"baseline={base_rate:.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02)
    ax.set_title("Precision-Recall curve"); ax.legend(loc="lower left")

    # (c) Training / validation loss, mean +/- std across folds
    ax = axes[1, 0]
    tl = np.array([h["train_loss"] for h in histories], dtype=float)
    epochs = np.arange(1, tl.shape[1] + 1)
    ax.plot(epochs, tl.mean(0), color="C0", label="train")
    ax.fill_between(epochs, tl.mean(0) - tl.std(0), tl.mean(0) + tl.std(0),
                    color="C0", alpha=0.2)
    if histories and histories[0]["val_loss"]:
        vl = np.array([h["val_loss"] for h in histories], dtype=float)
        ax.plot(epochs, vl.mean(0), color="C1", label="validation")
        ax.fill_between(epochs, vl.mean(0) - vl.std(0), vl.mean(0) + vl.std(0),
                        color="C1", alpha=0.2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss (mean +/- std over folds)"); ax.legend()

    # (d) Precision / recall / F1 vs threshold (shows why 0.5 is suboptimal)
    ax = axes[1, 1]
    ts = np.round(np.arange(0.05, 0.96, 0.05), 2)
    sweep = threshold_sweep(y, oof_prob, thresholds=ts)
    ax.plot(ts, [s["precision"] for s in sweep], label="precision")
    ax.plot(ts, [s["recall"] for s in sweep], label="recall")
    ax.plot(ts, [s["f1"] for s in sweep], label="F1")
    ax.axvline(thr_y, ls="--", color="red", label=f"Youden={thr_y:.2f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title("Metrics vs threshold (pooled OOF)"); ax.legend()

    fig.suptitle(model_name, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, f"{model_name}_curves.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  saved curves -> {path}", flush=True)


def plot_combined(plot_data, out_dir):
    """Overlay every model's ROC and PR curve on one axis for direct comparison."""
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    for d in sorted(plot_data, key=lambda x: -x["auc"]):
        fpr, tpr, _ = roc_curve(d["y"], d["oof_prob"])
        ax.plot(fpr, tpr, lw=1.8, label=f"{d['name']} (AUC={d['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — all models (pooled out-of-fold)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "combined_roc.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    for d in sorted(plot_data, key=lambda x: -x["ap"]):
        rec, prec, _ = pr_curve(d["y"], d["oof_prob"])
        ax.plot(rec, prec, lw=1.8, label=f"{d['name']} (AP={d['ap']:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02)
    ax.set_title("Precision-Recall — all models (pooled out-of-fold)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "combined_pr.png"), dpi=120)
    plt.close(fig)
    print(f"Saved combined ROC/PR plots in {out_dir}/")


# ============================================================
# RUN ONE MODEL ACROSS 5 FOLDS
# ============================================================
def benchmark_model(model_name, pairs, labels):
    size = model_input_size(model_name)
    print(f"\n=== {model_name}  (input {size}px) ===")
    print("  caching base tensors...", flush=True)
    base = [pair_to_6ch(p[0], p[1], size=size) for p in pairs]
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    folds = stratified_folds(labels, n_folds=N_FOLDS, seed=SEED)
    oof_prob = np.full(len(pairs), np.nan)        # out-of-fold P(damaged)
    per_fold = []                                  # per-fold F2-opt metrics
    fold_accuracies = []                           # per-fold validation accuracy
    histories = []                                 # per-fold train/val loss curves
    n_params = None
    t0 = time.time()
    best_fold_acc = -1.0
    best_fold_state = None
    best_fold_idx = None

    for k, test_idx in enumerate(folds):
        train_idx = np.array([i for i in range(len(pairs)) if i not in set(test_idx.tolist())])
        model, history = train_fold(model_name, base, labels, train_idx,
                                    val_idx=test_idx.tolist())
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
                "model_name": model_name,
                "fold": best_fold_idx,
                "input_size": size,
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
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"{model_name}_bestFocalDice.pth",
        )
        torch.save(best_fold_state, checkpoint_path)

    # pooled out-of-fold metrics (every crop predicted once, unseen)
    y_all = np.asarray(labels)
    pooled_auc = roc_auc(y_all, oof_prob)
    pooled_ap = average_precision(y_all, oof_prob)
    pooled_sweep = threshold_sweep(y_all, oof_prob)
    pooled_best = max(pooled_sweep, key=lambda r: fbeta(r["precision"], r["recall"], FBETA))
    youden_thr, _, _, _ = best_threshold_youden(y_all, oof_prob)

    # diagnostic curves for this model (ROC / PR / loss / threshold sweep)
    plot_model_curves(model_name, y_all, oof_prob, histories, PLOTS_DIR)

    def ms(key):
        vals = np.array([f[key] for f in per_fold], dtype=float)
        return float(vals.mean()), float(vals.std())

    rec_m, rec_s = ms("recall")
    prec_m, prec_s = ms("precision")
    f1_m, f1_s = ms("f1")
    acc_m, acc_s = float(np.mean(fold_accuracies)), float(np.std(fold_accuracies))
    elapsed = (time.time() - t0) / 60.0

    row = {
        "model": model_name,
        "weights": MODELS[model_name]["weights"],
        "loss": LOSS_CFG.get("type", "cross_entropy"),
        "input_size": size,
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
    plot_data = {"name": model_name, "y": y_all, "oof_prob": oof_prob.copy(),
                 "auc": float(pooled_auc), "ap": float(pooled_ap)}
    return row, plot_data


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="5-fold CV TorchGeo benchmark")
    parser.add_argument("--config", default=CONFIG_PATH,
                        help="path to config.yaml (default: alongside this script)")
    args = parser.parse_args()
    if os.path.isfile(args.config):
        apply_config(load_config(args.config))
        print(f"Loaded config: {args.config}")
    else:
        print(f"WARNING: config '{args.config}' not found; using built-in defaults.")

    time0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    pairs = []
    for d in DATA_DIRS:
        pairs.extend(build_labelled_pairs(d))
    labels = [p[2] for p in pairs]
    n0 = labels.count(0); n1 = labels.count(1)
    print(f"Pooled dataset: {len(pairs)} pairs (no_damage: {n0}, damaged: {n1})")
    print(f"{N_FOLDS}-fold CV | {EPOCHS} epochs/fold | aug x{AUG_FACTOR} | "
          f"freeze_backbone={FREEZE_BACKBONE} | loss={LOSS_CFG.get('type')} | "
          f"device={DEVICE}")
    print(f"Models: {MODELS_TO_RUN}")

    rows = []
    plot_data = []
    for model_name in MODELS_TO_RUN:
        try:
            row, pdata = benchmark_model(model_name, pairs, labels)
            rows.append(row); plot_data.append(pdata)
        except Exception as e:
            print(f"  !! {model_name} FAILED: {type(e).__name__}: {e}")

    if not rows:
        raise SystemExit("No models completed.")

    if plot_data:
        plot_combined(plot_data, PLOTS_DIR)

    rows.sort(key=lambda r: -r["pooled_auc"])      # leaderboard by AUC

    fields = list(rows[0].keys())
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    with open(RESULTS_JSON, "w") as f:
        json.dump(rows, f, indent=2)
    



    print("\n" + "=" * 70)
    print("  BENCHMARK LEADERBOARD (sorted by pooled out-of-fold AUC)")
    print("=" * 70)
    print(f"  {'model':26s} {'AUC':>6s} {'acc':>10s} {'recall':>13s} {'prec':>6s} {'F1':>6s}")
    for r in rows:
        print(f"  {r['model']:26s} {r['pooled_auc']:6.3f} "
              f"{r['cv_accuracy_mean']:.2f}+/-{r['cv_accuracy_std']:.2f} "
              f"{r['pooled_recall']:.2f}+/-{r['cv_recall_std']:.2f} "
              f"{r['pooled_precision']:6.2f} {r['pooled_f1']:6.2f}")
    print("=" * 70)
    print(f"Saved {RESULTS_CSV}, {RESULTS_JSON}, best fold checkpoints in "
          f"{CHECKPOINT_DIR}/, and diagnostic curves in {PLOTS_DIR}/")
    
    endtime = time.time() - time0
    print(f"\nTotal benchmark time: {endtime/60:.1f} minutes")


if __name__ == "__main__":
    main()
