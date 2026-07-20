# ============================================================
# EVALUATE THE TORCHGEO MODEL ON THE LINZ HELD-OUT SPLIT
# ------------------------------------------------------------
# Runs on CG_250m/hold (DISJOINT from CG_250m/train), so the numbers
# are honest. Reports the same diagnostics as densenetEval_linz.py:
#   * default 0.5 confusion matrix + metrics
#   * ROC-AUC (threshold-free separability)
#   * a decision-threshold sweep + best-F1 operating point
#   * the exact crops it misses (false negatives) and false alarms
# ============================================================
import os
import json
import numpy as np
import torch
import torch.nn.functional as F

from torchgeo_common_linz import (
    build_model, build_labelled_pairs, pair_to_6ch, model_input_size, NUM_CLASSES,
    lora_target_modules, compute_metrics, roc_auc, threshold_sweep, fbeta,
    roc_curve, pr_curve, best_threshold_youden, average_precision,
)

# headless matplotlib (no display on NeSI). The common import above already
# points MPLCONFIGDIR at nobackup, so importing matplotlib here is safe.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================
MODEL_NAME   = "swin_v2_b_satlas_naip"   # must match the fine-tuned checkpoint
TRAIN_MODE   = "linear_probe" #'linear_probe' or 'lora'
MODEL_PATH   = "/home/c/cpehesar/research/torchGeo/checkpoints/swin_v2_b_satlas_naip_bestFocal.pth"  # Set checkpoint path here
HOLD_DIR     = "/home/c/cpehesar/research/Gisborne_250m"
RESULTS_JSON = "eval_torchgeo_linz.json"
PLOTS_DIR    = "torchGeo_plots"
FBETA        = 2.0     # recall-weighted operating point (missed damage costs more)
SIZE         = model_input_size(MODEL_NAME)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# MODEL
# ============================================================
model = build_model(MODEL_NAME, num_classes=NUM_CLASSES, freeze_backbone=True).to(DEVICE)

if TRAIN_MODE == "lora":
    target_modules = lora_target_modules(MODEL_NAME)
    print(f"Using LoRA targets for {MODEL_NAME}: {target_modules}")

    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config).to(DEVICE)

if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Checkpoint not found: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
# Extract actual state_dict if checkpoint has metadata
state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
model.load_state_dict(state_dict)
# the benchmark stores the ROC-optimal cutoff chosen at training time; prefer it
# over a test-set-fit threshold so the operating point isn't tuned on the test data
CKPT_THRESHOLD = checkpoint.get("operating_threshold") if isinstance(checkpoint, dict) else None
CKPT_THR_RULE  = checkpoint.get("threshold_rule", "youden_j") if isinstance(checkpoint, dict) else None
model.eval()
print(f"Model loaded from {MODEL_PATH} using mode={TRAIN_MODE}.\n")


# ============================================================
# BUILD TEST SET
# ============================================================
def build_crop_list(hold_dir):
    crops = []
    for pre_tif, post_tif, label, name in build_labelled_pairs(hold_dir):
        t = pair_to_6ch(pre_tif, post_tif, size=SIZE)
        if t is not None:
            crops.append((t, label, name))
    return crops


# ============================================================
# RUN MODEL — keep P(damaged) for threshold-free analysis
# ============================================================
def run_model(crops, batch_size=16):
    true_labels, damaged_prob, names = [], [], []
    for start in range(0, len(crops), batch_size):
        batch = crops[start:start + batch_size]
        x = torch.stack([c[0] for c in batch], dim=0).to(DEVICE)   # [B, 6, S, S]
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1)[:, 1].cpu().numpy()
        damaged_prob.extend(probs.tolist())
        true_labels.extend(c[1] for c in batch)
        names.extend(c[2] for c in batch)
    return np.array(true_labels), np.array(damaged_prob), names


# ============================================================
# METRICS + DIAGNOSTICS  (compute_metrics / roc_auc / threshold_sweep are
# now imported from torchgeo_common_linz — one source of truth shared with
# the benchmark script, so eval and benchmark always agree.)
# ============================================================
def _draw_cm(ax, cm, title, normalize=False):
    """Render a 2x2 confusion matrix (rows=actual, cols=predicted) on an axis."""
    disp = cm.astype(float)
    if normalize:
        rowsum = disp.sum(axis=1, keepdims=True)
        disp = np.divide(disp, rowsum, out=np.zeros_like(disp), where=rowsum != 0)
    vmax = disp.max() if disp.max() > 0 else 1.0
    ax.imshow(disp, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["no_damage", "damaged"]); ax.set_yticklabels(["no_damage", "damaged"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    for i in range(2):
        for j in range(2):
            txt = f"{disp[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=12,
                    color="white" if disp[i, j] > vmax / 2 else "black")


def _calibration(y, p, n_bins=10):
    """Reliability-diagram points: mean predicted prob vs observed positive rate."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 0:
            xs.append(float(p[m].mean())); ys.append(float(y[m].mean()))
    return np.array(xs), np.array(ys)


def plot_confusion(y, prob, thr_default, thr_op, thr_src, model_name, out_dir):
    """Confusion matrices (counts + recall-normalized) at the default 0.5 cutoff
    and at the operating threshold."""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for row, (thr, tag) in enumerate(
            [(thr_default, f"thr={thr_default:.2f} (default)"),
             (thr_op,      f"thr={thr_op:.2f} ({thr_src})")]):
        m = compute_metrics(y, (prob >= thr).astype(int))
        cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        _draw_cm(axes[row, 0], cm, f"Counts — {tag}", normalize=False)
        _draw_cm(axes[row, 1], cm, f"Row-normalized — {tag}", normalize=True)
    fig.suptitle(f"{model_name} — confusion matrices (held-out)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, f"eval_{model_name}_confusion.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  saved confusion matrices -> {path}")


def plot_eval_curves(y, prob, thr_op, model_name, out_dir):
    """Six-panel evaluation: ROC, PR, threshold sweep, score histogram,
    calibration, and a metric-summary bar chart."""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (a) ROC
    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc(y, prob)
    ty, tpr_y, fpr_y, _ = best_threshold_youden(y, prob)
    ax = axes[0, 0]
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.scatter([fpr_y], [tpr_y], color="red", zorder=5, label=f"Youden thr={ty:.2f}")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend(loc="lower right")

    # (b) Precision-Recall
    rec, prec, _ = pr_curve(y, prob)
    ap = average_precision(y, prob)
    base_rate = float(np.mean(y))
    ax = axes[0, 1]
    ax.plot(rec, prec, lw=2, label=f"PR (AP={ap:.3f})")
    ax.axhline(base_rate, ls="--", color="gray", lw=1, label=f"baseline={base_rate:.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02)
    ax.set_title("Precision-Recall curve"); ax.legend(loc="lower left")

    # (c) Metrics vs threshold
    ax = axes[0, 2]
    ts = np.round(np.arange(0.05, 0.96, 0.05), 2)
    sweep = threshold_sweep(y, prob, thresholds=ts)
    ax.plot(ts, [s["precision"] for s in sweep], label="precision")
    ax.plot(ts, [s["recall"] for s in sweep], label="recall")
    ax.plot(ts, [s["f1"] for s in sweep], label="F1")
    ax.axvline(0.5, ls=":", color="gray", label="0.5")
    ax.axvline(thr_op, ls="--", color="red", label=f"op={thr_op:.2f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title("Metrics vs threshold"); ax.legend()

    # (d) Score distribution by true class
    ax = axes[1, 0]
    ax.hist(prob[y == 0], bins=20, range=(0, 1), alpha=0.6, color="C0", label="no_damage")
    ax.hist(prob[y == 1], bins=20, range=(0, 1), alpha=0.6, color="C3", label="damaged")
    ax.axvline(thr_op, ls="--", color="red", label=f"op={thr_op:.2f}")
    ax.set_xlabel("P(damaged)"); ax.set_ylabel("count")
    ax.set_title("Score distribution by true class"); ax.legend()

    # (e) Calibration / reliability
    ax = axes[1, 1]
    xs, ys = _calibration(y, prob)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect")
    ax.plot(xs, ys, "o-", color="C2", label="model")
    ax.set_xlabel("Mean predicted P(damaged)"); ax.set_ylabel("Observed fraction damaged")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Calibration (reliability)"); ax.legend(loc="upper left")

    # (f) Metric summary at the operating threshold
    ax = axes[1, 2]
    m = compute_metrics(y, (prob >= thr_op).astype(int))
    labels = ["accuracy", "precision", "recall", "f1", "roc_auc", "avg_prec"]
    vals = [m["accuracy"], m["precision"], m["recall"], m["f1"], auc, ap]
    bars = ax.bar(labels, vals, color=["C0", "C1", "C3", "C4", "C5", "C6"])
    ax.set_ylim(0, 1.05); ax.tick_params(axis="x", rotation=30)
    ax.set_title(f"Metrics @ op threshold {thr_op:.2f}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=8)

    fig.suptitle(f"{model_name} — held-out evaluation ({len(y)} crops)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, f"eval_{model_name}_curves.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  saved eval curves -> {path}")


def print_report(m, n_no_damage, n_damaged):
    print("\n" + "="*55)
    print("  TORCHGEO LINZ EVALUATION — held-out (per 250 m crop)")
    print("="*55)
    print(f"\n  Crops tested : {m['total']}  (no_damage: {n_no_damage}, damaged: {n_damaged})")
    print(f"\n  Metrics @ threshold 0.5:")
    print(f"    Accuracy  : {m['accuracy']:.4f}  ({m['accuracy']*100:.1f}%)")
    print(f"    Precision : {m['precision']:.4f}")
    print(f"    Recall    : {m['recall']:.4f}   <- of all damaged crops, how many caught")
    print(f"    F1 Score  : {m['f1']:.4f}")
    print(f"\n  Confusion matrix:")
    print(f"                       Predicted")
    print(f"                   no_damage | damaged")
    print(f"    Actual no_dmg [  {m['TN']:6d}  |  {m['FP']:6d} ]")
    print(f"    Actual damaged[  {m['FN']:6d}  |  {m['TP']:6d} ]")
    if m['TP'] == 0:
        print("\n  *** WARNING: model never predicted 'damaged' ***")
    print("="*55)


def print_sweep(rows):
    print("\n  Threshold sweep (P(damaged) >= thr):")
    print("    thr    acc    prec    rec     f1    TP  FP  FN  TN")
    for r in rows:
        print(f"    {r['threshold']:.2f}  {r['accuracy']:.3f}  {r['precision']:.3f}  "
              f"{r['recall']:.3f}  {r['f1']:.3f}  {r['TP']:3d} {r['FP']:3d} "
              f"{r['FN']:3d} {r['TN']:3d}")


# ============================================================
# RUN
# ============================================================
print("--- Building held-out test set ---")
crops = build_crop_list(HOLD_DIR)
print(f"Extracted {len(crops)} crops.\n")
if len(crops) == 0:
    raise SystemExit(f"No crops under {HOLD_DIR}")

n_no_damage = sum(1 for c in crops if c[1] == 0)
n_damaged   = sum(1 for c in crops if c[1] == 1)

print("--- Running model ---")
true_labels, damaged_prob, names = run_model(crops)
names = np.array(names)

preds_default = (damaged_prob >= 0.5).astype(int)
metrics = compute_metrics(true_labels, preds_default)
print_report(metrics, n_no_damage, n_damaged)

auc = roc_auc(true_labels, damaged_prob)
print(f"\n  ROC-AUC (threshold-free) : {auc:.4f}")


sweep = threshold_sweep(true_labels, damaged_prob)
print_sweep(sweep)
best = max(sweep, key=lambda r: r["f1"])
print(f"\n  Best-F1 threshold : {best['threshold']:.2f}  "
      f"(F1={best['f1']:.3f}, recall={best['recall']:.3f}, precision={best['precision']:.3f})")

# recall-weighted (F-beta) operating point — same rule the benchmark reports
best_fbeta = max(sweep, key=lambda r: fbeta(r["precision"], r["recall"], FBETA))
print(f"  Best-F{int(FBETA)} threshold : {best_fbeta['threshold']:.2f}  "
      f"(recall={best_fbeta['recall']:.3f}, precision={best_fbeta['precision']:.3f})")

# decision threshold for the headline confusion matrix: prefer the cutoff the
# benchmark stored in the checkpoint; else fall back to the ROC-optimal (Youden)
# point fit on this test set (reported, but noted as test-fit).
youden_thr, _, _, _ = best_threshold_youden(true_labels, damaged_prob)
if CKPT_THRESHOLD is not None:
    op_threshold = float(CKPT_THRESHOLD); thr_src = f"ckpt {CKPT_THR_RULE}"
else:
    op_threshold = float(youden_thr); thr_src = "Youden/test"
op_metrics = compute_metrics(true_labels, (damaged_prob >= op_threshold).astype(int))
print(f"\n  Operating threshold : {op_threshold:.2f} ({thr_src})")
print(f"    acc={op_metrics['accuracy']:.3f}  recall={op_metrics['recall']:.3f}  "
      f"precision={op_metrics['precision']:.3f}  f1={op_metrics['f1']:.3f}")

# ---- evaluation plots ----
print("\n--- Writing evaluation plots ---")
plot_confusion(true_labels, damaged_prob, 0.5, op_threshold, thr_src, MODEL_NAME, PLOTS_DIR)
plot_eval_curves(true_labels, damaged_prob, op_threshold, MODEL_NAME, PLOTS_DIR)

fn_mask = (true_labels == 1) & (preds_default == 0)
fp_mask = (true_labels == 0) & (preds_default == 1)
false_negatives = [{"crop": n, "p_damaged": round(float(p), 4)}
                   for n, p in sorted(zip(names[fn_mask], damaged_prob[fn_mask]),
                                      key=lambda x: x[1])]
false_positives = [{"crop": n, "p_damaged": round(float(p), 4)}
                   for n, p in sorted(zip(names[fp_mask], damaged_prob[fp_mask]),
                                      key=lambda x: -x[1])]
print(f"\n  MISSED damage (false negatives, {len(false_negatives)}) — lowest P first:")
for fn in false_negatives:
    print(f"    {fn['p_damaged']:.3f}  {fn['crop']}")
print(f"\n  False alarms (false positives, {len(false_positives)}):")
for fp in false_positives:
    print(f"    {fp['p_damaged']:.3f}  {fp['crop']}")

def _round(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}

ap = average_precision(true_labels, damaged_prob)
results = _round(metrics)
results.update({
    "n_no_damage": n_no_damage, "n_damaged": n_damaged,
    "unit": "250m_crop", "dataset": "LINZ_cyclone_gabrielle_hold",
    "model": MODEL_NAME, "roc_auc": round(float(auc), 4),
    "average_precision": round(float(ap), 4),
    "operating_threshold": round(op_threshold, 4),
    "operating_threshold_source": thr_src,
    "operating_point": _round(op_metrics),
    "confusion_matrix_at_operating_point": {
        "TN": op_metrics["TN"], "FP": op_metrics["FP"],
        "FN": op_metrics["FN"], "TP": op_metrics["TP"],
    },
    "threshold_sweep": [_round(r) for r in sweep],
    "best_f1_operating_point": _round(best),
    "best_fbeta_operating_point": {"beta": FBETA, **_round(best_fbeta)},
    "false_negatives": false_negatives, "false_positives": false_positives,
})
with open(RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved metrics to {RESULTS_JSON} and plots to {PLOTS_DIR}/")
