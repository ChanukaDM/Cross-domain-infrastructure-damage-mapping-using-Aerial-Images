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
    lora_target_modules,
)


# ============================================================
# CONFIG
# ============================================================
MODEL_NAME   = "vit_small_sentinel2"   # must match the fine-tuned checkpoint
TRAIN_MODE   = "linear_probe" #'linear_probe' or 'lora'
MODEL_PATH   = "/nesi/nobackup/massey04767/checkpoints/vit_small_sentinel2_best.pth"  # Set checkpoint path here
HOLD_DIR     = "/nesi/nobackup/massey04767/CG_250m/test"
RESULTS_JSON = "eval_torchgeo_linz.json"
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
# METRICS + DIAGNOSTICS
# ============================================================
def compute_metrics(true_labels, predictions):
    TP = int(((predictions == 1) & (true_labels == 1)).sum())
    TN = int(((predictions == 0) & (true_labels == 0)).sum())
    FP = int(((predictions == 1) & (true_labels == 0)).sum())
    FN = int(((predictions == 0) & (true_labels == 1)).sum())
    total = len(true_labels)
    accuracy  = (TP + TN) / total if total else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "TP": TP, "TN": TN, "FP": FP, "FN": FN, "total": total}


def roc_auc(true_labels, scores):
    """Threshold-free separability via the Mann-Whitney rank formula (ties -> avg rank)."""
    n_pos = int((true_labels == 1).sum())
    n_neg = int((true_labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ss = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i, n = 0, len(scores)
    while i < n:
        j = i
        while j + 1 < n and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return (ranks[true_labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def threshold_sweep(true_labels, scores, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    return [{"threshold": float(t), **compute_metrics(true_labels, (scores >= t).astype(int))}
            for t in thresholds]


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

results = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
results.update({
    "n_no_damage": n_no_damage, "n_damaged": n_damaged,
    "unit": "250m_crop", "dataset": "LINZ_cyclone_gabrielle_hold",
    "model": MODEL_NAME, "roc_auc": round(float(auc), 4),
    "threshold_sweep": [{k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in r.items()} for r in sweep],
    "best_f1_operating_point": {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in best.items()},
    "false_negatives": false_negatives, "false_positives": false_positives,
})
with open(RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {RESULTS_JSON}")
