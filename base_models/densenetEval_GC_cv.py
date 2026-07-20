# ============================================================
# EVALUATE THE 5-FOLD-CV DENSENET121 (xBD) ON THE LINZ TEST SPLIT
# ------------------------------------------------------------
# Counterpart to torchGeoEval_linz.py, but for the adapter-DenseNet121 and
# WITHOUT any torchgeo dependency (DenseNet runs in an env that has no
# torchgeo). Loads the best-fold checkpoint written by
# densenet_finetune_linz_cv.py and scores it on CG_250m/test (disjoint
# from the train+hold pool used for CV), then writes the full set of
# classification-evaluation plots to PLOTS_DIR:
#   * confusion matrices (counts + normalized) @ 0.5 and the operating point
#   * ROC, Precision-Recall, metric-vs-threshold curves
#   * score histogram by class, calibration (reliability), metric bars
#   * training/validation loss curves (read from the checkpoint histories)
#
# The operating threshold is the ROC-optimal (Youden) cutoff the trainer
# stored in the checkpoint, so the headline numbers aren't tuned on the test set.
# ============================================================
import os
import json
import numpy as np
import torch
import torch.nn.functional as F

from xbd_crops_common import (
    build_densenet, pair_to_6ch, build_labelled_pairs, OUTPUT_SIZE,
    compute_metrics, roc_auc, threshold_sweep, fbeta, best_threshold_youden,
    average_precision, plot_confusion, plot_eval_curves, plot_training_curves,
)
import sys
from pathlib import Path

# 1. Get the absolute path of the root directory
# "Path(__file__).resolve()" finds working_file.py, ".parents[1]" steps up 2 levels to my_project/
root_path = Path(__file__).resolve().parents[1]

# 2. Append the root path to Python's system path if it isn't already there
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

# 3. Import your function normally
from gradcam_utils import run_gradcam


# ============================================================
# CONFIG
# ============================================================
MODEL_TAG    = "densenet121_xbd_kd"
MODEL_PATH   = "/home/c/cpehesar/research/checkpoints/kd_ckpts/densenet121_xbd_kd_best.pth"   # best-fold CV checkpoint
TEST_DIR     = "/home/c/cpehesar/research/CG_250m/test"
RESULTS_JSON = "/home/c/cpehesar/research/eval_densenet_linz_cv.json"
PLOTS_DIR    = "bm_plots"
FBETA        = 2.0     # recall-weighted operating point (missed damage costs more)
BATCH_SIZE   = 16
GRADCAM_DIR  = "gradcam_overlays"  # directory to save Grad-CAM overlays


# Grad-CAM: list the test crop names (as "class/filename.tif") you want an
# overlay for; [] skips it. Only the overlay PNG is written, into PLOTS_DIR.
GRADCAM_IMAGES = ["damaged/1319_x1903053_y5572585.tif",
                  "damaged/1764_x1907553_y5628585.tif",
                  "damaged/2226_x1911553_y5626585.tif",
                  "damaged/4548_x1923053_y5620585.tif",
                  "damaged/8230_x1985053_y5685085.tif",
                  "damaged/8186_x1984553_y5685585.tif",
                  "no_damage/658_x1897553_y5568585.tif",
                  "no_damage/816_x1899053_y5568085.tif",
                  "no_damage/1255_x1902553_y5575585.tif",
                  "no_damage/2013_x1909553_y5577085.tif",
                  "no_damage/4554_x1923053_y5617585.tif",
                  "no_damage/766_x1898553_y5567085.tif"]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# LOAD MODEL + CHECKPOINT METADATA
# ------------------------------------------------------------
# The CV checkpoint is a dict {state_dict, operating_threshold, threshold_rule,
# train_histories, ...}; a bare state_dict (older runs) is also supported.
# ============================================================
if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Checkpoint not found: {MODEL_PATH}")

ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    state_dict   = ckpt["state_dict"]
    CKPT_THRESH  = ckpt.get("operating_threshold")
    CKPT_RULE    = ckpt.get("threshold_rule", "youden_j")
    HISTORIES    = ckpt.get("train_histories")
else:
    state_dict, CKPT_THRESH, CKPT_RULE, HISTORIES = ckpt, None, None, None

model = build_densenet(pretrained=False).to(DEVICE)
model.load_state_dict(state_dict)
model.eval()
print(f"Loaded {MODEL_PATH} (operating_threshold={CKPT_THRESH}, rule={CKPT_RULE})\n")


# ============================================================
# BUILD TEST SET + RUN MODEL (keep P(damaged) for threshold-free analysis)
# ============================================================
def build_crop_list(test_dir):
    crops = []
    for pre_tif, post_tif, label, name in build_labelled_pairs(test_dir):
        t = pair_to_6ch(pre_tif, post_tif, size=OUTPUT_SIZE)
        if t is not None:
            crops.append((t, label, name))
    return crops


@torch.no_grad()
def run_model(crops):
    true_labels, damaged_prob, names = [], [], []
    for start in range(0, len(crops), BATCH_SIZE):
        batch = crops[start:start + BATCH_SIZE]
        x = torch.stack([c[0] for c in batch], dim=0).to(DEVICE)   # [B, 6, S, S]
        probs = F.softmax(model(x), dim=1)[:, 1].cpu().numpy()
        damaged_prob.extend(probs.tolist())
        true_labels.extend(c[1] for c in batch)
        names.extend(c[2] for c in batch)
    return np.array(true_labels), np.array(damaged_prob), np.array(names)


def print_report(m, n_no_damage, n_damaged, tag):
    print("\n" + "=" * 55)
    print(f"  DENSENET LINZ EVALUATION — {tag}")
    print("=" * 55)
    print(f"\n  Crops tested : {m['total']}  (no_damage: {n_no_damage}, damaged: {n_damaged})")
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
    print("=" * 55)


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
print("--- Building LINZ test set ---")
crops = build_crop_list(TEST_DIR)
print(f"Extracted {len(crops)} crops.\n")
if len(crops) == 0:
    raise SystemExit(f"No crops under {TEST_DIR}")

n_no_damage = sum(1 for c in crops if c[1] == 0)
n_damaged   = sum(1 for c in crops if c[1] == 1)

print("--- Running model ---")
true_labels, damaged_prob, names = run_model(crops)

# default 0.5 operating point
preds_default = (damaged_prob >= 0.5).astype(int)
metrics = compute_metrics(true_labels, preds_default)
print_report(metrics, n_no_damage, n_damaged, "threshold 0.5")

auc = roc_auc(true_labels, damaged_prob)
ap  = average_precision(true_labels, damaged_prob)
print(f"\n  ROC-AUC : {auc:.4f}   Average precision : {ap:.4f}")

sweep = threshold_sweep(true_labels, damaged_prob)
print_sweep(sweep)
best = max(sweep, key=lambda r: r["f1"])
print(f"\n  Best-F1 threshold : {best['threshold']:.2f}  "
      f"(F1={best['f1']:.3f}, recall={best['recall']:.3f}, precision={best['precision']:.3f})")
best_fbeta = max(sweep, key=lambda r: fbeta(r["precision"], r["recall"], FBETA))
print(f"  Best-F{int(FBETA)} threshold : {best_fbeta['threshold']:.2f}  "
      f"(recall={best_fbeta['recall']:.3f}, precision={best_fbeta['precision']:.3f})")

# operating threshold: prefer the cutoff the trainer stored in the checkpoint,
# else fall back to a test-set Youden point (reported as test-fit).
youden_thr, _, _, _ = best_threshold_youden(true_labels, damaged_prob)
if CKPT_THRESH is not None:
    op_threshold = float(CKPT_THRESH); thr_src = f"ckpt {CKPT_RULE}"
else:
    op_threshold = float(youden_thr); thr_src = "Youden/test"
op_metrics = compute_metrics(true_labels, (damaged_prob >= op_threshold).astype(int))
print_report(op_metrics, n_no_damage, n_damaged, f"operating thr {op_threshold:.2f} ({thr_src})")

# ============================================================
# PLOTS
# ============================================================
print("\n--- Writing evaluation plots ---")
plot_confusion(true_labels, damaged_prob, 0.5, op_threshold, thr_src, MODEL_TAG, PLOTS_DIR)
plot_eval_curves(true_labels, damaged_prob, op_threshold, MODEL_TAG, PLOTS_DIR)
plot_training_curves(HISTORIES, MODEL_TAG, PLOTS_DIR)   # no-op if no histories in ckpt

# Grad-CAM overlays for the requested test crops (last DenseNet feature map)
run_gradcam(model, {c[2]: c[0] for c in crops}, GRADCAM_IMAGES,
            target_layer=model.features, layout="bchw",
            model_tag=MODEL_TAG, out_dir=GRADCAM_DIR, device=DEVICE)

# ============================================================
# FALSE NEGATIVES / POSITIVES (at the operating threshold)
# ============================================================
op_pred = (damaged_prob >= op_threshold).astype(int)
fn_mask = (true_labels == 1) & (op_pred == 0)
fp_mask = (true_labels == 0) & (op_pred == 1)
false_negatives = [{"crop": n, "p_damaged": round(float(p), 4)}
                   for n, p in sorted(zip(names[fn_mask], damaged_prob[fn_mask]),
                                      key=lambda x: x[1])]
false_positives = [{"crop": n, "p_damaged": round(float(p), 4)}
                   for n, p in sorted(zip(names[fp_mask], damaged_prob[fp_mask]),
                                      key=lambda x: -x[1])]
print(f"\n  MISSED damage (false negatives @ op, {len(false_negatives)}) — lowest P first:")
for fn in false_negatives:
    print(f"    {fn['p_damaged']:.3f}  {fn['crop']}")
print(f"\n  False alarms (false positives @ op, {len(false_positives)}):")
for fp in false_positives:
    print(f"    {fp['p_damaged']:.3f}  {fp['crop']}")


# ============================================================
# SAVE JSON
# ============================================================
def _round(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}

results = _round(metrics)
results.update({
    "n_no_damage": n_no_damage, "n_damaged": n_damaged,
    "unit": "250m_crop", "dataset": "LINZ_cyclone_gabrielle_test",
    "model": MODEL_TAG, "weights": os.path.basename(MODEL_PATH),
    "roc_auc": round(float(auc), 4), "average_precision": round(float(ap), 4),
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
