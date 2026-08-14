# ============================================================
# EVALUATE THE DENSENET ON LINZ CYCLONE GABRIELLE 250 m CROPS
# ------------------------------------------------------------
# Unlike the xBD eval, the LINZ images are ALREADY 250 m x 250 m crops
# and the damage label comes from the FOLDER they live in — there are
# no JSON labels to parse and no cropping/clustering to do.
#
# Layout expected:
#     CG_250m/
#       damaged/   pre/<name>.tif   post/<name>.tif   -> label 1
#       no_damage/ pre/<name>.tif   post/<name>.tif   -> label 0
#
# pre and post share the same filename. Note the LINZ tiffs are
# EPSG:2193 (metres) uint8 and pre/post can be different pixel sizes
# (e.g. 833 vs 2500) — that's fine, every crop is min-max stretched and
# resized to OUTPUT_SIZE before being stacked, exactly like training.
# ============================================================
import os
import json
import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from xbd_crops_common import (
    build_densenet, scale_to_uint8, resize_crop, OUTPUT_SIZE,
)


# ============================================================
# CONFIG
# ============================================================
LINZ_DIR     = "./CG_250m/hold"
MODEL_PATH   = "./base_models/checkpoints/densenet121_linz_new.pth"
RESULTS_JSON = "eval_densenet_linz.json"

# folder name -> label
CLASS_LABELS = {"no_damage": 0, "damaged": 1}

MAX_PER_CLASS = None   # set to an int to evaluate only the first N pairs/class (quick test)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# STEP 1: REBUILD THE MODEL — shared builder (logits head)
# ============================================================
model = build_densenet().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded.\n")


# ============================================================
# STEP 2: LOAD A LINZ PRE/POST PAIR AS A 6-CHANNEL TENSOR
# ------------------------------------------------------------
# Same normalization as training (xbd_crops_common.crop_to_6ch): read the
# 3-band crop, per-image min-max stretch to uint8, resize to OUTPUT_SIZE,
# scale to 0..1, then stack pre(3) + post(3) -> [6, S, S].
# ============================================================
def read_rgb(tif_path):
    with rasterio.open(tif_path) as src:
        arr = src.read()                # [bands, H, W]
    return arr[:3, :, :]                 # keep 3 bands


def pair_to_6ch(pre_tif, post_tif):
    pre  = read_rgb(pre_tif)
    post = read_rgb(post_tif)
    if pre.size == 0 or post.size == 0:
        return None
    pre_resized  = resize_crop(scale_to_uint8(pre),  OUTPUT_SIZE).astype(np.float32) / 255.0
    post_resized = resize_crop(scale_to_uint8(post), OUTPUT_SIZE).astype(np.float32) / 255.0
    return np.concatenate([pre_resized, post_resized], axis=0)   # [6, S, S]


# ============================================================
# STEP 3: BUILD THE TEST SET FROM THE FOLDER TREE
# ============================================================
def build_crop_list(linz_dir):
    crops = []     # each: (combined_6ch_float01, label)
    for class_name, label in CLASS_LABELS.items():
        pre_dir  = os.path.join(linz_dir, class_name, "pre")
        post_dir = os.path.join(linz_dir, class_name, "post")
        if not (os.path.isdir(pre_dir) and os.path.isdir(post_dir)):
            print(f"  WARNING: missing pre/post folder for '{class_name}', skipping.")
            continue

        # only keep filenames present in BOTH pre and post
        pre_files  = {f for f in os.listdir(pre_dir)  if f.lower().endswith(".tif")}
        post_files = {f for f in os.listdir(post_dir) if f.lower().endswith(".tif")}
        names = sorted(pre_files & post_files)
        missing = (pre_files ^ post_files)
        if missing:
            print(f"  NOTE: {len(missing)} unpaired file(s) in '{class_name}' ignored.")

        if MAX_PER_CLASS is not None:
            names = names[:MAX_PER_CLASS]

        kept = 0
        for name in names:
            combined = pair_to_6ch(os.path.join(pre_dir, name),
                                   os.path.join(post_dir, name))
            if combined is not None:
                crops.append((combined, label, f"{class_name}/{name}"))
                kept += 1
        print(f"  {class_name:9s}: {kept} pairs (label {label})")

    return crops


# ============================================================
# STEP 4: RUN THE MODEL — collect P(damaged) for every crop
# ------------------------------------------------------------
# We keep the raw softmax probability of the "damaged" class (index 1)
# rather than only the argmax decision, so we can sweep the decision
# threshold and compute AUC afterwards.
# ============================================================
def run_model(crops, batch_size=16):
    true_labels = []
    damaged_prob = []
    names = []

    for start in range(0, len(crops), batch_size):
        batch = crops[start : start + batch_size]
        batch_tensor = torch.from_numpy(
            np.stack([c[0] for c in batch], axis=0)).to(DEVICE)   # [B, 6, S, S]

        with torch.no_grad():
            logits = model(batch_tensor)
            probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(damaged)

        damaged_prob.extend(probs.tolist())
        true_labels.extend(c[1] for c in batch)
        names.extend(c[2] for c in batch)

        if (start // batch_size + 1) % 20 == 0:
            print(f"  ... {min(start + batch_size, len(crops))}/{len(crops)} crops",
                  flush=True)

    return np.array(true_labels), np.array(damaged_prob), names


# ============================================================
# STEP 5: COMPUTE METRICS
# ============================================================
def compute_metrics(true_labels, predictions):
    TP = int(((predictions == 1) & (true_labels == 1)).sum())
    TN = int(((predictions == 0) & (true_labels == 0)).sum())
    FP = int(((predictions == 1) & (true_labels == 0)).sum())
    FN = int(((predictions == 0) & (true_labels == 1)).sum())
    total = len(true_labels)

    accuracy  = (TP + TN) / total if total > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "TP": TP, "TN": TN, "FP": FP, "FN": FN, "total": total}


# ============================================================
# DIAGNOSTIC: ROC-AUC (threshold-free) via the Mann-Whitney rank formula
# ------------------------------------------------------------
# AUC = P(a random damaged crop scores higher than a random fine crop).
# 0.5 = useless features; ~1.0 = perfectly separable. This is the single
# number that says whether weak recall is a THRESHOLD problem (high AUC,
# just mis-thresholded) or a real FEATURE/domain gap (AUC near 0.5).
# Ties get average ranks so it matches sklearn's roc_auc_score.
# ============================================================
def roc_auc(true_labels, scores):
    n_pos = int((true_labels == 1).sum())
    n_neg = int((true_labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order  = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks  = np.empty(len(scores), dtype=np.float64)
    i, n = 0, len(scores)
    while i < n:                                   # assign average ranks to ties
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0   # 1-based average rank
        i = j + 1

    rank_sum_pos = ranks[true_labels == 1].sum()
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ============================================================
# DIAGNOSTIC: sweep the decision threshold, find best-F1 operating point
# ============================================================
def threshold_sweep(true_labels, scores, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = []
    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        m = compute_metrics(true_labels, preds)
        rows.append({"threshold": float(thr), **m})
    return rows


def print_sweep(rows):
    print("\n  Threshold sweep (P(damaged) >= thr):")
    print("    thr    acc    prec    rec     f1    TP  FP  FN  TN")
    for r in rows:
        print(f"    {r['threshold']:.2f}  {r['accuracy']:.3f}  {r['precision']:.3f}  "
              f"{r['recall']:.3f}  {r['f1']:.3f}  {r['TP']:3d} {r['FP']:3d} "
              f"{r['FN']:3d} {r['TN']:3d}")


def print_report(m, n_no_damage, n_damaged):
    print("\n" + "="*55)
    print("  LINZ CYCLONE GABRIELLE EVALUATION (per 250 m crop)")
    print("="*55)
    print(f"\n  Crops tested : {m['total']}")
    print(f"    no_damage  : {n_no_damage}")
    print(f"    damaged    : {n_damaged}")
    print(f"\n  Metrics:")
    print(f"    Accuracy  : {m['accuracy']:.4f}  ({m['accuracy']*100:.1f}%)")
    print(f"    Precision : {m['precision']:.4f}")
    print(f"    Recall    : {m['recall']:.4f}   <- of all damaged crops, how many caught")
    print(f"    F1 Score  : {m['f1']:.4f}")
    print(f"\n  Confusion matrix:")
    print(f"                       Predicted")
    print(f"                   no_damage | damaged")
    print(f"    Actual no_dmg [  {m['TN']:6d}  |  {m['FP']:6d} ]")
    print(f"    Actual damaged[  {m['FN']:6d}  |  {m['TP']:6d} ]")
    print(f"\n  Plain english:")
    print(f"    {m['TN']:6d} fine crops correctly called fine")
    print(f"    {m['TP']:6d} damaged crops correctly caught")
    print(f"    {m['FP']:6d} false alarms (fine crop called damaged)")
    print(f"    {m['FN']:6d} MISSED damage (damaged crop called fine)")
    if m['TP'] == 0:
        print("\n  *** WARNING: model never predicted 'damaged' ***")
    print("="*55)


# ============================================================
# STEP 6: RUN
# ============================================================
print("--- Building LINZ 250 m crop test set ---")
crops = build_crop_list(LINZ_DIR)
print(f"Extracted {len(crops)} crops.\n")

if len(crops) == 0:
    print("ERROR: No crops found. Check that CG_250m/<class>/{pre,post}/*.tif exist.")
    exit()

n_no_damage = sum(1 for c in crops if c[1] == 0)
n_damaged   = sum(1 for c in crops if c[1] == 1)

print("--- Running model ---")
true_labels, damaged_prob, names = run_model(crops)
names = np.array(names)

# ── default operating point (argmax == threshold 0.5) ──────────────
preds_default = (damaged_prob >= 0.5).astype(int)
metrics = compute_metrics(true_labels, preds_default)
print_report(metrics, n_no_damage, n_damaged)

# ── DIAGNOSTIC 1: threshold-free separability ──────────────────────
auc = roc_auc(true_labels, damaged_prob)
print(f"\n  ROC-AUC (threshold-free) : {auc:.4f}")
if auc >= 0.70:
    print("    -> features DO separate; weak recall is mostly a THRESHOLD problem.")
elif auc >= 0.60:
    print("    -> partial signal; threshold tuning helps but a real gap remains.")
else:
    print("    -> features barely separate; this is a genuine DOMAIN/FEATURE gap.")

# ── DIAGNOSTIC 2: threshold sweep + best-F1 operating point ────────
sweep = threshold_sweep(true_labels, damaged_prob)
print_sweep(sweep)
best = max(sweep, key=lambda r: r["f1"])
print(f"\n  Best-F1 threshold : {best['threshold']:.2f}  "
      f"(F1={best['f1']:.3f}, recall={best['recall']:.3f}, precision={best['precision']:.3f})")

# ── DIAGNOSTIC 3: which crops the model gets wrong (at thr=0.5) ─────
fn_mask = (true_labels == 1) & (preds_default == 0)   # missed damage
fp_mask = (true_labels == 0) & (preds_default == 1)   # false alarms
false_negatives = [{"crop": n, "p_damaged": round(float(p), 4)}
                   for n, p in sorted(zip(names[fn_mask], damaged_prob[fn_mask]),
                                      key=lambda x: x[1])]
false_positives = [{"crop": n, "p_damaged": round(float(p), 4)}
                   for n, p in sorted(zip(names[fp_mask], damaged_prob[fp_mask]),
                                      key=lambda x: -x[1])]
print(f"\n  MISSED damage (false negatives, {len(false_negatives)}) "
      f"— lowest P(damaged) first:")
for fn in false_negatives:
    print(f"    {fn['p_damaged']:.3f}  {fn['crop']}")
print(f"\n  False alarms (false positives, {len(false_positives)}):")
for fp in false_positives:
    print(f"    {fp['p_damaged']:.3f}  {fp['crop']}")

# ── save everything ────────────────────────────────────────────────
results = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
results["n_no_damage"]     = n_no_damage
results["n_damaged"]       = n_damaged
results["unit"]            = "250m_crop"
results["dataset"]         = "LINZ_cyclone_gabrielle"
results["roc_auc"]         = round(float(auc), 4)
results["threshold_sweep"] = [{k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in r.items()} for r in sweep]
results["best_f1_operating_point"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                      for k, v in best.items()}
results["false_negatives"] = false_negatives
results["false_positives"] = false_positives
with open(RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {RESULTS_JSON}")
