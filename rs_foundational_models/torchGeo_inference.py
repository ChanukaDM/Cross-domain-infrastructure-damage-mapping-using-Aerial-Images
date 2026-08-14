# ============================================================
# INFERENCE — run the fine-tuned TorchGeo model on new LINZ pairs
# ------------------------------------------------------------
# Takes a folder of UNLABELLED pre/post GeoTIFF pairs and writes a
# per-crop damage prediction (P(damaged) + decision at a chosen
# threshold). No labels required.
#
# Expected input layout:
#     <INPUT_DIR>/pre/<name>.tif
#     <INPUT_DIR>/post/<name>.tif       (matching filenames)
#
# OPERATING_THRESHOLD: lower than 0.5 to favour recall (catching damage)
# over precision — pick it from the held-out threshold sweep printed by
# torchGeoEval_linz.py, NOT from this unlabelled data.
# ============================================================
import os
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F

from torchgeo_common_linz import (
    build_model, build_unlabelled_pairs, pair_to_6ch, model_input_size, NUM_CLASSES,
)


# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "resnet50_fmow_rgb"   # must match the fine-tuned checkpoint
INPUT_DIR  = "/nesi/nobackup/massey04767/CG_250m/hold"   # folder with pre/ and post/
MODEL_PATH = f"torchgeo_{MODEL_NAME}_linz.pth"
OUT_CSV    = "torchgeo_inference.csv"
OUT_JSON   = "torchgeo_inference.json"
SIZE       = model_input_size(MODEL_NAME)

OPERATING_THRESHOLD = 0.2     # decision cutoff on P(damaged); tune from eval sweep
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# MODEL
# ============================================================
model = build_model(MODEL_NAME, num_classes=NUM_CLASSES, freeze_backbone=False).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print(f"Model loaded from {MODEL_PATH}\n")


# ============================================================
# DISCOVER PAIRS
# ============================================================
pairs = build_unlabelled_pairs(INPUT_DIR)
if len(pairs) == 0:
    raise SystemExit(f"No pre/post pairs found under {INPUT_DIR} "
                     f"(need {INPUT_DIR}/pre/*.tif and {INPUT_DIR}/post/*.tif)")
print(f"Found {len(pairs)} pre/post pairs to score.")


# ============================================================
# RUN INFERENCE (batched)
# ============================================================
records = []
batch_tensors, batch_names = [], []


def flush(batch_tensors, batch_names):
    if not batch_tensors:
        return
    x = torch.stack(batch_tensors, dim=0).to(DEVICE)
    with torch.no_grad():
        p = F.softmax(model(x), dim=1)[:, 1].cpu().numpy()
    for name, prob in zip(batch_names, p):
        records.append({
            "crop": name,
            "p_damaged": round(float(prob), 4),
            "prediction": "damaged" if prob >= OPERATING_THRESHOLD else "no_damage",
        })


for pre_tif, post_tif, name in pairs:
    t = pair_to_6ch(pre_tif, post_tif, size=SIZE)
    if t is None:
        print(f"  skipped (unreadable): {name}")
        continue
    batch_tensors.append(t)
    batch_names.append(name)
    if len(batch_tensors) == BATCH_SIZE:
        flush(batch_tensors, batch_names)
        batch_tensors, batch_names = [], []
flush(batch_tensors, batch_names)


# ============================================================
# SUMMARISE + SAVE
# ============================================================
n_damaged = sum(1 for r in records if r["prediction"] == "damaged")
print(f"\nScored {len(records)} crops at threshold {OPERATING_THRESHOLD}:")
print(f"  predicted damaged   : {n_damaged}")
print(f"  predicted no_damage : {len(records) - n_damaged}")

records.sort(key=lambda r: -r["p_damaged"])

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["crop", "p_damaged", "prediction"])
    writer.writeheader()
    writer.writerows(records)

with open(OUT_JSON, "w") as f:
    json.dump({"threshold": OPERATING_THRESHOLD,
               "model": MODEL_NAME,
               "n_crops": len(records), "n_damaged": n_damaged,
               "predictions": records}, f, indent=2)

print(f"\nSaved {OUT_CSV} and {OUT_JSON} (sorted by P(damaged), highest first).")
