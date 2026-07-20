# ============================================================
# IMPORTS
# ============================================================
import time
import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from xbd_crops_common import build_densenet, OUTPUT_SIZE


#xBD
# PRE_IMAGE_PATH  = "/nesi/nobackup/massey04767/geotiffs/test/images/hurricane-harvey_00000135_pre_disaster.tif"   # ~833 x 833 px
# POST_IMAGE_PATH = "/nesi/nobackup/massey04767/geotiffs/test/images/hurricane-harvey_00000135_post_disaster.tif"  # ~2500 x 2500 px

#LINZ

PRE_IMAGE_PATH  = "/nesi/nobackup/massey04767/data/pre/5916_x1930053_y5615585.tif"   # ~833 x 833 px
POST_IMAGE_PATH = "/nesi/nobackup/massey04767/data/post/5916_x1930053_y5615585.tif"  # ~2500 x 2500 px



MODEL_PATH      = "densenet121_linz.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# STEP 1: REBUILD THE MODEL — shared builder (logits head)
# ============================================================
model = build_densenet().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded.")


# ============================================================
# STEP 2: LOAD TIFF
# ------------------------------------------------------------
# Read the first 3 bands and min-max normalise to 0..1. For a single
# 250 m tile this per-image min-max equals the per-crop normalisation
# used at training time.
# ============================================================
def load_tiff(path):
    with rasterio.open(path) as src:
        arr = src.read()
    arr = arr.astype(np.float32)
    arr = arr[:3, :, :]
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    print(f"  Loaded: {path}  shape={arr.shape}")
    return arr


# ============================================================
# STEP 3: RESIZE
# ------------------------------------------------------------
# pre is ~833 x 833 px (0.3 m/px) and post is 2500 x 2500 px
# (0.1 m/px). Downscale post to pre's grid so the two line up,
# then resize both to OUTPUT_SIZE (512) to match the training crops.
# ============================================================
def resize_arr(arr, target_H, target_W):
    t = torch.from_numpy(arr).unsqueeze(0)   # [1, C, H, W]
    t = F.interpolate(t, size=(target_H, target_W),
                      mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy()


# ============================================================
# STEP 4: CLASSIFY THE WHOLE TILE
# ------------------------------------------------------------
# Stack pre + post into one 6-channel tensor at OUTPUT_SIZE and run a
# single forward pass. The model outputs raw logits, so we apply
# softmax here to get class probabilities.
# ============================================================
def classify(pre_arr, post_arr):
    # align post to pre's grid, then both to the training size
    _, pre_H, pre_W = pre_arr.shape
    post_arr = resize_arr(post_arr, pre_H, pre_W)
    pre_arr  = resize_arr(pre_arr,  OUTPUT_SIZE, OUTPUT_SIZE)
    post_arr = resize_arr(post_arr, OUTPUT_SIZE, OUTPUT_SIZE)

    combined = np.concatenate([pre_arr, post_arr], axis=0)        # [6, S, S]
    batch = torch.from_numpy(combined).unsqueeze(0).to(DEVICE)    # [1, 6, S, S]
    with torch.no_grad():
        logits = model(batch)
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # [2]
    return probs


# ============================================================
# RUN
# ============================================================
print("\n--- Loading images ---")
pre_arr  = load_tiff(PRE_IMAGE_PATH)
post_arr = load_tiff(POST_IMAGE_PATH)

start_time = time.time()
print("\n--- Running model ---")
probs = classify(pre_arr, post_arr)

prob_no_damage, prob_damaged = float(probs[0]), float(probs[1])
final_label = "DAMAGED" if prob_damaged > prob_no_damage else "NO DAMAGE"

print("\n" + "=" * 50)
print(f"  RESULT             : {final_label}")
print(f"  Damaged   prob     : {prob_damaged:.1%}")
print(f"  No-damage prob     : {prob_no_damage:.1%}")
print("=" * 50)
print(f"\nTotal inference time: {(time.time() - start_time)/60:.2f} minutes")
