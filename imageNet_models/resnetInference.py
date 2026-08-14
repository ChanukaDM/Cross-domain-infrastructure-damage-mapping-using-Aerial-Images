# ============================================================
# IMPORTS
# ============================================================
import os
import json
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image              # NEW: needed to save PNG output images


# ============================================================
# CONFIG
# ============================================================
PRE_IMAGE_PATH  = "data/pre/5617_x1928553_y5607585.tif"
POST_IMAGE_PATH = "data/post/5617_x1928553_y5607585.tif"
MODEL_PATH      = "buildings_fields50.pth"
OUTPUT_DIR      = "inference_output"              # NEW: folder where all output files go

PATCH_SIZE  = 128
STRIDE      = 128
NUM_CLASSES = 2
ADAPTER_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)            # create output folder if it doesn't exist


# ============================================================
# STEP 1: REBUILD MODEL — identical to before
# ============================================================
class Adapter(nn.Module):
    def __init__(self, channels, adapter_dim):
        super().__init__()
        self.down = nn.Linear(channels, adapter_dim)
        self.act  = nn.ReLU()
        self.up   = nn.Linear(adapter_dim, channels)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x_perm = x.permute(0, 2, 3, 1)
        out = self.down(x_perm)
        out = self.act(out)
        out = self.up(out)
        out = out.permute(0, 3, 1, 2)
        return x + out


class BlockWithAdapter(nn.Module):
    def __init__(self, block, channels, adapter_dim):
        super().__init__()
        self.block   = block
        self.adapter = Adapter(channels, adapter_dim)

    def forward(self, x):
        x = self.block(x)
        x = self.adapter(x)
        return x


def build_model():
    model = models.resnet50(weights="IMAGENET1K_V2")

    # replace first conv to accept 6 channels
    old_first_layer = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=6,
        out_channels=old_first_layer.out_channels,
        kernel_size=old_first_layer.kernel_size,
        stride=old_first_layer.stride,
        padding=old_first_layer.padding,
        bias=False
    )

    # freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze the new first conv (must learn from scratch)
    for param in model.conv1.parameters():
        param.requires_grad = True

    # replace final layer (unfrozen by default)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, NUM_CLASSES),
        nn.Softmax(dim=1)
    )

    # attach adapters to every block
    layer_channel_map = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
    for layer_name, channels in layer_channel_map.items():
        layer = getattr(model, layer_name)
        new_blocks = nn.Sequential()
        for i, block in enumerate(layer):
            wrapped = BlockWithAdapter(block, channels, ADAPTER_DIM)
            new_blocks.add_module(str(i), wrapped)
        setattr(model, layer_name, new_blocks)

    return model


model = build_model()
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.to(torch.float16)
model.eval()
print("Model loaded.")


# ============================================================
# STEP 2: LOAD TIFF — identical to before
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
# STEP 3: ALIGN — identical to before
# ============================================================
def align_images(pre_arr, post_arr):
    _, pre_H, pre_W   = pre_arr.shape
    _, post_H, post_W = post_arr.shape
    if pre_H == post_H and pre_W == post_W:
        return pre_arr, post_arr
    target_H, target_W = max(pre_H, post_H), max(pre_W, post_W)
    print(f"  Aligning: pre {pre_H}x{pre_W} → {target_H}x{target_W}")
    if pre_H < post_H or pre_W < post_W:
        t = torch.from_numpy(pre_arr).unsqueeze(0)
        t = F.interpolate(t, size=(target_H, target_W), mode="bilinear", align_corners=False)
        pre_arr = t.squeeze(0).numpy()
    else:
        t = torch.from_numpy(post_arr).unsqueeze(0)
        t = F.interpolate(t, size=(target_H, target_W), mode="bilinear", align_corners=False)
        post_arr = t.squeeze(0).numpy()
    return pre_arr, post_arr


# ============================================================
# STEP 4: EXTRACT PATCHES — identical to before
# ============================================================
def extract_patches(pre_arr, post_arr, patch_size, stride):
    _, H, W = pre_arr.shape
    patches, positions = [], []

    for row in range(0, H - patch_size + 1, stride):
        for col in range(0, W - patch_size + 1, stride):
            patches.append((pre_arr[:, row:row+patch_size, col:col+patch_size],
                            post_arr[:, row:row+patch_size, col:col+patch_size]))
            positions.append((row, col))

    for row in range(0, H - patch_size + 1, stride):   # right edge
        col = W - patch_size
        patches.append((pre_arr[:, row:row+patch_size, col:col+patch_size],
                        post_arr[:, row:row+patch_size, col:col+patch_size]))
        positions.append((row, col))

    for col in range(0, W - patch_size + 1, stride):   # bottom edge
        row = H - patch_size
        patches.append((pre_arr[:, row:row+patch_size, col:col+patch_size],
                        post_arr[:, row:row+patch_size, col:col+patch_size]))
        positions.append((row, col))

    row, col = H - patch_size, W - patch_size           # bottom-right corner
    patches.append((pre_arr[:, row:row+patch_size, col:col+patch_size],
                    post_arr[:, row:row+patch_size, col:col+patch_size]))
    positions.append((row, col))

    print(f"  Extracted {len(patches)} patches from {H}x{W}  (stride={stride})")
    return patches, positions


# ============================================================
# STEP 5: RUN MODEL — identical to before
# ============================================================
def run_patches(model, patches, batch_size=32):
    all_probs = []
    for start in range(0, len(patches), batch_size):
        batch_pairs = patches[start : start + batch_size]
        batch_tensors = []
        for pre_patch, post_patch in batch_pairs:
            combined = torch.cat([torch.from_numpy(pre_patch),
                                  torch.from_numpy(post_patch)], dim=0)
            batch_tensors.append(combined)
        batch = torch.stack(batch_tensors).to(DEVICE)
        with torch.no_grad():
            outputs = model(batch)
            probs   = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)            # shape [N, 2]


# ============================================================
# STEP 6: BUILD THE DAMAGE HEATMAP  ← NEW
# ------------------------------------------------------------
# We have a (row, col) position and a damaged-probability for
# every patch. We paint those probabilities onto a canvas the
# same size as the image.
#
# Because patches overlap (stride < patch_size), multiple patches
# cover the same pixel. We accumulate all their scores and then
# divide by the count — this gives a smooth average at each pixel
# rather than blocky hard edges.
# ============================================================
def build_heatmap(positions, all_probs, image_H, image_W, patch_size):
    # score_map  : running sum of damaged-probability at each pixel
    # count_map  : how many patches have covered each pixel
    score_map = np.zeros((image_H, image_W), dtype=np.float32)
    count_map = np.zeros((image_H, image_W), dtype=np.float32)

    for (row, col), prob in zip(positions, all_probs):
        damaged_prob = prob[1]                          # index 1 = "damaged" probability
        # add this patch's score to every pixel it covers
        score_map[row:row+patch_size, col:col+patch_size] += damaged_prob
        count_map[row:row+patch_size, col:col+patch_size] += 1.0

    # avoid dividing by zero for any pixel no patch touched
    count_map = np.where(count_map == 0, 1, count_map)

    # final heatmap: average damaged-probability at each pixel, values 0..1
    heatmap = score_map / count_map                    # shape [H, W], float 0..1
    return heatmap


# ============================================================
# STEP 7: COLOUR THE HEATMAP  ← NEW
# ------------------------------------------------------------
# Convert the 0..1 probability map into an RGB colour image.
# Low score  → green  (safe)
# High score → red    (damaged)
# We blend between green and red linearly by the damage score.
# ============================================================
def colorize_heatmap(heatmap):
    H, W = heatmap.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)   # empty RGB canvas

    # green channel decreases as damage increases  (255 → 0)
    rgb[:, :, 1] = ((1.0 - heatmap) * 255).astype(np.uint8)   # G

    # red channel increases as damage increases    (0 → 255)
    rgb[:, :, 0] = (heatmap * 255).astype(np.uint8)            # R

    # blue stays 0 throughout
    return rgb                                   # shape [H, W, 3], uint8 RGB


# ============================================================
# STEP 8: SAVE ALL OUTPUTS  ← NEW
# ------------------------------------------------------------
# We save three things:
#   1. heatmap.png          — the colour overlay on its own
#   2. overlay.png          — heatmap blended on top of the post image
#   3. patch_results.csv    — every patch's index, position, score, label
#      so you can query "what is patch 412?" easily
# ============================================================
def save_outputs(heatmap, color_map, post_arr, positions, all_probs, output_dir):

    # --- 8a. Save the standalone heatmap ---
    heatmap_img = Image.fromarray(color_map)           # color_map is already [H,W,3] uint8
    heatmap_path = os.path.join(output_dir, "heatmap.png")
    heatmap_img.save(heatmap_path)
    print(f"  Saved heatmap:  {heatmap_path}")

    # --- 8b. Save the heatmap blended over the post image ---
    # convert post_arr (float 0..1, shape [3,H,W]) to uint8 RGB (shape [H,W,3])
    post_uint8 = (post_arr.transpose(1, 2, 0) * 255).astype(np.uint8)  # [H, W, 3]

    # blend: 60% original image, 40% colour heatmap
    # this lets you see the real image underneath while the damage colours show on top
    alpha = 0.4
    blended = ((1 - alpha) * post_uint8.astype(np.float32)
               + alpha     * color_map.astype(np.float32)).astype(np.uint8)

    overlay_img  = Image.fromarray(blended)
    overlay_path = os.path.join(output_dir, "overlay.png")
    overlay_img.save(overlay_path)
    print(f"  Saved overlay:  {overlay_path}")

    # --- 8c. Save the per-patch CSV ---
    # Each row = one patch.
    # patch_index  : a simple number (0, 1, 2 ...) — use this to look up a specific patch
    # row, col     : top-left pixel coordinate of this patch in the full image
    # damaged_prob : the model's confidence that this patch is damaged (0.0 to 1.0)
    # label        : "damaged" if prob > 0.5, else "no_damage"



# ============================================================
# STEP 9: QUERY TOOL — look up any patch by index  ← NEW
# ------------------------------------------------------------
# After inference you can call this to investigate a specific patch.
# It prints the patch's position, score, and saves a small PNG crop
# of just that patch from the post image so you can visually inspect it.
# ============================================================


# ============================================================
# STEP 10: RUN THE FULL PIPELINE
# ============================================================
print("\n--- Loading images ---")
pre_arr  = load_tiff(PRE_IMAGE_PATH)
post_arr = load_tiff(POST_IMAGE_PATH)

print("\n--- Aligning ---")
pre_arr, post_arr = align_images(pre_arr, post_arr)

_, image_H, image_W = post_arr.shape              # final image dimensions after alignment

print("\n--- Extracting patches ---")
patches, positions = extract_patches(pre_arr, post_arr, PATCH_SIZE, STRIDE)

print("\n--- Running model ---")
all_probs = run_patches(model, patches, batch_size=32)

print("\n--- Building heatmap ---")
heatmap   = build_heatmap(positions, all_probs, image_H, image_W, PATCH_SIZE)
color_map = colorize_heatmap(heatmap)

print("\n--- Saving outputs ---")
save_outputs(heatmap, color_map, post_arr, positions, all_probs, OUTPUT_DIR)

# ── quick summary ──────────────────────────────────────────
damaged_probs    = all_probs[:, 1]
n_damaged        = (damaged_probs > 0.5).sum()
mean_prob        = damaged_probs.mean()
final_label      = "DAMAGED" if mean_prob > 0.5 else "NO DAMAGE"

print("\n" + "="*50)
print(f"  RESULT  : {final_label}")
print(f"  Mean damaged prob   : {mean_prob:.1%}")
print(f"  Patches damaged     : {n_damaged} / {len(patches)}")
print("="*50)

# ============================================================
# STEP 11: INSPECT SPECIFIC PATCHES  ← NEW
# ------------------------------------------------------------
# Now you have the CSV open, you can look up any patch by its
# patch_index number and call inspect_patch() to see it.
#
# Examples:
#   inspect_patch(0,   positions, all_probs, post_arr, OUTPUT_DIR)  # first patch
#   inspect_patch(412, positions, all_probs, post_arr, OUTPUT_DIR)  # patch 412
#   inspect_patch(999, positions, all_probs, post_arr, OUTPUT_DIR)  # any index
#
