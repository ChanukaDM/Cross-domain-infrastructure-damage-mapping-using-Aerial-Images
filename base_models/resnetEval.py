# ============================================================
# IMPORTS
# ============================================================
import os
import json
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely import wkt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# CONFIG
# ============================================================
HOLD_IMAGES_DIR = "geotiffs/hold/images"
HOLD_LABELS_DIR = "geotiffs/hold/labels"
MODEL_PATH      = "buildings_fields50.pth"

CROP_SIZE       = 128                  # must match training
MIN_BUILDING_PX = 16                   # skip tiny buildings (same as preprocessing)
PADDING         = 10                   # context padding (same as preprocessing)
NUM_CLASSES     = 2
ADAPTER_DIM     = 128
BATCH_SIZE      = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# STEP 1: REBUILD THE MODEL — identical to training
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
model.eval()
print("Model loaded.\n")


# ============================================================
# STEP 2: HELPERS — crop reading and scaling (same as preprocessing)
# ============================================================
def scale_to_01(arr):
    arr = arr.astype(np.float32)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    return arr


def read_crop(tif_path, row_off, col_off, height, width):
    with rasterio.open(tif_path) as src:
        row_off = max(0, row_off)
        col_off = max(0, col_off)
        height  = min(height, src.height - row_off)
        width   = min(width,  src.width  - col_off)
        if height <= 0 or width <= 0:
            return np.zeros((3, 1, 1), dtype=np.float32)
        window = Window(col_off, row_off, width, height)
        arr = src.read(window=window)
    return arr[:3, :, :]


def resize_crop(arr, size):
    t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy()


# ============================================================
# STEP 3: BUILD A LIST OF BUILDING CROPS FROM THE HOLD SET
# ------------------------------------------------------------
# For every building in every hold label file, we record:
#   the pre crop, the post crop, and the true label.
# This mirrors exactly what the preprocessing script did,
# so the model is tested the same way it was trained.
# ============================================================
def build_crop_list(images_dir, labels_dir):
    crops = []     # each item: (pre_6ch_tensor_inputs..., label) — we store the stacked tensor + label
    label_files = [f for f in os.listdir(labels_dir) if f.endswith("_post_disaster.json")]
    print(f"Processing {len(label_files)} hold label files...")

    for label_filename in label_files:
        base     = label_filename.replace("_post_disaster.json", "")
        post_tif = os.path.join(images_dir, base + "_post_disaster.tif")
        pre_tif  = os.path.join(images_dir, base + "_pre_disaster.tif")
        if not (os.path.exists(post_tif) and os.path.exists(pre_tif)):
            continue

        with open(os.path.join(labels_dir, label_filename), "r") as f:
            label_data = json.load(f)

        for building in label_data["features"]["xy"]:
            subtype = building["properties"].get("subtype", "no-damage")
            if subtype == "un-classified":
                continue
            label = 1 if subtype in ["minor-damage", "major-damage", "destroyed"] else 0

            try:
                geom = wkt.loads(building["wkt"])
            except Exception:
                continue

            min_x, min_y, max_x, max_y = geom.bounds
            col_min, row_min = int(min_x) - PADDING, int(min_y) - PADDING
            col_max, row_max = int(max_x) + PADDING, int(max_y) + PADDING

            crop_h = row_max - row_min
            crop_w = col_max - col_min
            if crop_h < MIN_BUILDING_PX or crop_w < MIN_BUILDING_PX:
                continue

            post_crop = read_crop(post_tif, row_min, col_min, crop_h, crop_w)
            pre_crop  = read_crop(pre_tif,  row_min, col_min, crop_h, crop_w)
            if post_crop.size <= 3 or pre_crop.size <= 3:
                continue

            # scale, resize, stack -> 6-channel tensor
            pre_resized  = resize_crop(scale_to_01(pre_crop),  CROP_SIZE)
            post_resized = resize_crop(scale_to_01(post_crop), CROP_SIZE)
            combined = np.concatenate([pre_resized, post_resized], axis=0)  # [6, 128, 128]

            crops.append((combined, label))

    return crops


# ============================================================
# STEP 4: RUN THE MODEL ON ALL CROPS
# ============================================================
def evaluate(crops):
    true_labels = []
    predictions = []

    for start in range(0, len(crops), BATCH_SIZE):
        batch = crops[start : start + BATCH_SIZE]
        batch_tensors = torch.stack([torch.from_numpy(c[0]) for c in batch]).to(DEVICE)
        batch_labels  = [c[1] for c in batch]

        with torch.no_grad():
            outputs = model(batch_tensors)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy().tolist())
            true_labels.extend(batch_labels)

    return np.array(true_labels), np.array(predictions)


# ============================================================
# STEP 5: COMPUTE METRICS — same formulas as before
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


def print_report(m, n_no_damage, n_damaged):
    print("\n" + "="*55)
    print("  EVALUATION REPORT (per building)")
    print("="*55)
    print(f"\n  Buildings tested : {m['total']}")
    print(f"    no_damage      : {n_no_damage}")
    print(f"    damaged        : {n_damaged}")
    print(f"\n  Metrics:")
    print(f"    Accuracy  : {m['accuracy']:.4f}  ({m['accuracy']*100:.1f}%)")
    print(f"    Precision : {m['precision']:.4f}")
    print(f"    Recall    : {m['recall']:.4f}   <- of all damaged buildings, how many caught")
    print(f"    F1 Score  : {m['f1']:.4f}")
    print(f"\n  Confusion matrix:")
    print(f"                       Predicted")
    print(f"                   no_damage | damaged")
    print(f"    Actual no_dmg [  {m['TN']:6d}  |  {m['FP']:6d} ]")
    print(f"    Actual damaged[  {m['FN']:6d}  |  {m['TP']:6d} ]")
    print(f"\n  Plain english:")
    print(f"    {m['TN']:6d} fine buildings correctly called fine")
    print(f"    {m['TP']:6d} damaged buildings correctly caught")
    print(f"    {m['FP']:6d} false alarms (fine building called damaged)")
    print(f"    {m['FN']:6d} MISSED damage (damaged building called fine)")
    if m['TP'] == 0:
        print("\n  *** WARNING: model never predicted 'damaged' ***")
    print("="*55)


# ============================================================
# STEP 6: RUN
# ============================================================
print("--- Cropping buildings from hold set ---")
crops = build_crop_list(HOLD_IMAGES_DIR, HOLD_LABELS_DIR)
print(f"Extracted {len(crops)} building crops.\n")

if len(crops) == 0:
    print("ERROR: No crops found. Check paths and that hold labels exist.")
    exit()

n_no_damage = sum(1 for _, label in crops if label == 0)
n_damaged   = sum(1 for _, label in crops if label == 1)

print("--- Running model ---")
true_labels, predictions = evaluate(crops)

metrics = compute_metrics(true_labels, predictions)
print_report(metrics, n_no_damage, n_damaged)

# save results
results = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
results["n_no_damage"] = n_no_damage
results["n_damaged"]   = n_damaged
with open("eval_Test.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to eval_Test.json")