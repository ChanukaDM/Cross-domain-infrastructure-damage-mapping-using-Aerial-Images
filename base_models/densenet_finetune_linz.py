# ============================================================
# FINE-TUNE THE EXISTING DENSENET ON LINZ CYCLONE GABRIELLE
# ------------------------------------------------------------
# Unlike the xBD trainer, this script:
#   1. LOADS the already-fine-tuned weights (densenet121.pth) and keeps
#      training FROM them (domain adaptation), instead of starting from
#      ImageNet.
#   2. Reads LINZ pre/post GeoTIFF pairs straight from CG_250m/ (no PNGs,
#      no JSON) — the folder name is the label.
#   3. The LINZ set is tiny (~80 pairs), so each pair is expanded
#      AUG_FACTOR times with geometry + photometric augmentation to give
#      the optimiser more variety per epoch.
#
# Layout expected:
#     CG_250m/
#       damaged/   pre/<name>.tif  post/<name>.tif   -> label 1
#       no_damage/ pre/<name>.tif  post/<name>.tif   -> label 0
#
# *** METHODOLOGY WARNING ***
# These same CG_250m crops are what densenetEval_linz.py evaluates on.
# If you fine-tune here on ALL of CG_250m and then "evaluate" on CG_250m
# you are testing on training data — the numbers will be meaningless.
# Hold out a disjoint test split (or k-fold) before trusting any score.
# ============================================================
import os
import time
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from xbd_crops_common import (
    build_densenet, scale_to_uint8, resize_crop, OUTPUT_SIZE,
)


# ============================================================
# CONFIG
# ============================================================
LINZ_DIR        = "/nesi/nobackup/massey04767/CG_250m/train"
PRETRAINED_PATH = "densenet121.pth"        # existing fine-tuned weights to start FROM
OUTPUT_PATH     = "densenet121_linz_new.pth"   # where the LINZ-adapted model is saved
                                           # (kept separate so we don't clobber the source)

AUG_FACTOR    = 5        # each pair -> 1 clean + (AUG_FACTOR-1) augmented variants per epoch
BATCH_SIZE    = 8        # 512x512 6-channel inputs are memory-heavy
EPOCHS        = 30       # small set + tiny LR: keep it short to avoid overfitting
LEARNING_RATE = 5e-5     # lower than the xBD run — we're nudging, not retraining
NUM_CLASSES   = 2        # 0 = no_damage, 1 = damaged
ADAPTER_DIM   = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# STEP 1: GATHER LINZ PRE/POST PAIRS (folder name = label)
# ============================================================
def build_pairs(linz_dir):
    pairs = []
    class_labels = {"no_damage": 0, "damaged": 1}
    for class_name, label in class_labels.items():
        pre_dir  = os.path.join(linz_dir, class_name, "pre")
        post_dir = os.path.join(linz_dir, class_name, "post")
        pre_files  = {f for f in os.listdir(pre_dir)  if f.lower().endswith(".tif")}
        post_files = {f for f in os.listdir(post_dir) if f.lower().endswith(".tif")}
        for name in sorted(pre_files & post_files):     # only filenames present in both
            pairs.append((os.path.join(pre_dir, name),
                          os.path.join(post_dir, name), label))
    return pairs


# ============================================================
# STEP 2: LOAD A PRE/POST PAIR AS A 6-CHANNEL TENSOR (0..1)
# ------------------------------------------------------------
# Same normalization as training/eval: per-image min-max stretch -> uint8,
# resize to OUTPUT_SIZE, scale to 0..1, stack pre(3)+post(3).
# ============================================================
def pair_to_6ch(pre_tif, post_tif):
    with rasterio.open(pre_tif) as s:
        pre = s.read()[:3]
    with rasterio.open(post_tif) as s:
        post = s.read()[:3]
    pre  = resize_crop(scale_to_uint8(pre),  OUTPUT_SIZE).astype(np.float32) / 255.0
    post = resize_crop(scale_to_uint8(post), OUTPUT_SIZE).astype(np.float32) / 255.0
    return torch.from_numpy(np.concatenate([pre, post], axis=0))   # [6, S, S]


# ============================================================
# STEP 3: AUGMENTATION — 5 different "looks" per pair
# ------------------------------------------------------------
# Geometric ops (flip / 90-rot) are applied IDENTICALLY to the pre and
# post halves so the two stay registered. Photometric jitter uses the
# SAME factor on both halves so the pre->post CHANGE signal is preserved.
# Nadir aerial imagery has no canonical up, so flips/rotations are label-
# preserving here.
# ============================================================
def augment_6ch(t):
    if torch.rand(1).item() < 0.5:                 # horizontal flip
        t = torch.flip(t, dims=[2])
    if torch.rand(1).item() < 0.5:                 # vertical flip
        t = torch.flip(t, dims=[1])
    k = int(torch.randint(0, 4, (1,)).item())      # 0/90/180/270 rotation
    if k:
        t = torch.rot90(t, k, dims=[1, 2])

    brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.4   # 0.8 .. 1.2
    t = t * brightness
    contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.4     # 0.8 .. 1.2
    mean = t.mean()
    t = (t - mean) * contrast + mean
    t = t + torch.randn_like(t) * 0.02             # mild sensor-like noise
    return t.clamp(0.0, 1.0)


# ============================================================
# STEP 4: DATASET — caches base tensors, expands AUG_FACTOR x
# ------------------------------------------------------------
# len = n_pairs * AUG_FACTOR. variant 0 is the clean image; variants
# 1..AUG_FACTOR-1 are randomly augmented (and re-randomised every epoch,
# so the model effectively sees even more than AUG_FACTOR variations).
# Base tensors are decoded once and held in RAM (~80 * 6MB) to avoid
# re-reading the tiffs thousands of times.
# ============================================================
class LinzPairDataset(Dataset):
    def __init__(self, pairs, aug_factor=AUG_FACTOR):
        self.labels     = [p[2] for p in pairs]
        self.aug_factor = aug_factor
        print("Caching base tensors...")
        self.base = [pair_to_6ch(pre, post) for pre, post, _ in pairs]

    def __len__(self):
        return len(self.base) * self.aug_factor

    def __getitem__(self, index):
        pair_idx = index // self.aug_factor
        variant  = index %  self.aug_factor
        t = self.base[pair_idx]
        if variant != 0:                           # variant 0 = clean original
            t = augment_6ch(t)
        label = torch.tensor(self.labels[pair_idx], dtype=torch.long)
        return t, label


# ============================================================
# STEP 5: LOAD DATA + REPORT CLASS BALANCE
# ============================================================
pairs = build_pairs(LINZ_DIR)
n_no_damage = sum(1 for *_, label in pairs if label == 0)
n_damaged   = sum(1 for *_, label in pairs if label == 1)
print(f"Found {len(pairs)} LINZ pairs  (no_damage: {n_no_damage}, damaged: {n_damaged})")
print(f"After {AUG_FACTOR}x augmentation: {len(pairs) * AUG_FACTOR} training samples/epoch")

dataset = LinzPairDataset(pairs, aug_factor=AUG_FACTOR)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


# ============================================================
# STEP 6: BUILD MODEL + LOAD THE EXISTING FINE-TUNED WEIGHTS
# ------------------------------------------------------------
# Same architecture as eval/train (shared builder). We then load the
# already-fine-tuned densenet121.pth so we continue FROM it instead of
# from ImageNet — this is the actual "fine-tune the existing model" step.
# ============================================================
model = build_densenet(num_classes=NUM_CLASSES, adapter_dim=ADAPTER_DIM).to(DEVICE)

if not os.path.exists(PRETRAINED_PATH):
    raise FileNotFoundError(
        f"Existing weights '{PRETRAINED_PATH}' not found — this script fine-tunes "
        f"an already-trained model, it does not train from scratch.")
state_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
print(f"Loaded existing fine-tuned weights from {PRETRAINED_PATH}")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,}\n")


# ============================================================
# STEP 7: CLASS-WEIGHTED LOSS (recall fix) + OPTIMISER
# ------------------------------------------------------------
# Weights come from the BASE class counts; AUG_FACTOR multiplies both
# classes equally so the ratio is unchanged.
# ============================================================
total_samples    = n_no_damage + n_damaged
weight_no_damage = total_samples / (2 * max(n_no_damage, 1))
weight_damaged   = total_samples / (2 * max(n_damaged,   1))
class_weights = torch.tensor([weight_no_damage, weight_damaged],
                             dtype=torch.float32).to(DEVICE)
print(f"Class weights -> no_damage: {weight_no_damage:.2f}, damaged: {weight_damaged:.2f}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=LEARNING_RATE)


# ============================================================
# STEP 8: TRAINING LOOP
# ============================================================
print("\nFine-tuning Begins......")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = total_seen = 0
    correct_damaged = total_damaged = 0
    correct_no_damage = total_no_damage = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        total_seen += labels.size(0)
        correct += (predicted == labels).sum().item()

        damaged_mask   = (labels == 1)
        no_damage_mask = (labels == 0)
        total_damaged     += damaged_mask.sum().item()
        total_no_damage   += no_damage_mask.sum().item()
        correct_damaged   += ((predicted == labels) & damaged_mask).sum().item()
        correct_no_damage += ((predicted == labels) & no_damage_mask).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc  = 100 * correct / total_seen
    recall_damaged   = 100 * correct_damaged   / max(total_damaged, 1)
    recall_no_damage = 100 * correct_no_damage / max(total_no_damage, 1)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Loss: {epoch_loss:.4f} | "
          f"Acc: {epoch_acc:.1f}% | "
          f"Recall damaged: {recall_damaged:.1f}% | "
          f"Recall no_dmg: {recall_no_damage:.1f}%")

elapsed = time.time() - start_time
print(f"\nTotal fine-tuning time: {elapsed/60:.1f} minutes")


# ============================================================
# STEP 9: SAVE
# ============================================================
torch.save(model.state_dict(), OUTPUT_PATH)
print(f"\nFine-tuning done! Saved as {OUTPUT_PATH}")
print("Remember: evaluate on a HELD-OUT split, not the crops you trained on.")
