# ============================================================
# FINE-TUNE A TORCHGEO PRETRAINED MODEL DIRECTLY ON LINZ
# ------------------------------------------------------------
# Unlike the densenet path (ImageNet -> xBD -> LINZ), this goes
# straight from a remote-sensing pretrained backbone INTO LINZ:
#   pretrained ResNet50 (fMoW-RGB) -> fine-tune on CG_250m/train.
#
# Same recipe as the densenet LINZ finetune otherwise:
#   * pre/post GeoTIFF pairs, folder name = label
#   * tiny set, so each pair is expanded AUG_FACTOR x per epoch
#   * class-weighted loss to protect damaged-class recall
#
# Train data : CG_250m/train/<class>/{pre,post}/*.tif
# Held-out   : CG_250m/hold/...  (evaluate with torchGeoEval_linz.py)
# ============================================================
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchgeo_common_linz import (
    build_model, build_labelled_pairs, pair_to_6ch, augment_6ch,
    model_input_size, MODELS, NUM_CLASSES, _module_by_name,
    _first_of, _last_of,
    lora_target_modules,
)


# ============================================================
# CONFIG
# ============================================================
MODEL_NAME  = "resnet50_sentinel2_rgb"   # any key in torchgeo_common_linz.MODELS
TRAIN_MODE  = "linear_probe" #"linear_probe", "lora"
TRAIN_DIR   = "/nesi/nobackup/massey04767/CG_250m/train"
OUTPUT_PATH = f"torchgeo_{MODEL_NAME}_{TRAIN_MODE}_linz.pth"
SIZE        = model_input_size(MODEL_NAME)
FREEZE_BACKBONE = True   # train only the new 6-ch stem + head (False = full finetune)

AUG_FACTOR    = 5        # 1 clean + (AUG_FACTOR-1) augmented variants per pair, per epoch
BATCH_SIZE    = 16       # 256x256 6-ch inputs — lighter than the densenet 512s
EPOCHS        = 30
LEARNING_RATE = 1e-4     # training a fresh 6-ch stem + head (+layer4), so a touch higher
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# DATASET — cache base tensors once, expand AUG_FACTOR x
# ============================================================
class LinzPairDataset(Dataset):
    def __init__(self, pairs, aug_factor=AUG_FACTOR):
        self.labels     = [p[2] for p in pairs]
        self.aug_factor = aug_factor
        print("Caching base tensors...")
        self.base = [pair_to_6ch(pre, post, size=SIZE) for pre, post, *_ in pairs]

    def __len__(self):
        return len(self.base) * self.aug_factor

    def __getitem__(self, index):
        pair_idx = index // self.aug_factor
        variant  = index %  self.aug_factor
        t = self.base[pair_idx]
        if variant != 0:
            t = augment_6ch(t)
        return t, torch.tensor(self.labels[pair_idx], dtype=torch.long)


# ============================================================
# LOAD DATA + CLASS BALANCE
# ============================================================
pairs = build_labelled_pairs(TRAIN_DIR)
if len(pairs) == 0:
    raise SystemExit(f"No training pairs found under {TRAIN_DIR}")
n_no_damage = sum(1 for p in pairs if p[2] == 0)
n_damaged   = sum(1 for p in pairs if p[2] == 1)
print(f"Found {len(pairs)} train pairs  (no_damage: {n_no_damage}, damaged: {n_damaged})")
print(f"After {AUG_FACTOR}x augmentation: {len(pairs) * AUG_FACTOR} samples/epoch")

dataset = LinzPairDataset(pairs, aug_factor=AUG_FACTOR)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


# ============================================================
# MODEL — TorchGeo pretrained backbone, straight into LINZ
# ============================================================
model = build_model(MODEL_NAME, num_classes=NUM_CLASSES,
                    freeze_backbone=FREEZE_BACKBONE).to(DEVICE)

stem_name, _ = _first_of(model, nn.Conv2d)
head_name, _ = _last_of(model, nn.Linear)

if TRAIN_MODE == "lora":
    # Add LoRA adapters to the backbone while keeping the widened stem/head trainable.
    from peft import LoraConfig, get_peft_model

    target_modules = lora_target_modules(MODEL_NAME)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    base_model = model.base_model.model if hasattr(model, "base_model") else model
    for module_name in (stem_name, head_name):
        for param in _module_by_name(base_model, module_name).parameters():
            param.requires_grad = True

print(f"Model: {MODEL_NAME}  |  mode: {TRAIN_MODE}  |  weights: {MODELS[MODEL_NAME]['weights']}  |  input {SIZE}px")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params : {trainable:,} / {total:,} ({100*trainable/total:.1f}%)\n")


# ============================================================
# CLASS-WEIGHTED LOSS + OPTIMISER
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
# TRAIN
# ============================================================
print("\nFine-tuning Begins......")

#import sys; sys.exit(0)

start = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = correct = total_seen = 0
    correct_damaged = total_damaged = 0
    correct_no_damage = total_no_damage = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        total_seen += labels.size(0)
        correct += (predicted == labels).sum().item()
        dmg, nod = (labels == 1), (labels == 0)
        total_damaged     += dmg.sum().item()
        total_no_damage   += nod.sum().item()
        correct_damaged   += ((predicted == labels) & dmg).sum().item()
        correct_no_damage += ((predicted == labels) & nod).sum().item()

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Loss: {running_loss/len(loader):.4f} | "
          f"Acc: {100*correct/total_seen:.1f}% | "
          f"Recall damaged: {100*correct_damaged/max(total_damaged,1):.1f}% | "
          f"Recall no_dmg: {100*correct_no_damage/max(total_no_damage,1):.1f}%")

print(f"\nTotal fine-tuning time: {(time.time()-start)/60:.1f} minutes")


# ============================================================
# SAVE
# ============================================================
torch.save(model.state_dict(), OUTPUT_PATH)
print(f"\nDone! Saved as {OUTPUT_PATH}")
print("Evaluate on the held-out split with torchGeoEval_linz.py (CG_250m/hold).")
