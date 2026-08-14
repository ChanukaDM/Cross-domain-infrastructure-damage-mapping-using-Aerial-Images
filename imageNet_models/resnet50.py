# ============================================================
# IMPORTS
# ============================================================
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image


# ============================================================
# CONFIG
# ============================================================
CROPS_DIR     = "building_crops"      # folder with damaged/ and no_damage/ subfolders
CROP_SIZE     = 128                   # size each crop was saved at (must match preprocessing)
BATCH_SIZE    = 32                   # reduced for ResNet50's higher memory footprint
EPOCHS        = 50                   # more epochs is fine now since crops are small/fast
LEARNING_RATE = 0.0001
NUM_CLASSES   = 2                     # 0 = no_damage, 1 = damaged
ADAPTER_DIM   = 128                  # wider bottleneck to handle ResNet50's larger feature maps
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# STEP 1: BUILD FILE LIST FROM FOLDERS
# ------------------------------------------------------------
# No more JSON parsing. The crops are already sorted into folders,
# so the folder name IS the label.
#   building_crops/no_damage/*.png  -> label 0
#   building_crops/damaged/*.png    -> label 1
# ============================================================
def build_file_list(crops_dir):
    file_list = []
    class_folders = {"no_damage": 0, "damaged": 1}   # folder name -> label
    for folder_name, label in class_folders.items():
        folder_path = os.path.join(crops_dir, folder_name)
        for filename in os.listdir(folder_path):     # loop over every crop in the folder
            if not filename.endswith(".png"):
                continue
            full_path = os.path.join(folder_path, filename)
            file_list.append((full_path, label))
    return file_list[:]


# ============================================================
# STEP 2: DATASET
# ------------------------------------------------------------
# Each saved file is a side-by-side PNG: pre on the LEFT half,
# post on the RIGHT half. We split it back into two images,
# then stack them into a 6-channel tensor.
# ============================================================
class CropDataset(Dataset):
    def __init__(self, file_list, augment=False):
        self.data = file_list
        self.augment = augment           # whether to apply random flips (helps generalization)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, label = self.data[index]

        # open the side-by-side PNG
        combined_img = Image.open(path).convert("RGB")   # width = CROP_SIZE*2, height = CROP_SIZE
        combined_arr = np.array(combined_img)            # shape [H, W, 3], uint8

        # split into pre (left half) and post (right half)
        pre_arr  = combined_arr[:, :CROP_SIZE, :]        # [H, CROP_SIZE, 3]
        post_arr = combined_arr[:, CROP_SIZE:, :]        # [H, CROP_SIZE, 3]

        # convert from [H, W, 3] to [3, H, W] and scale 0..255 -> 0..1
        pre_arr  = pre_arr.transpose(2, 0, 1).astype(np.float32)  / 255.0
        post_arr = post_arr.transpose(2, 0, 1).astype(np.float32) / 255.0

        pre_t  = torch.from_numpy(pre_arr)
        post_t = torch.from_numpy(post_arr)

        # --- optional data augmentation: random horizontal flip ---
        # flipping both images the same way teaches the model that a building
        # is "damaged" regardless of its orientation. Cheap way to get more variety.
        if self.augment and torch.rand(1).item() > 0.5:
            pre_t  = torch.flip(pre_t,  dims=[2])   # flip left-right
            post_t = torch.flip(post_t, dims=[2])

        #One hot encode the label
        label = torch.tensor(label, dtype=torch.long)
        label = F.one_hot(label, num_classes=NUM_CLASSES).float()

        # stack pre + post -> 6 channels
        combined = torch.cat([pre_t, post_t], dim=0)     # [6, CROP_SIZE, CROP_SIZE]

        return combined, label


# ============================================================
# STEP 3: LOAD DATA + REPORT CLASS BALANCE
# ------------------------------------------------------------
# We count how many of each class we have. This number is used
# later to set class weights, which fixes the recall problem.
# ============================================================
all_files = build_file_list(CROPS_DIR)
print(f"Found {len(all_files)} building crops.")

# count each class
n_no_damage = sum(1 for _, label in all_files if label == 0)
n_damaged   = sum(1 for _, label in all_files if label == 1)
print(f"  no_damage : {n_no_damage}")
print(f"  damaged   : {n_damaged}")
print(f"  ratio     : {n_no_damage / max(n_damaged,1):.1f} no_damage per damaged")

dataset = CropDataset(all_files, augment=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


# ============================================================
# STEP 4: ADAPTER MODULES — identical to before
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


# ============================================================
# STEP 5: BUILD THE MODEL — identical structure to before
# ============================================================
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

model = model.to(DEVICE)

# parameter count
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"\nTrainable params: {trainable:,} / {total:,}\n")


# ============================================================
# STEP 6: LOSS WITH CLASS WEIGHTS  ← THE KEY FIX FOR RECALL
# ------------------------------------------------------------
# Because there are far more no_damage crops than damaged ones,
# the model is tempted to just always guess "no_damage".
#
# Class weighting tells the loss function: "mistakes on the rare
# 'damaged' class cost MORE." This forces the model to pay
# attention to damaged buildings instead of ignoring them.
#
# The weight for each class is set to: total / (num_classes * class_count)
# This gives the rare class a bigger weight automatically.
# ============================================================
total_samples = n_no_damage + n_damaged
weight_no_damage = total_samples / (2 * max(n_no_damage, 1))
weight_damaged   = total_samples / (2 * max(n_damaged,   1))

class_weights = torch.tensor([weight_no_damage, weight_damaged], dtype=torch.float32).to(DEVICE)
print(f"Class weights -> no_damage: {weight_no_damage:.2f}, damaged: {weight_damaged:.2f}")

criterion = nn.CrossEntropyLoss(weight=class_weights)   # weighted loss
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)



# ============================================================
# STEP 7: TRAINING LOOP
# ------------------------------------------------------------
print("\nTraining Begins......")

   

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total_seen = 0

    # per-class tracking
    correct_damaged   = 0
    total_damaged     = 0
    correct_no_damage = 0
    total_no_damage   = 0

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
        _, labels = torch.max(labels, dim=1)  # convert one-hot back to class indices
        total_seen += labels.size(0)
        correct += (predicted == labels).sum().item()

        # per-class counts
        damaged_mask   = (labels == 1)
        no_damage_mask = (labels == 0)
        total_damaged     += damaged_mask.sum().item()
        total_no_damage   += no_damage_mask.sum().item()
        correct_damaged   += ((predicted == labels) & damaged_mask).sum().item()
        correct_no_damage += ((predicted == labels) & no_damage_mask).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc  = 100 * correct / total_seen

    # recall for each class = how many of that class we caught
    recall_damaged   = 100 * correct_damaged   / max(total_damaged, 1)
    recall_no_damage = 100 * correct_no_damage / max(total_no_damage, 1)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Loss: {epoch_loss:.4f} | "
          f"Acc: {epoch_acc:.1f}% | "
          f"Recall damaged: {recall_damaged:.1f}% | "
          f"Recall no_dmg: {recall_no_damage:.1f}%")
    

# ============================================================
# STEP 8: SAVE
# ============================================================
torch.save(model.state_dict(), "buildings_fields50.pth")
print("\nTraining done! Saved as buildings_fields50.pth")