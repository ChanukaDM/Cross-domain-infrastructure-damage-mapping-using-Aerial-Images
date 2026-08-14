# ============================================================
# IMPORTS — load the tools we need
# ============================================================
import os                          # work with file paths and folders
import json                        # read the xBD label files (they are JSON text)
import numpy as np                 # array math (for scaling the image numbers)
import rasterio                    # reads int16 / multi-band TIFF images correctly
import torch                       # PyTorch: the core deep learning library
import torch.nn as nn              # neural network building blocks (layers, loss functions)
import torch.nn.functional as F    # extra functions, here used to resize images
import torch.optim as optim        # optimizers: algorithms that adjust the model to learn
from torch.utils.data import Dataset, DataLoader  # tools to feed data to the model in batches
from torchvision import models     # gives us the pretrained ResNet


# ============================================================
# CONFIG — settings in one place so they are easy to change
# ============================================================
TRAIN_IMAGES_DIR = "./geotiffs/tier1/images"   # folder holding the pre/post .tif images
TRAIN_LABELS_DIR = "./geotiffs/tier1/labels"   # folder holding the matching .json label files
BATCH_SIZE = 16                     # how many image-pairs the model sees before updating itself
EPOCHS = 10                         # how many full passes over the whole dataset
LEARNING_RATE = 0.0001              # size of each learning step (small = careful learning)
NUM_CLASSES = 2                     # two answers: 0 = no_damage, 1 = damaged
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # use the A40 GPU if available, else CPU


# ============================================================
# STEP 1: READ ONE JSON LABEL FILE -> a single 0 or 1
# ------------------------------------------------------------
# Each xBD label file lists every building and its damage level ("subtype").
# Rule: if ANY building is damaged, the whole image is "damaged" (1),
# otherwise it is "no_damage" (0).
# ============================================================
def get_label_from_json(json_path):
    with open(json_path, "r") as f:        # open the JSON file for reading
        data = json.load(f)                # turn the JSON text into a Python dictionary

    buildings = data["features"]["xy"]     # get the list of buildings in this image

    for building in buildings:                                  # look at each building one by one
        subtype = building["properties"].get("subtype", "no-damage")  # read its damage level
        if subtype in ["minor-damage", "major-damage", "destroyed"]:  # if it shows any damage
            return 1                                            # -> whole image is "damaged"
    return 0                                                    # no damage found -> "no_damage"


# ============================================================
# STEP 2: BUILD A LIST OF (pre_image, post_image, label)
# ------------------------------------------------------------
# xBD names files like:
#   guatemala-volcano_00000000_pre_disaster.tif    <- "before" image
#   guatemala-volcano_00000000_post_disaster.tif   <- "after" image
#   guatemala-volcano_00000000_post_disaster.json  <- labels for the after image
# We start from each POST image and find its matching PRE image and JSON.
# ============================================================
def build_file_list(images_dir, labels_dir):
    file_list = []                                  # start with an empty list

    for filename in os.listdir(images_dir):         # loop over every file in the images folder
        if "_post_disaster.tif" not in filename:    # we only start from POST images...
            continue                                # ...skip everything else

        post_image_path = os.path.join(images_dir, filename)  # full path to the post image

        # build the matching PRE image filename by swapping "post" for "pre"
        pre_filename = filename.replace("_post_disaster.tif", "_pre_disaster.tif")
        pre_image_path = os.path.join(images_dir, pre_filename)  # full path to the pre image

        # build the matching JSON label filename (.tif -> .json)
        json_filename = filename.replace(".tif", ".json")
        json_path = os.path.join(labels_dir, json_filename)      # full path to the label file

        # safety check: skip this sample if the pre image or the label is missing
        if not (os.path.exists(pre_image_path) and os.path.exists(json_path)):
            continue

        label = get_label_from_json(json_path)      # read the JSON -> 0 or 1
        file_list.append((pre_image_path, post_image_path, label))  # save the triple

    return file_list                                # hand back the finished list


# ============================================================
# STEP 3: DATASET — teach PyTorch how to load and prepare ONE sample
# ------------------------------------------------------------
# A sample = pre image + post image + label.
# Because the TIFFs are int16 (not normal 0-255 images), we read them with
# rasterio and scale them into the 0..1 range ourselves.
# ============================================================
class DamageDataset(Dataset):
    def __init__(self, file_list):
        self.data = file_list          # the list of (pre, post, label) triples

    def __len__(self):
        return len(self.data)          # how many samples there are

    def load_tiff(self, path):
        # open the TIFF and read it as a raw numerical array
        with rasterio.open(path) as src:        # open the image file
            arr = src.read()                    # shape [bands, height, width], type int16

        arr = arr.astype(np.float32)            # convert int16 -> float so we can do decimal math
        arr = arr[:3, :, :]                     # keep only the first 3 bands (RGB), drop extras

        # --- per-image scaling: stretch THIS image's range into 0..1 ---
        arr_min = arr.min()                     # the darkest value in this image
        arr_max = arr.max()                     # the brightest value in this image
        if arr_max > arr_min:                   # guard against dividing by zero (flat image)
            arr = (arr - arr_min) / (arr_max - arr_min)   # now every value sits between 0 and 1

        tensor = torch.from_numpy(arr)          # convert the numpy array into a PyTorch tensor

        # resize to 224x224 (the size ResNet expects).
        # interpolate needs shape [batch, channels, h, w], so we add a fake batch
        # dimension, resize, then remove it again.
        tensor = tensor.unsqueeze(0)            # [3, h, w] -> [1, 3, h, w]
        tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        tensor = tensor.squeeze(0)              # [1, 3, 224, 224] -> [3, 224, 224]

        return tensor                           # a clean [3, 224, 224] float tensor in 0..1

    def __getitem__(self, index):
        pre_path, post_path, label = self.data[index]   # paths + label for this position

        pre_tensor = self.load_tiff(pre_path)    # load + scale + resize the pre image
        post_tensor = self.load_tiff(post_path)  # same for the post image

        # stack pre (3 channels) and post (3 channels) -> one 6-channel tensor
        combined = torch.cat([pre_tensor, post_tensor], dim=0)  # shape [6, 224, 224]

        return combined, label                   # the 6-channel tensor and its 0/1 label


# ============================================================
# STEP 4: LOAD THE DATA
# ============================================================
all_files = build_file_list(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)  # get list of all samples
print(f"Found {len(all_files)} image pairs.")                    # show how many we found

dataset = DamageDataset(all_files)              # wrap the list in our Dataset class
loader = DataLoader(                            # DataLoader serves the data in batches
    dataset,
    batch_size=BATCH_SIZE,                      # images per batch
    shuffle=True,                               # shuffle the order each epoch (helps learning)
    num_workers=4                               # 4 background helpers to load images faster
)


# ============================================================
# STEP 5: THE MODEL — pretrained ResNet adapted to 6 channels + 2 classes
# ============================================================
model = models.resnet18(weights="IMAGENET1K_V1")    # load ResNet18 pretrained on ImageNet

# ResNet's first layer normally expects 3 channels, but we feed 6 (pre + post).
# So we replace it with a new first layer that accepts 6 channels.
old_first_layer = model.conv1                       # the original first layer (for its settings)
model.conv1 = nn.Conv2d(                            # build a new first convolution layer
    in_channels=6,                                  # accept 6 input channels instead of 3
    out_channels=old_first_layer.out_channels,      # keep the same number of outputs (64)
    kernel_size=old_first_layer.kernel_size,        # keep the same filter size (7x7)
    stride=old_first_layer.stride,                  # keep the same stride
    padding=old_first_layer.padding,                # keep the same padding
    bias=False                                      # ResNet conv layers use no bias
)

# ResNet's last layer normally outputs 1000 ImageNet classes; we want only 2.
num_features = model.fc.in_features                 # how many inputs the final layer takes
model.fc = nn.Linear(num_features, NUM_CLASSES)     # replace it with a 2-output layer

model = model.to(DEVICE)                            # move the model onto the GPU (or CPU)


# ============================================================
# STEP 6: LOSS + OPTIMIZER — how the model measures and fixes mistakes
# ============================================================
criterion = nn.CrossEntropyLoss()                   # measures how wrong the predictions are
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # adjusts weights to reduce the loss


# ============================================================
# STEP 7: TRAINING LOOP — where the learning actually happens
# ============================================================
for epoch in range(EPOCHS):                         # repeat over the whole dataset EPOCHS times
    model.train()                                   # put the model in training mode
    running_loss = 0.0                              # total error so far this epoch
    correct = 0                                     # how many predictions were correct
    total = 0                                       # how many images we have seen

    for images, labels in loader:                   # loop over each batch of data
        images = images.to(DEVICE)                  # move the 6-channel images to the GPU
        labels = labels.to(DEVICE)                  # move the labels to the GPU

        optimizer.zero_grad()                       # clear leftover gradients from the last step
        outputs = model(images)                     # run images through the model -> raw scores
        loss = criterion(outputs, labels)           # measure how wrong those scores are
        loss.backward()                             # work out how to adjust each weight
        optimizer.step()                            # apply the adjustments

        running_loss += loss.item()                 # add this batch's loss to the running total
        _, predicted = torch.max(outputs, 1)        # pick the highest-scoring class as the guess
        total += labels.size(0)                     # count images in this batch
        correct += (predicted == labels).sum().item()  # count how many guesses were correct

    epoch_loss = running_loss / len(loader)         # average loss across all batches
    epoch_acc = 100 * correct / total               # accuracy as a percentage
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")


# ============================================================
# STEP 8: SAVE THE TRAINED MODEL
# ============================================================
torch.save(model.state_dict(), "damage_model.pth")  # save the learned weights to a file
print("Training done! Model saved as damage_model.pth")