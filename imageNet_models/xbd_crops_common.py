# ============================================================
# SHARED 250 m CROP + MODEL LOGIC
# ------------------------------------------------------------
# Single source of truth shared by training (densenet.py),
# evaluation (densenetEval.py) and inference (densenetInference.py)
# so the 250 m footprint, label rule, normalization and model
# architecture can never drift apart again.
#
# The crop helpers are lifted verbatim from xBD_train/xbd_imageCrops.py,
# the script that actually built the training set.
# ============================================================
import os
import json
import math
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely import wkt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# CONSTANTS — must match xbd_imageCrops.py
# ============================================================
DAMAGE_SUBTYPES = {"major-damage", "destroyed"}   # -> damaged scene
INTACT_SUBTYPES = {"no-damage", "minor-damage"}   # -> valid intact building
CROP_METERS     = 250.0                           # ground footprint of every crop
OUTPUT_SIZE     = 512                             # each crop resized to this (per half)

NUM_CLASSES     = 2                               # 0 = no_damage, 1 = damaged
ADAPTER_DIM     = 128


# ============================================================
# CROP HELPERS — copied verbatim from xbd_imageCrops.py
# ============================================================
def scale_to_uint8(arr):
    arr = arr.astype(np.float32)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    return (arr * 255).astype(np.uint8)


def crop_pixels(src, meters):
    """How many pixels make up `meters` on each axis (xBD tiffs are in EPSG:4326)."""
    deg_per_px_x = abs(src.transform.a)
    deg_per_px_y = abs(src.transform.e)
    center_lat   = (src.bounds.top + src.bounds.bottom) / 2.0

    m_per_deg_lat = 110540.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))

    m_per_px_x = deg_per_px_x * m_per_deg_lon
    m_per_px_y = deg_per_px_y * m_per_deg_lat

    px_w = int(round(meters / m_per_px_x))
    px_h = int(round(meters / m_per_px_y))
    px_w = min(px_w, src.width)
    px_h = min(px_h, src.height)
    return px_w, px_h


def read_crop(tif_path, row_off, col_off, height, width):
    with rasterio.open(tif_path) as src:
        row_off = max(0, min(row_off, src.height - height))
        col_off = max(0, min(col_off, src.width  - width))
        if height <= 0 or width <= 0:
            return np.empty((3, 0, 0), dtype=np.uint8)
        window = Window(col_off, row_off, width, height)   # rasterio uses (col, row)
        arr = src.read(window=window)                       # [bands, height, width]
    return arr[:3, :, :]


def resize_crop(arr, size):
    t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)   # [1, 3, H, W]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy()                                 # [3, size, size]


def window_origin(center_col, center_row, win_w, win_h, img_w, img_h):
    col_off = int(round(center_col - win_w / 2.0))
    row_off = int(round(center_row - win_h / 2.0))
    col_off = max(0, min(col_off, img_w - win_w))
    row_off = max(0, min(row_off, img_h - win_h))
    return row_off, col_off


def cluster_centers(points, win_w, win_h):
    """Greedily cluster damaged-building centres into crop centres."""
    remaining = list(points)
    centers   = []
    half_w, half_h = win_w / 2.0, win_h / 2.0

    while remaining:
        seed = remaining[0]
        cluster = [p for p in remaining
                   if abs(p[0] - seed[0]) <= half_w and abs(p[1] - seed[1]) <= half_h]
        cx = sum(p[0] for p in cluster) / len(cluster)
        cy = sum(p[1] for p in cluster) / len(cluster)
        covered = [p for p in remaining
                   if abs(p[0] - cx) <= half_w and abs(p[1] - cy) <= half_h]
        centers.append((cx, cy))
        covered_set = set(covered)
        remaining = [p for p in remaining if p not in covered_set]

    return centers


# ============================================================
# CROP -> 6-CHANNEL TENSOR (matches training: scale_to_uint8 then /255)
# ============================================================
def crop_to_6ch(pre_tif, post_tif, row_off, col_off, win_h, win_w):
    """
    Read a (win_h x win_w) window from pre+post, normalise exactly as the
    training PNGs were built (per-crop min-max -> uint8 -> /255), resize to
    OUTPUT_SIZE, and stack into a [6, OUTPUT_SIZE, OUTPUT_SIZE] float array.
    Returns None if the window could not be read.
    """
    pre_crop  = read_crop(pre_tif,  row_off, col_off, win_h, win_w)
    post_crop = read_crop(post_tif, row_off, col_off, win_h, win_w)
    if pre_crop.size == 0 or post_crop.size == 0:
        return None

    pre_resized  = resize_crop(scale_to_uint8(pre_crop),  OUTPUT_SIZE).astype(np.float32) / 255.0
    post_resized = resize_crop(scale_to_uint8(post_crop), OUTPUT_SIZE).astype(np.float32) / 255.0
    return np.concatenate([pre_resized, post_resized], axis=0)   # [6, S, S]


def scene_crops(pre_tif, post_tif, post_json):
    """
    Turn one pre/post scene into a list of (combined_6ch_float01, label) 250 m
    crops, replicating xbd_imageCrops.py's scene-classification logic:
      * damaged scene  -> one crop per damage cluster (label 1)
      * else if intact -> one centred crop            (label 0)
      * no usable buildings -> []
    """
    with open(post_json, "r") as f:
        label_data = json.load(f)

    with rasterio.open(post_tif) as src:
        img_h, img_w = src.height, src.width
        win_w, win_h = crop_pixels(src, CROP_METERS)

    damaged_pts = []
    has_intact  = False
    for building in label_data["features"]["xy"]:
        subtype = building["properties"].get("subtype", "no-damage")
        if subtype in DAMAGE_SUBTYPES:
            try:
                geom = wkt.loads(building["wkt"])
            except Exception:
                continue
            c = geom.centroid
            damaged_pts.append((c.x, c.y))     # (col, row)
        elif subtype in INTACT_SUBTYPES:
            has_intact = True

    items = []

    if damaged_pts:
        for (cx, cy) in cluster_centers(damaged_pts, win_w, win_h):
            row_off, col_off = window_origin(cx, cy, win_w, win_h, img_w, img_h)
            combined = crop_to_6ch(pre_tif, post_tif, row_off, col_off, win_h, win_w)
            if combined is not None:
                items.append((combined, 1))
    elif has_intact:
        row_off, col_off = window_origin(img_w / 2.0, img_h / 2.0,
                                         win_w, win_h, img_w, img_h)
        combined = crop_to_6ch(pre_tif, post_tif, row_off, col_off, win_h, win_w)
        if combined is not None:
            items.append((combined, 0))

    return items


# ============================================================
# MODEL — DenseNet121 with 6-ch stem, frozen backbone, adapters
# ------------------------------------------------------------
# Classifier outputs RAW LOGITS (no Softmax). Apply F.softmax at
# inference/eval where probabilities are needed.
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


def build_densenet(num_classes=NUM_CLASSES, adapter_dim=ADAPTER_DIM, pretrained=True):
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.densenet121(weights=weights)

    # replace the stem conv to accept 6 channels
    old_first_layer = model.features.conv0
    model.features.conv0 = nn.Conv2d(
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

    # unfreeze the new stem conv (must learn from scratch)
    for param in model.features.conv0.parameters():
        param.requires_grad = True

    # replace final classifier with a plain Linear -> RAW LOGITS (no Softmax)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)

    # attach adapters after every dense block
    block_channel_map = {
        "denseblock1": 256,
        "denseblock2": 512,
        "denseblock3": 1024,
        "denseblock4": 1024,
    }
    for block_name, channels in block_channel_map.items():
        block = getattr(model.features, block_name)
        wrapped = BlockWithAdapter(block, channels, adapter_dim)
        setattr(model.features, block_name, wrapped)

    return model


# ============================================================
# PLAIN RESNET (ImageNet) WITH 6-CH STEM + 2-CLASS HEAD
# ------------------------------------------------------------
# Baseline backbones to compare against the remote-sensing foundational
# models: same 6-ch (pre+post) surgery, but starting from ImageNet weights.
# freeze_backbone=True  -> linear-probe protocol (train only conv1 stem + fc),
#   matching the TorchGeo benchmark so the comparison is apples-to-apples.
# freeze_backbone=False -> full fine-tune (used for the xBD pretraining step).
# ============================================================
RESNET_ARCHS = {"resnet18": models.resnet18,
                "resnet50": models.resnet50,
                "resnet101": models.resnet101}
RESNET_SIZE = 224                                 # torchvision ResNet input size


def build_resnet6ch(arch="resnet50", num_classes=NUM_CLASSES, in_channels=6,
                    freeze_backbone=False, pretrained=True):
    if arch not in RESNET_ARCHS:
        raise KeyError(f"Unknown resnet '{arch}'. Known: {list(RESNET_ARCHS)}")
    model = RESNET_ARCHS[arch](weights="IMAGENET1K_V1" if pretrained else None)

    # widen the conv1 stem from RGB(3) to pre+post(6), copying the pretrained
    # filters into each half (magnitude-preserved) instead of random init.
    old = model.conv1
    new = nn.Conv2d(in_channels, old.out_channels, old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=(old.bias is not None))
    with torch.no_grad():
        reps = math.ceil(in_channels / old.in_channels)
        w = old.weight.repeat(1, reps, 1, 1)[:, :in_channels] / reps
        new.weight.copy_(w)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    model.conv1 = new

    # 2-class head (raw logits)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.conv1.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True
    return model


def png_to_6ch(png_path, size=RESNET_SIZE):
    """xBD training crops are saved as a side-by-side RGB PNG (pre | post).
    Split into the two halves and stack into a [6, size, size] float tensor in
    0..1, using the same per-image stretch + resize as the tif pipeline."""
    from PIL import Image
    arr = np.array(Image.open(png_path).convert("RGB"))   # [H, 2W, 3] uint8
    w = arr.shape[1] // 2
    pre  = arr[:, :w, :].transpose(2, 0, 1)               # left half  -> [3, H, w]
    post = arr[:, w:2 * w, :].transpose(2, 0, 1)          # right half -> [3, H, w]
    pre  = resize_crop(scale_to_uint8(pre),  size).astype(np.float32) / 255.0
    post = resize_crop(scale_to_uint8(post), size).astype(np.float32) / 255.0
    return torch.from_numpy(np.concatenate([pre, post], axis=0))


# ============================================================
# LoRA FOR RESNET  (manual, peft-free — same PEFT philosophy as the DenseNet
# adapters: freeze the pretrained backbone, train only small injected modules
# plus the new 6-ch stem and the head)
# ------------------------------------------------------------
# LoRAConv2d wraps a frozen Conv2d with a low-rank residual:
#   out = base(x) + (alpha/r) * lora_B(lora_A(x))
# lora_A keeps the base kernel/stride/padding so spatial dims match; lora_B is
# 1x1 and zero-initialised, so training starts exactly at the pretrained model.
# ============================================================
class LoRAConv2d(nn.Module):
    def __init__(self, base, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Conv2d(base.in_channels, r, base.kernel_size,
                                stride=base.stride, padding=base.padding,
                                dilation=base.dilation, bias=False)
        self.lora_B = nn.Conv2d(r, base.out_channels, kernel_size=1, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)              # delta starts at 0

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.drop(x)))


def _set_submodule(model, name, new):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


def inject_lora_resnet(model, r=8, alpha=16, dropout=0.05,
                       leaves=("conv1", "conv2", "conv3"), block_prefix="layer"):
    """Replace the main conv layers inside the residual blocks (layer1..layer4)
    with LoRAConv2d, leaving the 6-ch stem (model.conv1) and fc head untouched
    so they keep training fully. Returns the number of layers wrapped."""
    targets = [name for name, m in model.named_modules()
               if isinstance(m, nn.Conv2d) and name.startswith(block_prefix)
               and name.split(".")[-1] in leaves]
    for name in targets:
        base = model.get_submodule(name)
        _set_submodule(model, name, LoRAConv2d(base, r=r, alpha=alpha, dropout=dropout))
    return len(targets)


# ============================================================
# ============================================================
#  SHARED CV / METRICS / LOSS / PLOTTING  (torchgeo-FREE)
# ------------------------------------------------------------
#  These mirror torchgeo_common_linz.py but live here because the
#  DenseNet scripts run in an environment WITHOUT torchgeo installed,
#  so they cannot import that module. Implementations are numpy/torch
#  only and kept identical so DenseNet and TorchGeo results are comparable.
# ============================================================
# ============================================================
CLASS_LABELS = {"no_damage": 0, "damaged": 1}


# ============================================================
# PAIR DISCOVERY  (folder name = label):  <root>/<class>/pre|post/<name>.tif
# ============================================================
def build_labelled_pairs(root):
    """-> list of (pre_path, post_path, label, 'class/name')."""
    pairs = []
    for class_name, label in CLASS_LABELS.items():
        pre_dir  = os.path.join(root, class_name, "pre")
        post_dir = os.path.join(root, class_name, "post")
        if not (os.path.isdir(pre_dir) and os.path.isdir(post_dir)):
            print(f"  WARNING: missing pre/post for '{class_name}' under {root}")
            continue
        pre_files  = {f for f in os.listdir(pre_dir)  if f.lower().endswith(".tif")}
        post_files = {f for f in os.listdir(post_dir) if f.lower().endswith(".tif")}
        for name in sorted(pre_files & post_files):
            pairs.append((os.path.join(pre_dir, name),
                          os.path.join(post_dir, name), label, f"{class_name}/{name}"))
    return pairs


def read_rgb(tif_path):
    with rasterio.open(tif_path) as src:
        arr = src.read()
    return arr[:3, :, :]


def pair_to_6ch(pre_tif, post_tif, size=OUTPUT_SIZE):
    """Read a pre/post pair -> [6, size, size] float tensor in 0..1, using the
    same min-max stretch + resize the DenseNet was trained on."""
    pre, post = read_rgb(pre_tif), read_rgb(post_tif)
    if pre.size == 0 or post.size == 0:
        return None
    pre  = resize_crop(scale_to_uint8(pre),  size).astype(np.float32) / 255.0
    post = resize_crop(scale_to_uint8(post), size).astype(np.float32) / 255.0
    return torch.from_numpy(np.concatenate([pre, post], axis=0))


# ============================================================
# AUGMENTATION  (geometry shared across channels; photometric shared across
# pre/post so the change signal is preserved)
# ============================================================
def augment_6ch(t):
    if torch.rand(1).item() < 0.5:
        t = torch.flip(t, dims=[2])
    if torch.rand(1).item() < 0.5:
        t = torch.flip(t, dims=[1])
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        t = torch.rot90(t, k, dims=[1, 2])
    t = t * (1.0 + (torch.rand(1).item() - 0.5) * 0.4)
    mean = t.mean()
    t = (t - mean) * (1.0 + (torch.rand(1).item() - 0.5) * 0.4) + mean
    t = t + torch.randn_like(t) * 0.02
    return t.clamp(0.0, 1.0)


# ============================================================
# STRATIFIED K-FOLD (numpy only)
# ============================================================
def stratified_folds(labels, n_folds=5, seed=42):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    fold_test = [[] for _ in range(n_folds)]
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        for k, chunk in enumerate(np.array_split(idx, n_folds)):
            fold_test[k].extend(chunk.tolist())
    return [np.array(sorted(f)) for f in fold_test]


# ============================================================
# METRICS
# ============================================================
def compute_metrics(true_labels, predictions):
    TP = int(((predictions == 1) & (true_labels == 1)).sum())
    TN = int(((predictions == 0) & (true_labels == 0)).sum())
    FP = int(((predictions == 1) & (true_labels == 0)).sum())
    FN = int(((predictions == 0) & (true_labels == 1)).sum())
    total = len(true_labels)
    accuracy  = (TP + TN) / total if total else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "TP": TP, "TN": TN, "FP": FP, "FN": FN, "total": total}


def roc_auc(true_labels, scores):
    """Threshold-free separability (Mann-Whitney rank formula; ties -> avg rank)."""
    true_labels = np.asarray(true_labels); scores = np.asarray(scores)
    n_pos = int((true_labels == 1).sum()); n_neg = int((true_labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ss = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i, n = 0, len(scores)
    while i < n:
        j = i
        while j + 1 < n and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return (ranks[true_labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def threshold_sweep(true_labels, scores, thresholds=None):
    true_labels = np.asarray(true_labels); scores = np.asarray(scores)
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    return [{"threshold": float(t),
             **compute_metrics(true_labels, (scores >= t).astype(int))}
            for t in thresholds]


def fbeta(precision, recall, beta=2.0):
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom else 0.0


# ============================================================
# ROC / PR CURVES + ROC-OPTIMAL THRESHOLD
# ============================================================
def roc_curve(true_labels, scores):
    """ROC points over every score threshold; starts at (0,0), ends at (1,1)."""
    y = np.asarray(true_labels); s = np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
    uniq = np.unique(s)[::-1]
    thresholds = np.concatenate([[uniq[0] + 1.0], uniq])
    tpr = np.empty(len(thresholds)); fpr = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        pred = s >= t
        tpr[i] = np.logical_and(pred, y == 1).sum() / n_pos
        fpr[i] = np.logical_and(pred, y == 0).sum() / n_neg
    return fpr, tpr, thresholds


def best_threshold_youden(true_labels, scores):
    """ROC-optimal threshold maximizing Youden's J = TPR - FPR. Returns
    (threshold, tpr, fpr, J)."""
    fpr, tpr, thr = roc_curve(true_labels, scores)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(thr[k]), float(tpr[k]), float(fpr[k]), float(j[k])


def pr_curve(true_labels, scores):
    """Precision-Recall points over every score threshold (recall non-decreasing)."""
    y = np.asarray(true_labels); s = np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum())
    thresholds = np.unique(s)[::-1]
    recall = np.empty(len(thresholds)); precision = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        pred = s >= t
        tp = int(np.logical_and(pred, y == 1).sum())
        fp = int(np.logical_and(pred, y == 0).sum())
        recall[i] = tp / n_pos if n_pos else 0.0
        precision[i] = tp / (tp + fp) if (tp + fp) else 1.0
    return recall, precision, thresholds


def average_precision(true_labels, scores):
    """Area under the precision-recall curve (trapezoidal over recall)."""
    recall, precision, _ = pr_curve(true_labels, scores)
    if len(recall) == 0:
        return float("nan")
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0]], precision])
    trap = getattr(np, "trapezoid", np.trapezoid)   # np>=2 renamed trapz -> trapezoid
    return float(trap(precision, recall))


# ============================================================
# LOSS FUNCTIONS  (cross_entropy / focal / dice / focal_dice)
# ============================================================
class FocalLoss(nn.Module):
    """Multiclass focal loss; gamma focuses on hard/minority examples,
    optional per-class `weight` (inverse frequency)."""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1.0 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device).gather(0, target)
        return loss.mean()


class DiceLoss(nn.Module):
    """Soft Dice on the positive-class probability (recall-friendly)."""
    def __init__(self, smooth=1.0, positive_index=1):
        super().__init__()
        self.smooth = smooth
        self.pos = positive_index

    def forward(self, logits, target):
        prob = F.softmax(logits, dim=1)[:, self.pos]
        tgt = (target == self.pos).float()
        inter = (prob * tgt).sum()
        denom = prob.sum() + tgt.sum()
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice


class ComboLoss(nn.Module):
    """Weighted sum of two criteria (e.g. focal + dice)."""
    def __init__(self, a, b, wa=1.0, wb=1.0):
        super().__init__()
        self.a, self.b, self.wa, self.wb = a, b, wa, wb

    def forward(self, logits, target):
        return self.wa * self.a(logits, target) + self.wb * self.b(logits, target)


def build_criterion(loss_cfg, class_weight=None):
    """Turn a loss config dict into a criterion callable(logits, target)."""
    cfg = dict(loss_cfg or {})
    ltype = str(cfg.get("type", "cross_entropy")).lower()
    w = class_weight if cfg.get("class_weighting", True) else None
    focal = cfg.get("focal", {}) or {}
    dice  = cfg.get("dice", {}) or {}
    combo = cfg.get("combo", {}) or {}
    if ltype in ("cross_entropy", "ce"):
        return nn.CrossEntropyLoss(weight=w)
    if ltype == "focal":
        return FocalLoss(gamma=float(focal.get("gamma", 2.0)), weight=w)
    if ltype == "dice":
        return DiceLoss(smooth=float(dice.get("smooth", 1.0)))
    if ltype in ("focal_dice", "focaldice", "combo"):
        return ComboLoss(
            FocalLoss(gamma=float(focal.get("gamma", 2.0)), weight=w),
            DiceLoss(smooth=float(dice.get("smooth", 1.0))),
            wa=float(combo.get("focal_weight", 1.0)),
            wb=float(combo.get("dice_weight", 1.0)))
    raise ValueError(f"Unknown loss type '{ltype}'. "
                     "Choose: cross_entropy, focal, dice, focal_dice.")


# ============================================================
# PLOTTING  (lazy matplotlib import so non-plotting scripts stay light)
# ============================================================
def _mpl():
    """Configure headless matplotlib with a writable cache dir on NeSI."""
    os.environ.setdefault("MPLCONFIGDIR", "/home/c/cpehesar/research/cache/mpl")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _draw_cm(ax, cm, title, normalize=False):
    """Render a 2x2 confusion matrix (rows=actual, cols=predicted) on an axis."""
    disp = cm.astype(float)
    if normalize:
        rowsum = disp.sum(axis=1, keepdims=True)
        disp = np.divide(disp, rowsum, out=np.zeros_like(disp), where=rowsum != 0)
    vmax = disp.max() if disp.max() > 0 else 1.0
    ax.imshow(disp, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["no_damage", "damaged"]); ax.set_yticklabels(["no_damage", "damaged"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    for i in range(2):
        for j in range(2):
            txt = f"{disp[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=12,
                    color="white" if disp[i, j] > vmax / 2 else "black")


def _calibration(y, p, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 0:
            xs.append(float(p[m].mean())); ys.append(float(y[m].mean()))
    return np.array(xs), np.array(ys)


def plot_confusion(y, prob, thr_default, thr_op, thr_src, tag, out_dir):
    """Confusion matrices (counts + row-normalized) at 0.5 and the operating threshold."""
    plt = _mpl()
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for row, (thr, lab) in enumerate(
            [(thr_default, f"thr={thr_default:.2f} (default)"),
             (thr_op,      f"thr={thr_op:.2f} ({thr_src})")]):
        m = compute_metrics(np.asarray(y), (np.asarray(prob) >= thr).astype(int))
        cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        _draw_cm(axes[row, 0], cm, f"Counts — {lab}", normalize=False)
        _draw_cm(axes[row, 1], cm, f"Row-normalized — {lab}", normalize=True)
    fig.suptitle(f"{tag} — confusion matrices (held-out)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, f"eval_{tag}_confusion.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  saved confusion matrices -> {path}")


def plot_eval_curves(y, prob, thr_op, tag, out_dir):
    """Six-panel evaluation: ROC, PR, threshold sweep, score histogram,
    calibration, and a metric-summary bar chart."""
    plt = _mpl()
    os.makedirs(out_dir, exist_ok=True)
    y = np.asarray(y); prob = np.asarray(prob)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc(y, prob)
    ty, tpr_y, fpr_y, _ = best_threshold_youden(y, prob)
    ax = axes[0, 0]
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.scatter([fpr_y], [tpr_y], color="red", zorder=5, label=f"Youden thr={ty:.2f}")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend(loc="lower right")

    rec, prec, _ = pr_curve(y, prob)
    ap = average_precision(y, prob)
    base_rate = float(np.mean(y))
    ax = axes[0, 1]
    ax.plot(rec, prec, lw=2, label=f"PR (AP={ap:.3f})")
    ax.axhline(base_rate, ls="--", color="gray", lw=1, label=f"baseline={base_rate:.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02)
    ax.set_title("Precision-Recall curve"); ax.legend(loc="lower left")

    ax = axes[0, 2]
    ts = np.round(np.arange(0.05, 0.96, 0.05), 2)
    sweep = threshold_sweep(y, prob, thresholds=ts)
    ax.plot(ts, [s["precision"] for s in sweep], label="precision")
    ax.plot(ts, [s["recall"] for s in sweep], label="recall")
    ax.plot(ts, [s["f1"] for s in sweep], label="F1")
    ax.axvline(0.5, ls=":", color="gray", label="0.5")
    ax.axvline(thr_op, ls="--", color="red", label=f"op={thr_op:.2f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title("Metrics vs threshold"); ax.legend()

    ax = axes[1, 0]
    ax.hist(prob[y == 0], bins=20, range=(0, 1), alpha=0.6, color="C0", label="no_damage")
    ax.hist(prob[y == 1], bins=20, range=(0, 1), alpha=0.6, color="C3", label="damaged")
    ax.axvline(thr_op, ls="--", color="red", label=f"op={thr_op:.2f}")
    ax.set_xlabel("P(damaged)"); ax.set_ylabel("count")
    ax.set_title("Score distribution by true class"); ax.legend()

    ax = axes[1, 1]
    xs, ys = _calibration(y, prob)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect")
    ax.plot(xs, ys, "o-", color="C2", label="model")
    ax.set_xlabel("Mean predicted P(damaged)"); ax.set_ylabel("Observed fraction damaged")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Calibration (reliability)"); ax.legend(loc="upper left")

    ax = axes[1, 2]
    m = compute_metrics(y, (prob >= thr_op).astype(int))
    labels = ["accuracy", "precision", "recall", "f1", "roc_auc", "avg_prec"]
    vals = [m["accuracy"], m["precision"], m["recall"], m["f1"], auc, ap]
    bars = ax.bar(labels, vals, color=["C0", "C1", "C3", "C4", "C5", "C6"])
    ax.set_ylim(0, 1.05); ax.tick_params(axis="x", rotation=30)
    ax.set_title(f"Metrics @ op threshold {thr_op:.2f}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    fig.suptitle(f"{tag} — held-out evaluation ({len(y)} crops)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, f"eval_{tag}_curves.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  saved eval curves -> {path}")


def plot_training_curves(histories, tag, out_dir):
    """Per-fold train/eval loss (mean +/- std over folds), from the loss
    histories saved in the CV checkpoint."""
    if not histories:
        return
    plt = _mpl()
    os.makedirs(out_dir, exist_ok=True)
    tl = np.array([h["train_loss"] for h in histories], dtype=float)
    epochs = np.arange(1, tl.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, tl.mean(0), color="C0", label="train")
    ax.fill_between(epochs, tl.mean(0) - tl.std(0), tl.mean(0) + tl.std(0),
                    color="C0", alpha=0.2)
    eval_key = None
    if histories[0].get("val_loss"):
        eval_key = "val_loss"
    elif histories[0].get("eval_loss"):
        eval_key = "eval_loss"
    if eval_key is not None:
        vl = np.array([h[eval_key] for h in histories], dtype=float)
        ax.plot(epochs, vl.mean(0), color="C1", label="validation" if eval_key == "val_loss" else "eval")
        ax.fill_between(epochs, vl.mean(0) - vl.std(0), vl.mean(0) + vl.std(0),
                        color="C1", alpha=0.2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"{tag} — train/eval loss (mean +/- std over folds)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, f"eval_{tag}_loss.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  saved loss curves -> {path}")
