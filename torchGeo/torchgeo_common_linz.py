# ============================================================
# SHARED TORCHGEO BENCHMARK LOGIC (LINZ Cyclone Gabrielle)
# ------------------------------------------------------------
# One source of truth for the TorchGeo finetune / eval / inference /
# 5-fold-CV-benchmark scripts. Lets you swap the backbone via a single
# MODEL_NAME string and keep every other knob (data, 6-ch input,
# normalization, augmentation, loss, split) identical across models —
# which is what makes the benchmark fair.
#
# Models are fine-tuned DIRECTLY on LINZ (no xBD step), from each
# backbone's remote-sensing pretrained weights.
#
# Architecture-agnostic surgery: for ANY backbone we (1) widen the FIRST
# Conv2d stem from RGB(3) to pre+post(6) channels by copying the
# pretrained filters into both halves, and (2) replace the FINAL Linear
# with a 2-class head. This works uniformly for ResNet / Swin-V2 / ViT
# because they all start with a Conv2d stem and end with a Linear head.
# ============================================================
import os

# NeSI home dirs have a tiny quota, but pretrained weights (esp. Satlas
# Swin-V2 ~350MB) download to ~/.cache by default and blow it out with
# "No space left on device". Redirect all model caches to nobackup BEFORE
# torch / torchgeo are imported. Override by exporting these yourself.
_CACHE = "/home/c/cpehesar/research/cache"
os.environ.setdefault("TORCH_HOME",   f"{_CACHE}/torch")
os.environ.setdefault("HF_HOME",      f"{_CACHE}/hf")
os.environ.setdefault("MPLCONFIGDIR", f"{_CACHE}/mpl")   # matplotlib font cache (torchgeo imports it)
for _d in ("TORCH_HOME", "HF_HOME", "MPLCONFIGDIR"):
    os.makedirs(os.environ[_d], exist_ok=True)

import math
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F



import torchgeo.models as tgm
from torchgeo.models import get_model, get_weight



# ============================================================
# MODEL REGISTRY  —  name -> (torchgeo builder, weight enum string, input size)
# ------------------------------------------------------------
# `weights` are given as STRINGS and resolved at build time, so an
# unavailable name fails loudly with the list of valid options for that
# family (see _resolve_weight) instead of an import error.
# ============================================================
MODELS = {
    # --- ResNet (RGB remote-sensing pretrain) ---
    "resnet50_fmow_rgb":      {"builder": "resnet50",
                               "weights": "ResNet50_Weights.FMOW_RGB_GASSL", "size": 256},
    "resnet50_sentinel2_rgb": {"builder": "resnet50",
                               "weights": "ResNet50_Weights.SENTINEL2_RGB_MOCO", "size": 256},
    "resnet18_sentinel2_rgb": {"builder": "resnet18",
                               "weights": "ResNet18_Weights.SENTINEL2_RGB_MOCO", "size": 256},
    # --- Satlas Swin-V2 (high-res aerial / satellite, RGB) ---
    "swin_v2_b_satlas_naip":  {"builder": "swin_v2_b",
                               "weights": "Swin_V2_B_Weights.NAIP_RGB_SI_SATLAS", "size": 256},
    "swin_v2_b_satlas_s2":    {"builder": "swin_v2_b",
                               "weights": "Swin_V2_B_Weights.SENTINEL2_SI_RGB_SATLAS", "size": 256},
    "swin_v2_t_satlas_s2":    {"builder": "swin_v2_t",
                               "weights": "Swin_V2_T_Weights.SENTINEL2_SI_RGB_SATLAS", "size": 256},
    # --- ViT-Small/16 (only multispectral pretrain exists; surgery maps 13->6ch) ---
    "vit_small_sentinel2":    {"builder": "vit_small_patch16_224",
                               "weights": "ViTSmall16_Weights.SENTINEL2_ALL_DINO", "size": 224},
}

# default single-model used by finetune/eval/inference scripts
DEFAULT_MODEL   = "resnet50_fmow_rgb"

NUM_CLASSES     = 2            # 0 = no_damage, 1 = damaged
IN_CHANNELS     = 6           # pre(3) + post(3)
TG_OUTPUT_SIZE  = MODELS[DEFAULT_MODEL]["size"]
CLASS_LABELS    = {"no_damage": 0, "damaged": 1}




# Safe fallback logic to handle the name change dynamically
if hasattr(np, 'trapezoid'):
    numpy_trap_func = np.trapezoid
else:
    numpy_trap_func = np.trapz



# ============================================================
# NORMALIZATION + I/O  (identical across all models for a fair benchmark)
# ============================================================
def scale_to_uint8(arr):
    arr = arr.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return (arr * 255).astype(np.uint8)


def resize_to(arr, size):
    t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)   # [1, 3, H, W]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy()


def read_rgb(tif_path):
    with rasterio.open(tif_path) as src:
        arr = src.read()
    return arr[:3, :, :]


def pair_to_6ch(pre_tif, post_tif, size=TG_OUTPUT_SIZE):
    """Read a pre/post pair -> [6, size, size] float tensor in 0..1."""
    pre, post = read_rgb(pre_tif), read_rgb(post_tif)
    if pre.size == 0 or post.size == 0:
        return None
    pre  = resize_to(scale_to_uint8(pre),  size).astype(np.float32) / 255.0
    post = resize_to(scale_to_uint8(post), size).astype(np.float32) / 255.0
    return torch.from_numpy(np.concatenate([pre, post], axis=0))


# ============================================================
# PAIR DISCOVERY
#   labelled  : <root>/<class>/pre|post/<name>.tif  -> (pre, post, label, name)
#   unlabelled: <root>/pre|post/<name>.tif          -> (pre, post, name)
# ============================================================
def build_labelled_pairs(root):
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


def build_unlabelled_pairs(root):
    pre_dir, post_dir = os.path.join(root, "pre"), os.path.join(root, "post")
    pre_files  = {f for f in os.listdir(pre_dir)  if f.lower().endswith(".tif")}
    post_files = {f for f in os.listdir(post_dir) if f.lower().endswith(".tif")}
    return [(os.path.join(pre_dir, n), os.path.join(post_dir, n), n)
            for n in sorted(pre_files & post_files)]


# ============================================================
# AUGMENTATION  (geometry shared across channels; photometric shared
# across pre/post so the change signal is preserved)
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
# GENERIC MODEL SURGERY
# ============================================================
def _module_by_name(model, name):
    mod = model
    for p in name.split("."):
        mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
    return mod


def _set_by_name(model, name, new):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


def _first_of(model, cls):
    for name, m in model.named_modules():
        if isinstance(m, cls):
            return name, m
    return None, None


def _last_of(model, cls):
    found = (None, None)
    for name, m in model.named_modules():
        if isinstance(m, cls):
            found = (name, m)
    return found


def _widen_first_conv(model, in_channels):
    """Replace the first Conv2d (RGB stem) with an in_channels version,
    copying pretrained filters into each repeated block (magnitude-preserved)."""
    name, conv = _first_of(model, nn.Conv2d)
    if conv is None:
        raise RuntimeError("No Conv2d stem found to widen.")
    new = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size,
                    stride=conv.stride, padding=conv.padding,
                    dilation=conv.dilation, groups=conv.groups,
                    bias=(conv.bias is not None))
    with torch.no_grad():
        reps = math.ceil(in_channels / conv.in_channels)
        w = conv.weight.repeat(1, reps, 1, 1)[:, :in_channels] / reps
        new.weight.copy_(w)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    _set_by_name(model, name, new)
    return name


def _replace_head(model, num_classes):
    """Replace the final Linear with a fresh num_classes head (raw logits)."""
    name, lin = _last_of(model, nn.Linear)
    if lin is None:
        raise RuntimeError("No Linear head found to replace.")
    _set_by_name(model, name, nn.Linear(lin.in_features, num_classes))
    return name


def _resolve_weight(wstr):
    try:
        return get_weight(wstr)
    except Exception:
        enum_name = wstr.split(".")[0]
        E = getattr(tgm, enum_name, None)
        avail = [f"{enum_name}.{w.name}" for w in E] if E is not None else []
        raise ValueError(f"Weight '{wstr}' not available. Valid options for "
                         f"{enum_name}:\n  " + "\n  ".join(avail or ["<enum not found>"]))


# ============================================================
# BUILD MODEL  —  registry name -> pretrained, 6-ch stem, 2-class head
# ------------------------------------------------------------
# freeze_backbone=True  => benchmark/linear-probe protocol: only the new
#   6-ch stem and the new head train (frozen RS features). Fast + fair.
# freeze_backbone=False => full fine-tune (use a smaller LR).
# ============================================================
def build_model(model_name=DEFAULT_MODEL, num_classes=NUM_CLASSES,
                in_channels=IN_CHANNELS, freeze_backbone=True):
    if model_name not in MODELS:
        raise KeyError(f"Unknown model '{model_name}'. Known: {list(MODELS)}")
    spec    = MODELS[model_name]
    weights = _resolve_weight(spec["weights"])
    model   = get_model(spec["builder"], weights=weights)

    stem_name = _widen_first_conv(model, in_channels)
    head_name = _replace_head(model, num_classes)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for nm in (stem_name, head_name):
            for p in _module_by_name(model, nm).parameters():
                p.requires_grad = True
    return model


def model_input_size(model_name):
    return MODELS[model_name]["size"]


def model_family(model_name):
    name = model_name.lower()
    if name.startswith("resnet"):
        return "resnet"
    if name.startswith("swin"):
        return "swin"
    if name.startswith("vit"):
        return "vit"
    return "generic"


def lora_target_modules(model_name):
    family = model_family(model_name)
    if family == "resnet":
        return ["conv1", "conv2", "conv3"]
    if family == "swin":
        return ["qkv", "proj"]
    if family == "vit":
        return ["qkv", "proj", "fc1", "fc2"]
    return ["fc"]


def checkpoint_path(model_name):
    return f"torchgeo_{model_name}_linz.pth"


# ============================================================
# METRICS + DIAGNOSTICS (shared by eval + benchmark)
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
    true_labels = np.asarray(true_labels)
    scores = np.asarray(scores)
    n_pos = int((true_labels == 1).sum())
    n_neg = int((true_labels == 0).sum())
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
    true_labels = np.asarray(true_labels)
    scores = np.asarray(scores)
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
# ROC / PR CURVES + ROC-OPTIMAL THRESHOLD  (numpy only — no sklearn)
# ============================================================
def roc_curve(true_labels, scores):
    """ROC points over every score threshold. Returns (fpr, tpr, thresholds);
    the curve starts at (0,0) [threshold above max score] and ends at (1,1)."""
    y = np.asarray(true_labels)
    s = np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
    uniq = np.unique(s)[::-1]                       # descending score thresholds
    thresholds = np.concatenate([[uniq[0] + 1.0], uniq])
    tpr = np.empty(len(thresholds)); fpr = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        pred = s >= t
        tpr[i] = np.logical_and(pred, y == 1).sum() / n_pos
        fpr[i] = np.logical_and(pred, y == 0).sum() / n_neg
    return fpr, tpr, thresholds


def best_threshold_youden(true_labels, scores):
    """ROC-optimal operating point: the threshold maximizing Youden's J =
    TPR - FPR (= sensitivity + specificity - 1). Returns (threshold, tpr, fpr, J).
    This is the principled replacement for a fixed 0.5 cutoff."""
    fpr, tpr, thr = roc_curve(true_labels, scores)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(thr[k]), float(tpr[k]), float(fpr[k]), float(j[k])


def pr_curve(true_labels, scores):
    """Precision-Recall points over every score threshold (recall non-decreasing).
    Returns (recall, precision, thresholds)."""
    y = np.asarray(true_labels)
    s = np.asarray(scores, dtype=float)
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
    return float(numpy_trap_func(precision, recall))


# ============================================================
# LOSS FUNCTIONS  (selectable: cross_entropy / focal / dice / focal_dice)
# ------------------------------------------------------------
# All take raw logits [B, C] and integer targets [B] (C=2 here). Focal and
# Dice both up-weight the rare "damaged" class to lift recall: focal by
# down-weighting easy negatives, dice by directly optimizing overlap.
# ============================================================
class FocalLoss(nn.Module):
    """Multiclass focal loss (Lin et al., 2017). gamma>0 focuses learning on
    hard / minority examples; optional per-class `weight` (inverse frequency)."""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # tensor [C] or None

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1.0 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device).gather(0, target)
        return loss.mean()


class DiceLoss(nn.Module):
    """Soft Dice loss on the positive-class probability. Directly optimizes
    overlap, which is recall-friendly under heavy class imbalance."""
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
    """Turn a loss config dict into a criterion callable(logits, target).

    loss_cfg keys:
      type            : cross_entropy | focal | dice | focal_dice
      class_weighting : apply the inverse-frequency `class_weight` (CE / focal)
      focal.gamma     : focusing parameter (default 2.0)
      dice.smooth     : Dice Laplace smoothing (default 1.0)
      combo.focal_weight / combo.dice_weight : focal_dice mix (default 1.0 each)
    """
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
            wb=float(combo.get("dice_weight", 1.0)),
        )
    raise ValueError(f"Unknown loss type '{ltype}'. "
                     "Choose: cross_entropy, focal, dice, focal_dice.")


# ============================================================
# STRATIFIED K-FOLD (numpy only — no sklearn dependency)
# ============================================================
def stratified_folds(labels, n_folds=5, seed=42):
    """Return a list of n_folds index arrays (the test indices of each fold),
    balanced per class."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    fold_test = [[] for _ in range(n_folds)]
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        for k, chunk in enumerate(np.array_split(idx, n_folds)):
            fold_test[k].extend(chunk.tolist())
    return [np.array(sorted(f)) for f in fold_test]
