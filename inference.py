
# COMMON INFERENCE OVER THE FULL DATASET

# Classifies every 500 m pre/post tile in ./data with ONE trained model
# (a DenseNet / ResNet base model, or a TorchGeo foundational model) and
# writes two JSONL files: the damaged tiles and the no_damage tiles.
#
# WHY TILING: the models were trained on 250 m x 250 m crops, but ./data
# tiles are 500 m x 500 m. So each tile is split into a 2x2 grid of 250 m
# crops and each crop is preprocessed EXACTLY as in training (per-crop
# min-max stretch -> resize to the model input size -> stack pre(3)+post(3)).
# A tile is labelled "damaged" if ANY of its 250 m crops is predicted
# damaged (recall-oriented; configurable via --agg).

import os
import json
import argparse
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# torchgeo-FREE imports (always available). TorchGeo is imported lazily below.
from base_models.xbd_crops_common import (
    scale_to_uint8, NUM_CLASSES, OUTPUT_SIZE, RESNET_SIZE, RESNET_ARCHS,
    ADAPTER_DIM, build_densenet, build_resnet6ch, inject_lora_resnet,
)


# ============================================================
# CONFIG
# ============================================================
DATA_DIR   = "/home/c/cpehesar/research/Gisborne_250m/damaged"          # holds pre/ and post/
OUT_DIR    = "/home/c/cpehesar/research/"
CKPT_DIRS  = {                                          # default checkpoint locations
    "torchgeo": "/home/c/cpehesar/research/torchGeo/checkpoints",
    "densenet": "/home/c/cpehesar/research/checkpoints/kd_ckpts",
    "resnet":   "/home/c/cpehesar/research/base_models",
}
N_SPLIT     = 2         # 500 m tile -> 2x2 grid of 250 m crops
AGG         = "any"     # tile label: any | majority | mean  (over its 250 m crops)
BATCH_TILES = 8         # tiles per forward (x N_SPLIT^2 crops each)
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# MODEL LOADING  (base models are torchgeo-free; torchgeo imported lazily)
# ============================================================
def _load_ckpt(path):
    if not os.path.exists(path):
        raise SystemExit(f"Checkpoint not found: {path}")
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(ck, dict) and "state_dict" in ck:
        return ck["state_dict"], ck
    return ck, {}


def build_model(model_name, ckpt_path):
    """Return (model.eval(), input_size, operating_threshold, tag)."""
    if model_name == "densenet121":
        path = ckpt_path or os.path.join(CKPT_DIRS["densenet"], "densenet121_xbd_best.pth")
        state, meta = _load_ckpt(path)
        model = build_densenet(num_classes=NUM_CLASSES, adapter_dim=ADAPTER_DIM,
                               pretrained=False)
        model.load_state_dict(state)
        return model.to(DEVICE).eval(), OUTPUT_SIZE, meta.get("operating_threshold"), "densenet121_xbd"

    if model_name in RESNET_ARCHS:
        path = ckpt_path or os.path.join(CKPT_DIRS["resnet"], f"{model_name}_xbd_lora_best.pth")
        state, meta = _load_ckpt(path)
        lora = meta.get("lora", {"r": 8, "alpha": 16, "dropout": 0.05})
        model = build_resnet6ch(meta.get("arch", model_name), num_classes=NUM_CLASSES,
                                freeze_backbone=True, pretrained=False)
        inject_lora_resnet(model, r=lora.get("r", 8), alpha=lora.get("alpha", 16),
                           dropout=lora.get("dropout", 0.05))
        model.load_state_dict(state)
        return model.to(DEVICE).eval(), RESNET_SIZE, meta.get("operating_threshold"), f"{model_name}_xbd_lora"

    # otherwise assume a TorchGeo registry name (needs envft)
    from torchGeo.torchgeo_common_linz import build_model as tg_build, model_input_size, MODELS
    if model_name not in MODELS:
        raise SystemExit(f"Unknown model '{model_name}'. Choose densenet121, "
                         f"{list(RESNET_ARCHS)}, or a TorchGeo name {list(MODELS)}.")
    path = ckpt_path or os.path.join(CKPT_DIRS["torchgeo"], f"{model_name}_bestFocal.pth")
    state, meta = _load_ckpt(path)
    model = tg_build(model_name, num_classes=NUM_CLASSES, freeze_backbone=True)
    model.load_state_dict(state)
    return model.to(DEVICE).eval(), model_input_size(model_name), meta.get("operating_threshold"), model_name


# ============================================================
# DATA: 500 m pre/post tile -> 2x2 grid of 250 m 6-channel crops
# (same per-crop stretch + resize the training scripts used)
# ============================================================
def build_pairs(data_dir):
    pre_dir, post_dir = os.path.join(data_dir, "pre"), os.path.join(data_dir, "post")
    pre  = {f for f in os.listdir(pre_dir)  if f.lower().endswith(".tif")}
    post = {f for f in os.listdir(post_dir) if f.lower().endswith(".tif")}
    return [(os.path.join(pre_dir, n), os.path.join(post_dir, n), n)
            for n in sorted(pre & post)]


def _read_rgb(path):
    with rasterio.open(path) as s:
        return s.read()[:3]                     # [3, H, W]


def _bounds(dim, n):
    return [round(k * dim / n) for k in range(n + 1)]


def _crop_to_6ch(pre_q, post_q, size):
    """Per-crop min-max stretch -> resize -> 0..1, stacked pre+post -> [6,size,size]."""
    def rs(a):
        t = torch.from_numpy(scale_to_uint8(a).astype(np.float32)).unsqueeze(0)
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        return t.squeeze(0) / 255.0
    return torch.cat([rs(pre_q), rs(post_q)], dim=0)


def tile_to_crops(pre_path, post_path, size, n_split):
    """Split a 500 m pre/post tile into n_split x n_split 250 m crops.
    pre and post are split by their OWN pixel grids (they differ in resolution
    but cover the same ground), so grid cell (r,c) is the same ground area."""
    pre, post = _read_rgb(pre_path), _read_rgb(post_path)
    prb, pcb = _bounds(pre.shape[1], n_split),  _bounds(pre.shape[2], n_split)
    orb, ocb = _bounds(post.shape[1], n_split), _bounds(post.shape[2], n_split)
    crops = []
    for r in range(n_split):
        for c in range(n_split):
            pq = pre[:,  prb[r]:prb[r + 1], pcb[c]:pcb[c + 1]]
            oq = post[:, orb[r]:orb[r + 1], ocb[c]:ocb[c + 1]]
            crops.append(_crop_to_6ch(pq, oq, size))
    return torch.stack(crops, dim=0)            # [n_split^2, 6, size, size]


class TileDataset(Dataset):
    def __init__(self, pairs, size, n_split):
        self.pairs, self.size, self.n = pairs, size, n_split

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        pre_p, post_p, name = self.pairs[i]
        try:
            return tile_to_crops(pre_p, post_p, self.size, self.n), name
        except Exception as e:
            print(f"  !! read failed for {name}: {type(e).__name__}: {e}", flush=True)
            # emit an all-zero tile so the batch shape stays consistent; label -> no_damage
            return torch.zeros(self.n * self.n, 6, self.size, self.size), name


# ============================================================
# AGGREGATE the 250 m crop probs -> one tile label
# ============================================================
def aggregate(probs, thr, mode):
    dmg = probs >= thr
    k = int(dmg.sum())
    if mode == "majority":
        is_damaged = k > len(probs) / 2
        p = float(probs.max())
    elif mode == "mean":
        is_damaged = float(probs.mean()) >= thr
        p = float(probs.mean())
    else:  # any
        is_damaged = bool(dmg.any())
        p = float(probs.max())
    return is_damaged, p, k


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Label ./data tiles with a trained model")
    parser.add_argument("--model", required=True,
                        help="densenet121 | resnet18/50/101 | a TorchGeo registry name")
    parser.add_argument("--ckpt", default=None, help="checkpoint path (default per model)")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the decision threshold (default: checkpoint Youden)")
    parser.add_argument("--agg", default=AGG, choices=["any", "majority", "mean"])
    parser.add_argument("--n-split", type=int, default=N_SPLIT)
    parser.add_argument("--batch-tiles", type=int, default=BATCH_TILES)
    parser.add_argument("--limit", type=int, default=None, help="only first N tiles (quick test)")
    args = parser.parse_args()

    model, size, ckpt_thr, tag = build_model(args.model, args.ckpt)
    thr = args.threshold if args.threshold is not None else (ckpt_thr if ckpt_thr is not None else 0.5)
    thr_src = ("cli" if args.threshold is not None
               else ("ckpt/youden" if ckpt_thr is not None else "default-0.5"))

    pairs = build_pairs(args.data_dir)
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        raise SystemExit(f"No pre/post pairs under {args.data_dir}")
    print(f"Model {tag} | input {size}px | {args.n_split}x{args.n_split} crops/tile | "
          f"threshold {thr:.3f} ({thr_src}) | agg={args.agg} | device={DEVICE}")
    print(f"Tiles to classify: {len(pairs)}")

    loader = DataLoader(TileDataset(pairs, size, args.n_split),
                        batch_size=args.batch_tiles, shuffle=False, num_workers=NUM_WORKERS)

    os.makedirs(args.out_dir, exist_ok=True)
    dmg_path = os.path.join(args.out_dir, f"{tag}_damage.jsonl")
    ndm_path = os.path.join(args.out_dir, f"{tag}_no_damage.jsonl")

    n_dmg = n_ndm = done = 0
    with torch.no_grad(), open(dmg_path, "w") as fd, open(ndm_path, "w") as fn:
        for crops_b, names in loader:                    # crops_b: [B, ncr, 6, S, S]
            B, ncr = crops_b.shape[0], crops_b.shape[1]
            x = crops_b.reshape(B * ncr, *crops_b.shape[2:]).to(DEVICE)
            p = F.softmax(model(x), dim=1)[:, 1].cpu().numpy().reshape(B, ncr)
            for bi, name in enumerate(names):
                probs = p[bi]
                is_damaged, p_rep, k = aggregate(probs, thr, args.agg)
                label = "damaged" if is_damaged else "no_damage"
                rec = {"image": name, "label": label, "path": f"{label}/{name}",
                       "p_damaged": round(p_rep, 4), "damaged_crops": k,
                       "total_crops": int(ncr),
                       "crop_probs": [round(float(v), 4) for v in probs]}
                if is_damaged:
                    fd.write(json.dumps(rec) + "\n"); n_dmg += 1
                else:
                    fn.write(json.dumps(rec) + "\n"); n_ndm += 1
            done += B
            if done % (args.batch_tiles * 25) < B:
                print(f"  {done}/{len(pairs)} tiles  (damaged {n_dmg}, no_damage {n_ndm})",
                      flush=True)

    print("\n" + "=" * 60)
    print(f"  {tag}: {len(pairs)} tiles -> damaged {n_dmg}, no_damage {n_ndm}")
    print("=" * 60)
    print(f"Saved {dmg_path}\n      {ndm_path}")


if __name__ == "__main__":
    main()
