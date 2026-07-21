# Label-Efficient Damage Assessment of Non-Building Infrastructure : Benchmark on Cyclone Gabrielle NZ Dataset 

Benchmark and code for detecting cyclone damage to **roads, bridges and land** from pre/post aerial imagery, using frozen remote-sensing foundation models and knowledge distillation.

Built on imagery of **Cyclone Gabrielle** (New Zealand, February 2023) from Land Information New Zealand (LINZ).

<!-- ─────────────────────────────────────────────
FIGURE SLOT 1 — HERO / TEASER IMAGE
Put a single wide image here: a pre/post tile pair side by side with a
damaged road or bridge, ideally with the model's prediction overlaid.
This is the first thing visitors see, so make it the most striking example.
Recommended: 1200x400 px, saved as docs/figures/teaser.png
───────────────────────────────────────────── -->

![Teaser](docs/figures/teaser.png)

---

## Overview

Most damage-assessment models are trained on **buildings** in **satellite** imagery (xBD/xView2). Cyclones and floods, though, mostly damage things that are not buildings, and response teams increasingly fly **aerial** surveys at sub-metre resolution. That is a double domain gap: different target, different sensor.

This repo asks two questions:

1. With only ~100 labels per class, what is the best way to adapt a pretrained model to this task — supervised transfer from xBD, a frozen foundation-model linear probe, or LoRA fine-tuning?
2. Can the winning model be compressed into something small enough to run cheaply, without losing damage recall?

**Short answers.** A frozen Swin-V2-B pretrained on aerial NAIP imagery wins in-region (0.9767 accuracy, 0.9943 AUC). Distilling it into a DenseNet-121 keeps the same damage recall (0.9583) at ~11x fewer parameters, and transfers better to an unseen region.

---

## Method

<!-- ─────────────────────────────────────────────
FIGURE SLOT 2 — ARCHITECTURE / PIPELINE DIAGRAM
This is the most important figure in the repo. Show:
  pre tile + post tile → 6-channel stack → [two arms] → damage label
  Arm A: ImageNet → xBD → LoRA
  Arm B: frozen TorchGeo backbone + linear probe
  Then: best backbone (teacher) → knowledge distillation → DenseNet student
Same diagram as Fig. 1 in the paper. Export at 2x resolution so it stays
readable on GitHub. Recommended: docs/figures/pipeline.png
───────────────────────────────────────────── -->

![Pipeline](docs/figures/pipeline.png)

### Input representation

Each sample is a co-registered pre/post tile pair (250 m x 250 m). The two RGB images are stacked into a single **6-channel tensor**, so a standard encoder sees appearance and change at once. Only the first conv / patch-embedding layer is replaced, to accept 6 channels instead of 3.

### Adaptation strategies

| Strategy | Pretraining | Trainable parts | Why |
|---|---|---|---|
| Supervised transfer | ImageNet (natural) | Backbone via xBD + LoRA | Large domain gap, features need updating |
| **Linear probe** | Aerial foundation model | 6-ch stem + linear head | Features already aligned; freezing avoids overfitting |
| LoRA | Aerial foundation model | Stem + head + adapters | Extra capacity risks overfitting on ~180 labels |


## Dataset

| | |
|---|---|
| Source | LINZ open aerial imagery (CC BY 4.0) |
| Pre-event | Hawke's Bay rural aerial, 2021–2022, 0.3 m/px |
| Post-event | Cyclone Gabrielle, 2023, 0.1 m/px |
| Tiling | 250 m x 250 m grid, WMTS streaming (QGIS + GDAL/rasterio) |
| Unlabelled corpus | ~7,000 pre/post pairs |
| External region | Gisborne (0.3 m pre, 0.2 m post) |

Hand-labelled subset, binary `damaged` vs `no_damage`:

| Split | Damaged | No damage | Total |
|---|---|---|---|
| Train | 55 | 36 | 91 |
| Hold | 33 | 59 | 92 |
| CV pool (train + hold) | 88 | 95 | 183 |
| Test | 48 | 81 | 129 |
| Gisborne (external) | 23 | 52 | 75 |

<!-- ─────────────────────────────────────────────
FIGURE SLOT 3 — DATASET EXAMPLES GRID
A grid of example pre/post pairs, one row per damage type:
  row 1: washed-out / silt-covered road
  row 2: damaged bridge
  row 3: land slip or flooding
  row 4: a no-damage pair (important — shows what the negative class looks like)
Label each column "Pre (2021–22)" and "Post (2023)".
Recommended: docs/figures/dataset_examples.png
───────────────────────────────────────────── -->

![Dataset examples](docs/figures/dataset_examples.png)

**Note on labels.** The `damaged` class bundles roads, bridges and land damage into one label. That is a scoping choice for this first benchmark; per-type labels are future work.

---

## Results

All numbers are on the held-out test set (129 pairs), after stratified 5-fold cross-validation on the pooled 183-pair set.

### Cross-domain transfer baseline (ImageNet → xBD → LoRA)

| Model | Acc. | Prec. | Rec. | F1 | AUC |
|---|---|---|---|---|---|
| ResNet18 | 0.8527 | 1.0000 | 0.6042 | 0.7533 | 0.9792 |
| ResNet50 | 0.9147 | 0.9744 | 0.7917 | 0.8736 | 0.9861 |
| ResNet101 | 0.9457 | 1.0000 | 0.8542 | 0.9213 | 0.9658 |
| DenseNet121 | 0.9457 | 0.9767 | 0.8750 | 0.9231 | 0.9928 |

### Foundation-model linear probe (TorchGeo backbones, frozen)

| Backbone | Acc. | Prec. | Rec. | F1 | AUC | Params (M) |
|---|---|---|---|---|---|---|
| resnet18_sentinel2 | 0.8915 | 0.9722 | 0.7292 | 0.8333 | 0.9825 | 11.7 |
| resnet50_sentinel2 | 0.9147 | 0.9744 | 0.7917 | 0.8736 | 0.9702 | 25.6 |
| vit_small_sentinel2 | 0.9225 | 0.9318 | 0.8542 | 0.8913 | 0.9792 | 22.0 |
| swin_v2_t_satlas_s2 | 0.9147 | 0.9512 | 0.8125 | 0.8764 | 0.9805 | 28.4 |
| swin_v2_b_satlas_s2 | 0.9380 | 0.9348 | 0.8958 | 0.9149 | 0.9771 | 87.9 |
| resnet50_fmow_rgb | 0.9457 | 0.9767 | 0.8750 | 0.9231 | 0.9905 | 25.6 |
| **swin_v2_b_satlas_naip** | **0.9767** | **0.9787** | **0.9583** | **0.9684** | **0.9943** | 87.9 |

**Aerial pretraining beats satellite pretraining.** Same architecture, ~4 accuracy points apart (NAIP vs Sentinel-2). A 25.6 M aerial-pretrained ResNet50 beats an 87.9 M Sentinel-2 Swin, so the pretraining sensor matters more than model size.

### Distillation and efficiency

| Model | Params (M) | GFLOPs | Acc. | Rec. | F1 | AUC |
|---|---|---|---|---|---|---|
| Swin-V2-B NAIP (teacher) | 87.9 | ~15.4 | 0.9767 | 0.9583 | 0.9684 | 0.9943 |
| DenseNet121, no KD | 8.0 | ~2.9 | 0.9457 | 0.8750 | 0.9231 | 0.9928 |
| **DenseNet121 + KD** | **8.0** | **~2.9** | **0.9535** | **0.9583** | **0.9388** | 0.9905 |

<!-- ─────────────────────────────────────────────
FIGURE SLOT 4 — GRAD-CAM COMPARISON
Rows = example tiles, columns = [post image | DenseNet student CAM | Swin probe CAM].
Include at least one FALSE POSITIVE row where Swin lights up on undamaged
ground and DenseNet stays quiet — that row is the visual evidence for the
precision gap in the Gisborne table below.
Recommended: docs/figures/gradcam.png
───────────────────────────────────────────── -->

![Grad-CAM comparison](docs/figures/gradcam.png)

### Cross-region generalisation (Gisborne, unseen)

| Model | Acc. | Prec. | Rec. | F1 |
|---|---|---|---|---|
| DenseNet121 + KD (student) | 0.8800 | 0.7692 | 0.8696 | 0.8163 |
| Swin-V2-B NAIP (probe) | 0.8267 | 0.6389 | 0.9665 | 0.7797 |

The two models fail differently. The Swin probe keeps its recall but loses precision, flagging many undamaged tiles. The distilled student stays balanced. Freezing the backbone protects the *features*, but the linear *head* can still overfit to the source region.

<!-- ─────────────────────────────────────────────
FIGURE SLOT 5 — TRAINING / VALIDATION LOSS CURVES
Two panels side by side, on the SAME y-axis scale:
  left  = distilled DenseNet student (stable, curves close together)
  right = Swin linear probe (widening train/val gap = head overfitting)
Recommended: docs/figures/loss_curves.png
───────────────────────────────────────────── -->

![Loss curves](docs/figures/loss_curves.png)

---

## Setup

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Main dependencies: `torch`, `torchgeo`, `timm`, `scikit-learn`, `rasterio`, `gdal`, `numpy`, `pandas`.

---



## Evaluation notes

Read the numbers with these caveats in mind:

- **Small test sets.** 129 in-region and 75 external tiles. One flipped prediction moves accuracy by about a point, so differences between the top models are provisional.
- **Single seed.** All runs use seed 42. Multi-seed runs and confidence intervals are the next step.
- **Oracle threshold.** Operating-point metrics (precision / recall / F1) use the threshold that maximises F1 on the same out-of-fold predictions they score, so treat them as an upper bound. Pooled AUC is threshold-free and is the primary ranking metric.
- **Gisborne is confounded.** It changes both region and resolution (0.2 m vs 0.1 m post-event), so the drop cannot be attributed to geography alone.



## Citation

```bibtex
@inproceedings{danansooriya2027label,
  title     = {Label-Efficient Cross-Domain Damage Assessment of Non-Building
               Infrastructure: A Benchmark on Cyclone Gabrielle Aerial Imagery},
  author    = {Danansooriya, Chanuka and Ranasinghe, Malintha and
               Prasanna, Raj and Aththanayake, Chamodya},
  booktitle = {Proc. IEEE Int. Geoscience and Remote Sensing Symposium (IGARSS)},
  year      = {2027}
}
```

---

## Acknowledgments

Supported by the Natural Hazards Commission Toka Tū Ake, New Zealand.
Aerial imagery © Land Information New Zealand, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Pretrained backbones from [TorchGeo](https://github.com/microsoft/torchgeo) and [SatlasPretrain](https://github.com/allenai/satlas).

## License

Code released under the MIT License. Derived imagery follows the LINZ CC BY 4.0 terms.
