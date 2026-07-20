# Grad-CAM utilities for CNNs and transformers.

# Handles three activation layouts:
#   "bchw"   - CNN feature map [B, C, H, W]              (ResNet layer4, DenseNet features)
#   "bhwc"   - channels-last   [B, H, W, C]              (torchvision Swin-V2 stages)
#   "tokens" - transformer     [B, N(+cls), C]           (ViT blocks; cls token dropped)
# ============================================================
import os
import numpy as np
import torch
import torch.nn.functional as F


def _mpl():
    """Headless matplotlib with a writable cache dir on NeSI."""
    os.environ.setdefault("MPLCONFIGDIR", "/home/c/cpehesar/research/cache/mpl")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


class GradCAM:
    """Hooks a target layer's forward activations + backward gradients and
    produces a normalized [H, W] heatmap for a chosen class."""
    def __init__(self, model, target_layer, layout="bchw"):
        self.model = model
        self.layout = layout
        self.acts = None
        self.grads = None
        # forward hook saves the activation and registers a grad hook on the SAME
        # output tensor. (A full_backward_hook breaks on nets that apply an
        # in-place op to the layer's output, e.g. DenseNet's relu(features).)
        self._h1 = target_layer.register_forward_hook(self._fwd)

    def _fwd(self, m, inp, out):
        self.acts = out.detach().clone()
        if isinstance(out, torch.Tensor) and out.requires_grad:
            out.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self.grads = grad.detach()

    def remove(self):
        self._h1.remove()

    def _to_bchw(self, t):
        if self.layout == "bchw":
            return t
        if self.layout == "bhwc":
            return t.permute(0, 3, 1, 2).contiguous()
        if self.layout == "tokens":
            B, N, C = t.shape
            h = int(round(N ** 0.5))
            if h * h != N:                       # drop a leading cls/dist token
                t = t[:, 1:, :]; N -= 1; h = int(round(N ** 0.5))
            return t.transpose(1, 2).reshape(B, C, h, h).contiguous()
        raise ValueError(f"unknown layout {self.layout}")

    def __call__(self, x, class_idx=None):
        """x: [1, C, H, W]. Returns (cam[H,W] in 0..1, predicted_class, probs)."""
        was_training = self.model.training
        self.model.eval()
        x = x.clone().detach().requires_grad_(True)   # graph even if params are frozen
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)                     # [1, num_classes]
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            logits[0, class_idx].backward()

        acts = self._to_bchw(self.acts.detach())       # [1, C, h, w]
        grads = self._to_bchw(self.grads.detach())
        weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP over spatial
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))            # [1,1,h,w]
        cam = F.interpolate(cam, size=(x.shape[-2], x.shape[-1]),
                            mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        if was_training:
            self.model.train()
        return cam.cpu().numpy(), class_idx, probs


def post_rgb_from_6ch(t):
    """The post (after) RGB half of a [6, S, S] tensor, as HxWx3 float 0..1."""
    return t[3:6].permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def _jet(cam):
    import matplotlib
    try:
        cmap = matplotlib.colormaps["jet"]
    except Exception:                               # older matplotlib
        import matplotlib.cm as cm
        cmap = cm.get_cmap("jet")
    return cmap(cam)[..., :3]


def save_gradcam_overlay(img_rgb, cam, title, path, alpha=0.45):
    """Blend the Grad-CAM heatmap over the image and save ONLY that overlay."""
    plt = _mpl()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    over = np.clip((1 - alpha) * img_rgb + alpha * _jet(cam), 0, 1)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(over); ax.axis("off")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved Grad-CAM -> {path}")


def run_gradcam(model, crops_by_name, wanted, target_layer, layout,
                model_tag, out_dir, device):
    """For each requested crop name, save a predicted-class Grad-CAM overlay
    (on the post image). Robust: a failure on one crop/model is warned, not fatal."""
    if not wanted:
        return
    if target_layer is None:
        print("  Grad-CAM: no target layer resolved for this model; skipping.")
        return
    cam = GradCAM(model, target_layer, layout=layout)
    try:
        for name in wanted:
            t = crops_by_name.get(name)
            if t is None:
                print(f"  Grad-CAM: '{name}' not in test set; skipping.")
                continue
            try:
                heat, cls, probs = cam(t.unsqueeze(0).to(device))
                label = "damaged" if cls == 1 else "no_damage"
                title = f"{name}\npred={label}  P(damaged)={probs[1]:.2f}"
                safe = name.replace("/", "_")
                path = os.path.join(out_dir, f"{model_tag}_{safe}.png")
                save_gradcam_overlay(post_rgb_from_6ch(t), heat, title, path)
            except Exception as e:
                print(f"  Grad-CAM FAILED for '{name}': {type(e).__name__}: {e}")
    finally:
        cam.remove()
