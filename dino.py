import math
import os
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageDraw
from transformers import AutoImageProcessor, AutoModel


DEFAULT_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_PERCENTILE = 80
DEFAULT_BLUR = 1.2


# Simple cache so we only load once per runtime
@lru_cache(maxsize=1)
def _load_dino_model(model_id: str = DEFAULT_MODEL_ID):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    try:
        model.set_attn_implementation("eager")
    except Exception:
        try:
            model.config.attn_implementation = "eager"
        except Exception:
            pass
    model.config.output_attentions = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"✅ Loaded DINOv3 model: {model_id} ({device})")
    return model, processor, device


def infer_garment_type_from_path(path: str) -> Optional[str]:
    parts = os.path.normpath(path).split(os.sep)
    lower = [p.lower() for p in parts]
    for idx in range(len(lower) - 1, -1, -1):
        if lower[idx] == "siksilk" and idx + 1 < len(parts):
            return parts[idx + 1].replace("_", " ").lower()
    return None


TOP_GARMENTS = {"t-shirt", "tshirt", "shirt", "sweater", "hoodie", "track top", "vest", "polo"}
BOTTOM_GARMENTS = {"shorts", "joggers", "jogger-trousers", "trousers", "jeans", "swimwear"}


def garment_position_hint(garment_type: str) -> str:
    t = (garment_type or "").strip().lower()
    if t in BOTTOM_GARMENTS or "jogger" in t or "short" in t or "jean" in t or "trouser" in t:
        return "bottom"
    if t in TOP_GARMENTS or t in ("hoodie", "sweater", "track top", "polo", "vest", "shirt", "t-shirt", "tshirt"):
        return "top"
    return "full"


def _apply_vertical_bias(heatmap: np.ndarray, garment_hint: str | None):
    pref = garment_position_hint(garment_hint)
    if pref == "top":
        bias = np.linspace(1.15, 0.85, heatmap.shape[0])[:, None]
        return heatmap * bias
    if pref == "bottom":
        bias = np.linspace(0.9, 1.15, heatmap.shape[0])[:, None]
        return heatmap * bias
    return heatmap


def _to_grid_from_attn(attn: torch.Tensor, model, inputs) -> Tuple[np.ndarray, int, int]:
    # attn: (heads, tokens, tokens)
    pixel_values = inputs.get("pixel_values")
    if pixel_values is None:
        raise ValueError("pixel_values missing from processor output.")

    patch_size = getattr(model.config, "patch_size", 16)
    if isinstance(patch_size, (tuple, list)):
        patch_size = patch_size[0]
    h_patches = max(1, pixel_values.shape[-2] // int(patch_size))
    w_patches = max(1, pixel_values.shape[-1] // int(patch_size))
    num_patches = h_patches * w_patches

    num_reg = int(getattr(model.config, "num_register_tokens", 0) or 0)
    start = 1 + num_reg  # skip CLS and register tokens
    end = start + num_patches
    if end > attn.shape[-1]:
        end = attn.shape[-1]
        start = max(1, end - num_patches)

    cls_attn = attn[:, 0, start:end]  # (heads, patches)
    cls_attn = cls_attn.mean(0)

    if cls_attn.shape[-1] != num_patches:
        num_patches = cls_attn.shape[-1]
        h_patches = w_patches = int(round(math.sqrt(max(1, num_patches))))

    heat = cls_attn.reshape(1, 1, h_patches, w_patches)
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-6)
    return heat, h_patches, w_patches


def compute_dino_heatmap(
    img: Image.Image,
    garment_hint: str | None = None,
    score_percentile: float = DEFAULT_PERCENTILE,
    blur_sigma: float = DEFAULT_BLUR,
):
    model, processor, device = _load_dino_model()
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    atts = outputs.attentions[-1] if hasattr(outputs, "attentions") and outputs.attentions else None
    if atts is None:
        raise ValueError("DINOv3 model did not return attentions; ensure transformers >=4.40")

    attn = atts[0]  # (heads, tokens, tokens)
    heat, h_patches, w_patches = _to_grid_from_attn(attn, model, inputs)
    heat = F.interpolate(
        heat,
        size=(img.height, img.width),
        mode="bicubic",
        align_corners=False,
    )[0, 0].cpu().numpy()

    heat = np.clip(heat, 0.0, 1.0)
    heat = _apply_vertical_bias(heat, garment_hint)
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-6)

    if blur_sigma and blur_sigma > 0:
        try:
            import cv2

            heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
            heat = np.clip(heat, 0.0, 1.0)
        except Exception:
            pass

    thresh = np.percentile(heat, score_percentile)
    mask = heat >= thresh
    if not mask.any():
        mask = heat >= np.percentile(heat, 60)

    ys, xs = np.where(mask)
    if xs.size == 0:
        cx, cy = img.width // 2, img.height // 2
        span = int(min(img.width, img.height) * 0.35)
        raw_box = [cx - span // 2, cy - span // 2, cx + span // 2, cy + span // 2]
    else:
        raw_box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]

    return heat, raw_box


def expand_square_bbox(box, image_size, padding=0):
    x0, y0, x1, y1 = [int(v) for v in box]
    w_img, h_img = image_size
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(w_img, x1 + padding)
    y1 = min(h_img, y1 + padding)
    bw, bh = x1 - x0, y1 - y0
    side = max(bw, bh, 1)
    if bw < side:
        extra = side - bw
        x0 = max(0, x0 - extra // 2)
        x1 = min(w_img, x1 + (extra - extra // 2))
    if bh < side:
        extra = side - bh
        y0 = max(0, y0 - extra // 2)
        y1 = min(h_img, y1 + (extra - extra // 2))
    return [int(x0), int(y0), int(x1), int(y1)]


def detect_garment_region_with_dinov3(
    img: Image.Image,
    garment_hint: str | None = None,
    score_percentile: float = DEFAULT_PERCENTILE,
    padding: int = 40,
):
    heat, raw_box = compute_dino_heatmap(
        img,
        garment_hint=garment_hint,
        score_percentile=score_percentile,
        blur_sigma=DEFAULT_BLUR,
    )
    bbox = expand_square_bbox(raw_box, img.size, padding=padding)
    return bbox, heat


def heatmap_to_pil(heatmap: np.ndarray):
    hm = heatmap - heatmap.min()
    hm = hm / (hm.max() + 1e-6)
    hm_img = Image.fromarray((hm * 255).astype(np.uint8), mode="L")
    return ImageOps.colorize(hm_img, black="black", white="#ff7f0e").convert("RGB")


def draw_bbox(img: Image.Image, bbox, color=(255, 99, 71), width: int = 6):
    vis = img.copy()
    d = ImageDraw.Draw(vis)
    d.rectangle(bbox, outline=color, width=width)
    return vis
