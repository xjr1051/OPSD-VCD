from dataclasses import dataclass
import random
import re
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter


@dataclass
class ImagePerturbationConfig:
    noise_std: float = 25.0
    noise_steps: int = 500
    gamma: float = 0.1
    mask_ratio: float = 0.25
    mask_min_ratio: Optional[float] = None
    mask_max_ratio: Optional[float] = None
    mask_count: int = 1
    blur_radius: float = 2.0
    fg_mask_keep_ratio: float = 0.35
    fg_mask_center_bias: float = 0.15
    object_bbox_field: str = ""
    target_object_field: str = ""
    object_bbox_label_field: str = ""


def normalize_perturbation_pair(pair: Tuple[str, str]) -> Tuple[str, str]:
    teacher_tag, student_tag = pair
    if teacher_tag == "clean" and student_tag == "mask":
        return ("mask", "clean")
    if teacher_tag == "clean" and student_tag in {"object", "objectmask", "fgmask", "object-mask", "foreground"}:
        return ("fgmask", "clean")
    return pair


def add_diffusion_noise_tensor(image_tensor: torch.Tensor, noise_step: int) -> torch.Tensor:
    num_steps = 1000
    step = int(np.clip(noise_step, 0, num_steps - 1))

    work_dtype = torch.float32
    betas = torch.linspace(-6, 6, num_steps, device=image_tensor.device, dtype=work_dtype)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    noise = torch.randn_like(image_tensor)
    alpha_t = alphas_bar_sqrt[step].to(dtype=image_tensor.dtype)
    sigma_t = one_minus_alphas_bar_sqrt[step].to(dtype=image_tensor.dtype)
    return alpha_t * image_tensor + sigma_t * noise


def _sample_mask_side_ratio(cfg: ImagePerturbationConfig) -> float:
    min_ratio = cfg.mask_min_ratio if cfg.mask_min_ratio is not None else cfg.mask_ratio
    max_ratio = cfg.mask_max_ratio if cfg.mask_max_ratio is not None else cfg.mask_ratio
    min_ratio = float(np.clip(min_ratio, 1e-4, 1.0))
    max_ratio = float(np.clip(max_ratio, min_ratio, 1.0))
    if abs(max_ratio - min_ratio) < 1e-8:
        return min_ratio
    return random.uniform(min_ratio, max_ratio)


def apply_mask_perturbation(image: Image.Image, cfg: ImagePerturbationConfig) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).copy()
    h, w = arr.shape[:2]
    mask_count = max(1, int(cfg.mask_count))
    for _ in range(mask_count):
        side_ratio = _sample_mask_side_ratio(cfg)
        mask_h = max(1, int(h * side_ratio))
        mask_w = max(1, int(w * side_ratio))
        top = random.randint(0, max(0, h - mask_h))
        left = random.randint(0, max(0, w - mask_w))
        arr[top : top + mask_h, left : left + mask_w] = 0
    return Image.fromarray(arr)


def _normalize_bbox(bbox, width: int, height: int):
    if isinstance(bbox, dict):
        if "bbox" in bbox:
            bbox = bbox["bbox"]
        elif all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            bbox = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
        elif all(k in bbox for k in ("x", "y", "w", "h")):
            x = bbox["x"]
            y = bbox["y"]
            w = bbox["w"]
            h = bbox["h"]
            bbox = [x, y, x + w, y + h]

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    x1, y1, x2, y2 = bbox
    try:
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)
    except (TypeError, ValueError):
        return None

    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height

    left = int(round(max(0.0, min(x1, x2))))
    right = int(round(min(float(width), max(x1, x2))))
    top = int(round(max(0.0, min(y1, y2))))
    bottom = int(round(min(float(height), max(y1, y2))))

    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _normalize_text_tokens(value) -> set[str]:
    if value is None:
        return set()

    texts = []
    if isinstance(value, str):
        texts = [value]
    elif isinstance(value, dict):
        for key in ("label", "name", "object", "target", "text"):
            if key in value:
                texts.append(str(value[key]))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, (str, int, float)):
                texts.append(str(item))
            elif isinstance(item, dict):
                for key in ("label", "name", "object", "target", "text"):
                    if key in item:
                        texts.append(str(item[key]))
                        break
    else:
        texts = [str(value)]

    tokens = set()
    for text in texts:
        norm = str(text).lower().replace("_", " ").replace("-", " ")
        parts = [p for p in re.split(r"[^a-z0-9]+", norm) if p]
        for p in parts:
            tokens.add(p)
            if p.endswith("s") and len(p) > 3:
                tokens.add(p[:-1])
    return tokens


def _label_matches_target(label_value, target_tokens: set[str]) -> bool:
    if not target_tokens:
        return True
    label_tokens = _normalize_text_tokens(label_value)
    if not label_tokens:
        return False
    return len(label_tokens.intersection(target_tokens)) > 0


def _extract_target_tokens(feature: dict, cfg: ImagePerturbationConfig) -> set[str]:
    if not cfg.target_object_field or cfg.target_object_field not in feature:
        return set()
    return _normalize_text_tokens(feature[cfg.target_object_field])


def _extract_bbox_label(raw_box, raw_labels, idx: int, cfg: ImagePerturbationConfig):
    label_value = None
    if isinstance(raw_box, dict):
        if cfg.object_bbox_label_field and cfg.object_bbox_label_field in raw_box:
            label_value = raw_box[cfg.object_bbox_label_field]
        if label_value is None:
            for key in ("label", "labels", "name", "category", "class", "object"):
                if key in raw_box:
                    label_value = raw_box[key]
                    break

    if label_value is None and isinstance(raw_labels, Sequence) and idx < len(raw_labels):
        label_value = raw_labels[idx]
    return label_value


def foreground_mask_from_feature(image: Image.Image, feature: dict, cfg: ImagePerturbationConfig):
    if not cfg.object_bbox_field or cfg.object_bbox_field not in feature:
        return None

    width, height = image.size
    raw_boxes = feature[cfg.object_bbox_field]
    raw_labels = None
    if cfg.object_bbox_label_field and cfg.object_bbox_label_field in feature:
        raw_labels = feature[cfg.object_bbox_label_field]

    target_tokens = _extract_target_tokens(feature, cfg)
    use_target_filter = len(target_tokens) > 0
    if isinstance(raw_boxes, (list, tuple)) and len(raw_boxes) == 4 and not isinstance(raw_boxes[0], (list, tuple)):
        raw_boxes = [raw_boxes]
    if not isinstance(raw_boxes, (list, tuple)):
        return None

    mask = np.zeros((height, width), dtype=bool)
    matched_target_box_count = 0
    for idx, raw_box in enumerate(raw_boxes):
        if use_target_filter:
            label_value = _extract_bbox_label(raw_box, raw_labels, idx, cfg)
            if not _label_matches_target(label_value, target_tokens):
                continue
            matched_target_box_count += 1

        box = _normalize_bbox(raw_box, width, height)
        if box is None:
            continue
        left, top, right, bottom = box
        mask[top:bottom, left:right] = True

    if use_target_filter and matched_target_box_count == 0:
        return None
    if mask.sum() == 0:
        return None
    return mask


def foreground_mask_saliency(image: Image.Image, cfg: ImagePerturbationConfig) -> np.ndarray:
    arr = np.asarray(image).astype(np.float32) / 255.0
    h, w = arr.shape[:2]
    if h <= 2 or w <= 2:
        return np.ones((h, w), dtype=bool)

    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy)
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-6)

    border_pixels = np.concatenate([arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0)
    border_mean = border_pixels.mean(axis=0, keepdims=True)
    color_dist = np.linalg.norm(arr - border_mean, axis=2)
    color_dist = (color_dist - color_dist.min()) / (color_dist.max() - color_dist.min() + 1e-6)

    yy, xx = np.mgrid[0:h, 0:w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    sy = max(h / 3.0, 1.0)
    sx = max(w / 3.0, 1.0)
    center_prior = np.exp(-(((yy - cy) ** 2) / (2.0 * sy * sy) + ((xx - cx) ** 2) / (2.0 * sx * sx)))

    saliency = 0.6 * color_dist + 0.4 * grad + cfg.fg_mask_center_bias * center_prior
    keep_ratio = float(np.clip(cfg.fg_mask_keep_ratio, 0.05, 0.9))
    keep_count = max(1, int(round(h * w * keep_ratio)))
    flat = saliency.reshape(-1)
    top_idx = np.argpartition(flat, -keep_count)[-keep_count:]

    mask = np.zeros(flat.shape[0], dtype=np.uint8)
    mask[top_idx] = 255
    mask = mask.reshape(h, w)

    mask_img = Image.fromarray(mask, mode="L")
    mask_img = mask_img.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    smooth_mask = np.asarray(mask_img) > 127
    if smooth_mask.sum() == 0:
        return np.ones((h, w), dtype=bool)
    return smooth_mask


def apply_foreground_mask(image: Image.Image, cfg: ImagePerturbationConfig, feature: Optional[dict] = None) -> Image.Image:
    mask = None
    if feature is not None:
        mask = foreground_mask_from_feature(image, feature, cfg)
    if mask is None:
        mask = foreground_mask_saliency(image, cfg)

    arr = np.asarray(image.convert("RGB")).copy()
    arr[~mask] = 0
    return Image.fromarray(arr)


def apply_image_perturbation(
    image: Image.Image,
    view_tag: str,
    cfg: ImagePerturbationConfig,
    feature: Optional[dict] = None,
) -> Image.Image:
    tag = str(view_tag).strip().lower()
    image = image.convert("RGB")

    if tag == "clean":
        return image
    if tag == "noise":
        arr = np.asarray(image).astype(np.float32)
        noise = np.random.normal(0.0, cfg.noise_std, size=arr.shape)
        arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(arr)
    if tag == "mask":
        return apply_mask_perturbation(image, cfg)
    if tag == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=cfg.blur_radius))
    if tag in {"fgmask", "object", "objectmask", "object-mask", "foreground"}:
        return apply_foreground_mask(image, cfg, feature=feature)
    raise ValueError(f"Unsupported perturbation tag: {view_tag}")
