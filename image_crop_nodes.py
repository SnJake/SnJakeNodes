import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

import folder_paths
import node_helpers


@dataclass
class CropRect:
    x: int
    y: int
    width: int
    height: int

    def clamp(self, max_w: int, max_h: int) -> "CropRect":
        x = max(0, min(self.x, max_w - 1 if max_w > 0 else 0))
        y = max(0, min(self.y, max_h - 1 if max_h > 0 else 0))

        width = max(1, self.width)
        height = max(1, self.height)

        if x + width > max_w:
            width = max(1, max_w - x)
        if y + height > max_h:
            height = max(1, max_h - y)

        return CropRect(x=x, y=y, width=width, height=height)

    def to_dict(self, image_w: int, image_h: int, image_name: str) -> Dict[str, Any]:
        return {
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
            "orig_width": int(image_w),
            "orig_height": int(image_h),
            "image": image_name,
        }


def _load_image_rgb(image_path: str) -> Image.Image:
    img = node_helpers.pillow(Image.open, image_path)
    img = node_helpers.pillow(ImageOps.exif_transpose, img)

    if getattr(img, "is_animated", False):
        try:
            img.seek(0)
        except EOFError:
            pass

    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    if img.mode == "RGBA":
        # Drop alpha channel while keeping RGB content
        r, g, b, _ = img.split()
        img = Image.merge("RGB", (r, g, b))
    else:
        img = img.convert("RGB")

    return img


def _tensor_from_pil(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    return torch.from_numpy(arr)[None, ...]


def _parse_crop_json(raw: str) -> Dict[str, float]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return {}


def _default_crop_for_image(width: int, height: int) -> CropRect:
    if width <= 0 or height <= 0:
        return CropRect(0, 0, 1, 1)

    side = min(width, height)
    # default to centered square
    x = max(0, (width - side) // 2)
    y = max(0, (height - side) // 2)
    return CropRect(x=x, y=y, width=side, height=side)


def _resolve_crop_rect(
    data: Dict[str, Any],
    image_size: Tuple[int, int],
) -> CropRect:
    img_w, img_h = image_size

    if not data:
        return _default_crop_for_image(img_w, img_h)

    x = int(round(data.get("x", 0)))
    y = int(round(data.get("y", 0)))
    width = int(round(data.get("width", img_w)))
    height = int(round(data.get("height", img_h)))

    prev_w = float(data.get("orig_width") or 0) or img_w
    prev_h = float(data.get("orig_height") or 0) or img_h

    if prev_w != img_w and prev_w > 0:
        scale_x = img_w / prev_w
        x = int(round(x * scale_x))
        width = int(round(width * scale_x))
    if prev_h != img_h and prev_h > 0:
        scale_y = img_h / prev_h
        y = int(round(y * scale_y))
        height = int(round(height * scale_y))

    rect = CropRect(x=x, y=y, width=max(1, width), height=max(1, height))
    return rect.clamp(img_w, img_h)


class SnJakeInteractiveCropLoader:
    CATEGORY = "😎 SnJake/Image"
    RETURN_TYPES = ("IMAGE", "IMAGE", "SJ_BBOX")
    RETURN_NAMES = ("crop", "original", "crop_info")
    FUNCTION = "load_and_crop"

    @classmethod
    def INPUT_TYPES(cls):
        files = []
        try:
            files = folder_paths.get_filename_list("input")
            files = folder_paths.filter_files_content_types(files, ["image"])
        except Exception:
            files = []

        if not files:
            files = []

        default_meta = json.dumps(
            {
                "x": 0,
                "y": 0,
                "width": 0,
                "height": 0,
                "orig_width": 0,
                "orig_height": 0,
                "image": "",
            }
        )

        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "crop_json": ("STRING", {"default": default_meta, "widget": "HIDDEN"}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, image: str, crop_json: str):
        image_path = folder_paths.get_annotated_filepath(image)
        h = hashlib.sha256()

        h.update(image.encode("utf-8", errors="ignore"))
        crop_dict = _parse_crop_json(crop_json)
        for key in ("x", "y", "width", "height", "orig_width", "orig_height"):
            h.update(str(crop_dict.get(key, "")).encode("utf-8", errors="ignore"))

        if os.path.isfile(image_path):
            with open(image_path, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    h.update(chunk)
        return h.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, image: str, crop_json: str):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        _parse_crop_json(crop_json)  # Ensure it's decodable
        return True

    def load_and_crop(self, image: str, crop_json: str):
        image_path = folder_paths.get_annotated_filepath(image)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        pil_img = _load_image_rgb(image_path)
        orig_w, orig_h = pil_img.size

        crop_dict = _parse_crop_json(crop_json)
        rect = _resolve_crop_rect(crop_dict, (orig_w, orig_h))

        cropped_img = pil_img.crop((rect.x, rect.y, rect.x + rect.width, rect.y + rect.height))

        crop_tensor = _tensor_from_pil(cropped_img)
        orig_tensor = _tensor_from_pil(pil_img)
        crop_info = rect.to_dict(orig_w, orig_h, image)

        return (crop_tensor, orig_tensor, crop_info)


class SnJakeImagePatchNode:
    CATEGORY = "😎 SnJake/Image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE", {}),
                "fragment": ("IMAGE", {}),
                "crop_info": ("SJ_BBOX", {}),
                "resize_fragment": ("BOOLEAN", {"default": True, "label_on": "Resize", "label_off": "Keep"}),
            }
        }

    @staticmethod
    def _normalize_tensor(image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            return image.unsqueeze(0)
        return image

    @staticmethod
    def _ensure_same_batch(base: torch.Tensor, frag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if base.shape[0] == frag.shape[0]:
            return base, frag
        if frag.shape[0] == 1:
            frag = frag.expand(base.shape[0], *frag.shape[1:])
            return base, frag
        raise ValueError("Fragment batch size does not match base image batch size.")

    @staticmethod
    def _parse_crop_info(crop_info: Any) -> CropRect:
        if isinstance(crop_info, CropRect):
            return crop_info

        source = crop_info
        if isinstance(crop_info, str):
            source = _parse_crop_json(crop_info)

        if not isinstance(source, dict):
            raise ValueError("crop_info must be a dict or JSON string")

        rect = CropRect(
            x=int(round(source.get("x", 0))),
            y=int(round(source.get("y", 0))),
            width=max(1, int(round(source.get("width", 0)))),
            height=max(1, int(round(source.get("height", 0)))),
        )
        return rect

    def apply_patch(self, base_image: torch.Tensor, fragment: torch.Tensor, crop_info: Any, resize_fragment: bool):
        base = self._normalize_tensor(base_image).clone()
        frag = self._normalize_tensor(fragment)

        base, frag = self._ensure_same_batch(base, frag)

        if base.shape[-1] != frag.shape[-1]:
            raise ValueError("Base image and fragment must have the same number of channels.")

        rect = self._parse_crop_info(crop_info)
        _, base_h, base_w, _ = base.shape

        rect = rect.clamp(base_w, base_h)

        target_h = rect.height
        target_w = rect.width

        if resize_fragment:
            if frag.shape[1] != target_h or frag.shape[2] != target_w:
                frag_nchw = frag.permute(0, 3, 1, 2)
                frag_resized = F.interpolate(frag_nchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
                frag = frag_resized.permute(0, 2, 3, 1)
        else:
            target_h = min(target_h, frag.shape[1])
            target_w = min(target_w, frag.shape[2])

        y_end = min(base_h, rect.y + target_h)
        x_end = min(base_w, rect.x + target_w)

        slice_h = y_end - rect.y
        slice_w = x_end - rect.x

        if slice_h <= 0 or slice_w <= 0:
            raise ValueError("Crop rectangle is outside the base image bounds.")

        base[:, rect.y:y_end, rect.x:x_end, :] = frag[:, :slice_h, :slice_w, :]
        return (base,)

