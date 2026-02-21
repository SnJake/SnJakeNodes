# file: ultralytics.py

from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

import folder_paths
from server import PromptServer

# ──────────────────────────── helpers ──────────────────────────────
_ALLOWED_EXTS = {".pt", ".safetensors", ".onnx"}

def _root() -> Path:
    return Path(folder_paths.base_path) / "models" / "ultralytics"

def _scan(kind: str) -> List[str]:
    d = _root() / kind
    if not d.exists():
        return []
    return sorted(
        str(p.relative_to(_root()).as_posix())
        for p in d.rglob("*")
        if p.suffix.lower() in _ALLOWED_EXTS and p.is_file()
    )

def _resolve(r: str) -> str:
    return str((_root() / r).resolve())

# ────────────────────── ComfyUI custom types ──────────────────────
class _Any(str):
    def __eq__(self, other):
        return True
    __ne__ = lambda self, other: False

YOLO_MODEL = _Any("YOLO_MODEL")
BBOX_TYPE  = _Any("BBOX")

# ───────────────────────── loader node ────────────────────────────
class YoloModelLoader:
    CATEGORY = "😎 SnJake/YOLO"
    FUNCTION = "load"
    RETURN_TYPES = (YOLO_MODEL, "STRING")
    RETURN_NAMES = ("model", "model_path")

    @classmethod
    def INPUT_TYPES(cls):
        opts = _scan("bbox") + _scan("segm") or ["<no models>"]
        return {
            "required": {
                "task": (["bbox", "segm"], {"default": "bbox"}),
                "model_name": (opts, {"default": opts[0]}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            }
        }

    def load(self, task: str, model_name: str, device: str = "auto"):
        if model_name.startswith("<no models"):
            raise FileNotFoundError("Папка models/ultralytics пуста")
        if not model_name.startswith(f"{task}/"):
            PromptServer.instance.send_sync("yolo_loader.warn", {"msg": f"'{model_name}' не в подпапке '{task}/' – проверьте, та ли это модель."})
        model = YOLO(_resolve(model_name), task="detect" if task == "bbox" else "segment")
        if device != "auto":
            model.to(device)
        return (model, model_name)

# ─────────────────────── inference node ───────────────────────────
class YoloInference:
    CATEGORY = "😎 SnJake/YOLO"
    FUNCTION = "infer"
    RETURN_TYPES = ("IMAGE", "MASK", BBOX_TYPE, "SAM3_BOXES_PROMPT")
    RETURN_NAMES = ("image_with_bboxes", "mask", "bboxes", "sam3_positive_boxes")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "model": (YOLO_MODEL, {"forceInput": True}),
                "conf": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}), # Повысил стандартное значение
                "iou": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}), # Повысил стандартное значение
                "filter_classes": ("STRING", {"default": "", "placeholder": "comma‑separated names"}),
                "mask_bbox_fuzz": ("INT", {"default": 0, "min": 0, "max": 32}), # Убрал стандартное размытие
            }
        }

    @staticmethod
    def _tensor_to_uint8(img: torch.Tensor):
        return (img * 255).clamp(0, 255).byte().cpu().numpy()

    @staticmethod
    def _boxes_to_mask(h: int, w: int, boxes: List[Dict], fuzz: int = 0) -> torch.Tensor:
        mask = torch.zeros((h, w), dtype=torch.bool)
        for b in boxes:
            x1, y1, x2, y2 = map(int, b["xyxy"])
            x1 = max(0, x1 - fuzz)
            y1 = max(0, y1 - fuzz)
            x2 = min(w, x2 + fuzz)
            y2 = min(h, y2 + fuzz)
            mask[y1:y2, x1:x2] = True
        return mask

    @staticmethod
    def _draw_boxes(img_tensor, boxes):
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default(size=20)

        for b in boxes:
            xyxy = b["xyxy"]
            label = f"{b['class_name']} {b['confidence']:.2f}"
            draw.rectangle(xyxy, outline="red", width=3)
            
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            
            text_y = xyxy[1] - text_h - 5
            if text_y < 0:
                text_y = xyxy[1] + 2

            draw.rectangle([xyxy[0], text_y, xyxy[0] + text_w + 4, text_y + text_h + 4], fill="red")
            draw.text((xyxy[0] + 2, text_y + 2), label, fill="white", font=font)
            
        return torch.from_numpy(np.array(img_pil).astype(np.float32) / 255.0)

    @staticmethod
    def _to_sam3_boxes_prompt(boxes: List[Dict], image_h: int, image_w: int):
        """
        Convert YOLO xyxy pixel boxes into SAM3_BOXES_PROMPT:
        {
            "boxes": [[center_x, center_y, width, height], ...],  # normalized 0..1
            "labels": [True, ...]  # positive prompts
        }
        """
        if image_h <= 0 or image_w <= 0:
            return {"boxes": [], "labels": []}

        sam3_boxes = []
        sam3_labels = []

        for b in boxes:
            x1, y1, x2, y2 = b["xyxy"]

            # Clamp to image bounds and normalize.
            x1 = max(0.0, min(float(x1), float(image_w)))
            y1 = max(0.0, min(float(y1), float(image_h)))
            x2 = max(0.0, min(float(x2), float(image_w)))
            y2 = max(0.0, min(float(y2), float(image_h)))

            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            x1_norm = x1 / float(image_w)
            y1_norm = y1 / float(image_h)
            x2_norm = x2 / float(image_w)
            y2_norm = y2 / float(image_h)

            center_x = (x1_norm + x2_norm) * 0.5
            center_y = (y1_norm + y2_norm) * 0.5
            width = max(0.0, x2_norm - x1_norm)
            height = max(0.0, y2_norm - y1_norm)

            sam3_boxes.append([center_x, center_y, width, height])
            sam3_labels.append(True)

        return {"boxes": sam3_boxes, "labels": sam3_labels}

    def infer(self, image, model, conf: float, iou: float, filter_classes: str, mask_bbox_fuzz: int):
        if not hasattr(model, "predict"):
            raise TypeError("Входные данные не являются моделью Ultralytics YOLO")

        allowed = {s.strip() for s in filter_classes.split(",") if s.strip()}
        imgs_np = [self._tensor_to_uint8(im) for im in image]
        
        results = model.predict(imgs_np, conf=conf, iou=iou, stream=False, verbose=False)

        batch_boxes: List[List[Dict]] = []
        batch_mask: List[torch.Tensor] = []
        drawn_images: List[torch.Tensor] = []

        for i, res in enumerate(results):
            boxes_cur = []
            keep_idx = []
            if res.boxes is not None:
                for idx, (xyxy, conf_, cls_) in enumerate(zip(res.boxes.xyxy.cpu(), res.boxes.conf.cpu(), res.boxes.cls.cpu())):
                    cls_name = model.names[int(cls_)]
                    if allowed and cls_name not in allowed:
                        continue
                    keep_idx.append(idx)
                    boxes_cur.append({
                        "xyxy": xyxy.tolist(),
                        "confidence": float(conf_),
                        "class_id": int(cls_),
                        "class_name": cls_name,
                    })
            batch_boxes.append(boxes_cur)
            
            drawn_images.append(self._draw_boxes(image[i], boxes_cur))

            if getattr(res, "masks", None) is not None and keep_idx:
                m = torch.any(res.masks.data[torch.tensor(keep_idx, device=res.masks.data.device)].float() > 0.5, dim=0).cpu()
            else:
                h, w = res.orig_shape
                m = self._boxes_to_mask(h, w, boxes_cur, mask_bbox_fuzz)
            batch_mask.append(m)

        images_out = torch.stack(drawn_images, dim=0)
        
        if not batch_mask: # Если масок нет, возвращаем пустой тензор
             masks_out = torch.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=torch.float32)
        else:
             masks_out = torch.stack(batch_mask, dim=0).float()

        sam3_positive_boxes = {"boxes": [], "labels": []}
        if len(batch_boxes) > 0:
            # SAM3Grounding consumes one IMAGE; for batched YOLO output expose boxes from the first image.
            image_h = int(image.shape[1])
            image_w = int(image.shape[2])
            sam3_positive_boxes = self._to_sam3_boxes_prompt(batch_boxes[0], image_h, image_w)

            if len(batch_boxes) > 1:
                print("[YoloInference] Batch size > 1: 'sam3_positive_boxes' is generated from the first image only.")

        return (images_out, masks_out, batch_boxes, sam3_positive_boxes)

