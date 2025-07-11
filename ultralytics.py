from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
from ultralytics import YOLO  # pip install ultralytics>=0.4.8

# ComfyUI
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
    def __eq__(self, other):  # type: ignore
        return True
    __ne__ = lambda self, other: False  # type: ignore

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

    # ------------------------------------------------------------------
    def load(self, task: str, model_name: str, device: str = "auto"):
        if model_name.startswith("<no models"):
            raise FileNotFoundError("models/ultralytics пусто")
        if not model_name.startswith(f"{task}/"):
            PromptServer.instance.send_sync(
                "yolo_loader.warn",
                {"msg": f"'{model_name}' не в подпапке '{task}/' – проверьте, та ли это модель."},
            )
        model = YOLO(_resolve(model_name), task="detect" if task == "bbox" else "segment")
        if device != "auto":
            model.to(device)
        return (model, model_name)

# ─────────────────────── inference node ───────────────────────────
class YoloInference:
    CATEGORY = "😎 SnJake/YOLO"
    FUNCTION = "infer"

    RETURN_TYPES = (BBOX_TYPE, "MASK")
    RETURN_NAMES = ("bboxes", "mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "model": (YOLO_MODEL, {"forceInput": True}),
                "conf": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 1.0, "step": 0.05}),
                "iou": ("FLOAT", {"default": 0.45, "min": 0.1, "max": 1.0, "step": 0.05}),
                "filter_classes": ("STRING", {"default": "", "placeholder": "comma‑separated names"}),
                "mask_bbox_fuzz": ("INT", {"default": 2, "min": 0, "max": 32}),
            }
        }

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def infer(self, image, model, conf: float, iou: float, filter_classes: str, mask_bbox_fuzz: int):
        if not hasattr(model, "predict"):
            raise TypeError("Input is not a Ultralytics YOLO model")

        allowed = {s.strip() for s in filter_classes.split(",") if s.strip()}
        imgs_np = [self._tensor_to_uint8(im) for im in image]  # split batch
        results = model.predict(imgs_np, conf=conf, iou=iou, imgsz=max(image.shape[1:-1]), stream=False, verbose=False)

        batch_boxes: List[List[Dict]] = []
        batch_mask: List[torch.Tensor] = []

        for res in results:
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

            # choose mask source
            if getattr(res, "masks", None) is not None and keep_idx:
                m = torch.any(res.masks.data[keep_idx].float() > 0.5, dim=0).cpu()
            else:
                h, w = res.orig_shape
                m = self._boxes_to_mask(h, w, boxes_cur, mask_bbox_fuzz)
            batch_mask.append(m)

        # stack to one tensor [B,H,W]
        H = max(m.shape[0] for m in batch_mask)
        W = max(m.shape[1] for m in batch_mask)
        masks_out = torch.zeros((len(batch_mask), H, W), dtype=torch.bool)
        for i, m in enumerate(batch_mask):
            masks_out[i, : m.shape[0], : m.shape[1]] = m

        return (batch_boxes, masks_out)