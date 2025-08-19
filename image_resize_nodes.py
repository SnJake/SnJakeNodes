import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

class SnJakeResizeIfLarger:
    """
    Эта нода изменяет размер изображения, только если оно больше указанного
    целевого разрешения по одной из сторон.
    """
    
    FUNCTION = "resize_if_larger"
    CATEGORY = "😎 SnJake/Utils"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("resized_image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_resolution": ("INT", {
                    "default": 1024, 
                    "min": 64, 
                    "max": 8192,
                    "step": 8,
                }),
                "keep_aspect_ratio": ("BOOLEAN", {
                    "default": True, 
                    "label_on": "enabled", 
                    "label_off": "disabled"
                }),
                "interpolation_method": (["lanczos", "bicubic", "bilinear", "nearest"], {
                    "default": "lanczos"
                }),
            }
        }

    def resize_if_larger(self, image, target_resolution, keep_aspect_ratio, interpolation_method):
        # Получаем размеры батча изображений. image.shape: [B, H, W, C]
        _batch, height, width, _channels = image.shape

        # --- УСЛОВИЕ ПРОВЕРКИ ---
        if height <= target_resolution and width <= target_resolution:
            print(f"SnJake Resize: Image is {width}x{height}, which is within the {target_resolution}px limit. Skipping.")
            return (image,)

        print(f"SnJake Resize: Resizing image from {width}x{height} to target ~{target_resolution}px using {interpolation_method}")

        if keep_aspect_ratio:
            if width > height:
                scale_factor = target_resolution / width
            else:
                scale_factor = target_resolution / height
            
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
        else:
            new_width = target_resolution
            new_height = target_resolution

        # Для функции resize нам нужно изменить порядок измерений на [B, C, H, W]
        img_bchw = image.permute(0, 3, 1, 2)

        # Словарь для сопоставления строковых имен методов с объектами InterpolationMode
        interpolation_map = {
            "lanczos": InterpolationMode.LANCZOS,
            "bicubic": InterpolationMode.BICUBIC,
            "bilinear": InterpolationMode.BILINEAR,
            "nearest": InterpolationMode.NEAREST,
        }
        
        # Выполняем изменение размера с использованием выбранного метода
        resized_img = F.resize(
            img_bchw, 
            size=[new_height, new_width], 
            interpolation=interpolation_map.get(interpolation_method, InterpolationMode.LANCZOS),
            antialias=True  # Рекомендуется для лучшего качества
        )
        
        # Возвращаем порядок измерений обратно к [B, H, W, C]
        resized_img_bhwc = resized_img.permute(0, 2, 3, 1)

        return (resized_img_bhwc,)
