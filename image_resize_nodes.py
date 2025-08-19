import torch
import comfy.utils

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
                # 2. Список методов, поддерживаемых comfy.utils.common_upscale
                "upscale_method": (["lanczos", "bicubic", "bilinear", "nearest-exact", "area"], {
                    "default": "lanczos"
                }),
            }
        }

    def resize_if_larger(self, image, target_resolution, keep_aspect_ratio, upscale_method):
        # Получаем размеры батча изображений. image.shape: [B, H, W, C]
        _batch, height, width, _channels = image.shape

        # --- УСЛОВИЕ ПРОВЕРКИ ---
        if height <= target_resolution and width <= target_resolution:
            print(f"SnJake Resize: Image is {width}x{height}, which is within the {target_resolution}px limit. Skipping.")
            return (image,)

        print(f"SnJake Resize: Resizing image from {width}x{height} to target ~{target_resolution}px using {upscale_method}")

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

        # Функции из comfy.utils ожидают формат [B, C, H, W]
        img_bchw = image.permute(0, 3, 1, 2)
        
        # 3. Используем универсальную функцию из ComfyUI
        # Она сама вызовет нужный метод интерполяции
        resized_img = comfy.utils.common_upscale(
            img_bchw, 
            new_width, 
            new_height, 
            upscale_method, 
            "disabled"  # Параметр crop нам не нужен, т.к. мы сами рассчитали размеры
        )
        
        # Возвращаем формат обратно к [B, H, W, C]
        resized_img_bhwc = resized_img.permute(0, 2, 3, 1)

        return (resized_img_bhwc,)
