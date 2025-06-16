import torch
import numpy as np
from PIL import ImageFilter, Image 

class CustomMaskNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_width": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "mask_height": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "position_x": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "position_y": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "canvas_width": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
                "canvas_height": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
            }
        }
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    CATEGORY = "😎 SnJake/Masks"
    FUNCTION = "create_custom_mask"

    def create_custom_mask(self, mask_width, mask_height, position_x, position_y, blur_radius, canvas_width, canvas_height):
        # 1. Создание маски
        mask = torch.ones((mask_height, mask_width), dtype=torch.float32)

        # 2. Создание черного полотна
        canvas = torch.zeros((canvas_height, canvas_width), dtype=torch.float32)

        # 3. Определение положения маски на полотне
        start_x = position_x
        start_y = position_y
        end_x = position_x + mask_width
        end_y = position_y + mask_height

        canvas_start_x = max(0, start_x)
        canvas_start_y = max(0, start_y)
        canvas_end_x = min(canvas_width, end_x)
        canvas_end_y = min(canvas_height, end_y)

        mask_start_x = max(0, -start_x)
        mask_start_y = max(0, -start_y)
        mask_end_x = mask_width - max(0, end_x - canvas_width)
        mask_end_y = mask_height - max(0, end_y - canvas_height)

        # 4. Размещение маски на полотне
        if canvas_end_y > canvas_start_y and canvas_end_x > canvas_start_x and mask_end_y > mask_start_y and mask_end_x > mask_start_x:
            canvas[canvas_start_y:canvas_end_y, canvas_start_x:canvas_end_x] = mask[mask_start_y:mask_end_y, mask_start_x:mask_end_x]

        # 5. Блюр маски
        if blur_radius > 0:
            # Convert to PIL Image for blurring
            mask_pil = Image.fromarray((canvas.cpu().numpy() * 255).astype(np.uint8))
            blurred_mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            blurred_mask = torch.from_numpy(np.array(blurred_mask_pil).astype(np.float32) / 255.0)
            canvas = blurred_mask

        return (canvas,)