import torch
import numpy as np

class CRTEffectNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "scanlines_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scanlines_interval": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "pixel_structure_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "crt_effect"
    CATEGORY = "😎 SnJake/Effects"

    def crt_effect(self, image: torch.Tensor, scanlines_strength: float = 0.5, scanlines_interval: int = 2, pixel_structure_strength: float = 0.8):
        batch_size, height, width, channels = image.shape
        output_image = torch.zeros_like(image)

        for b in range(batch_size):
            img = image[b].clone()

            # Apply CRT pixel structure
            crt_pixels = torch.zeros_like(img)
            for y in range(height):
                for x in range(width):
                    if channels == 3:
                        if x % 3 == 0:
                            crt_pixels[y, x, 0] = img[y, x, 0]
                        elif x % 3 == 1:
                            crt_pixels[y, x, 1] = img[y, x, 1]
                        elif x % 3 == 2:
                            crt_pixels[y, x, 2] = img[y, x, 2]
                    elif channels == 4: # Assuming RGBA
                        if x % 3 == 0:
                            crt_pixels[y, x, 0] = img[y, x, 0]
                            crt_pixels[y, x, 3] = img[y, x, 3] # Keep alpha
                        elif x % 3 == 1:
                            crt_pixels[y, x, 1] = img[y, x, 1]
                            crt_pixels[y, x, 3] = img[y, x, 3] # Keep alpha
                        elif x % 3 == 2:
                            crt_pixels[y, x, 2] = img[y, x, 2]
                            crt_pixels[y, x, 3] = img[y, x, 3] # Keep alpha

            if pixel_structure_strength < 1.0:
                output_image[b] = (pixel_structure_strength * crt_pixels) + ((1 - pixel_structure_strength) * img)
            else:
                output_image[b] = crt_pixels

            # Apply scanlines
            if scanlines_strength > 0:
                scanlines = torch.ones((height, width, 1), dtype=torch.float32)
                for y in range(0, height, scanlines_interval):
                    if y + 1 < height:
                        scanlines[y, :, :] *= (1.0 - scanlines_strength)

                if channels == 3:
                    scanlines = scanlines.repeat(1, 1, 3)
                elif channels == 4:
                    scanlines = scanlines.repeat(1, 1, 4)

                output_image[b] *= scanlines

        return (output_image,)