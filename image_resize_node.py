try:
    from .utils_snjake.bislerp_standalone import bislerp as standalone_bislerp
except ImportError:
    print("Warning: Could not import bislerp_standalone. Ensure it's in the correct path.")
    standalone_bislerp = None


import nodes
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
# from scipy.ndimage import zoom # Больше не нужен для bislerp
# from scipy.special import sinc # Не используется
# import torchvision.transforms.functional as TF # Не используется

class ImageResizeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["Rescale", "Resize"], {"default": "Rescale"}),
                "supersampling": ("BOOLEAN", {"default": False}),
                "resampling": (["nearest", "bilinear", "bicubic", "bislerp", "lanczos", "area", "nearest-exact"], {"default": "bislerp"}),
                "rescale_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "resize_width": ("INT", {"default": 512, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "resize_height": ("INT", {"default": 512, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_image"
    CATEGORY = "😎 SnJake/Adjustment"

    def resize_image(self, image, mode, supersampling, resampling, rescale_factor, resize_width, resize_height):
        if resampling == "bislerp" and standalone_bislerp is None:
            raise ImportError("Bislerp standalone function not imported. Cannot use 'bislerp' resampling.")

        # В ComfyUI image имеет форму (Batch, Height, Width, Channels) и значения [0,1]
        # Ваш standalone_bislerp ожидает (N, C, H, W)
        image_tensor_nchw = image.permute(0, 3, 1, 2) # B,H,W,C -> B,C,H,W

        b, c, h, w = image_tensor_nchw.shape

        if mode == "Rescale":
            new_width = w * rescale_factor
            new_height = h * rescale_factor
        else:  # Режим Resize
            new_width = resize_width
            new_height = resize_height

        new_width = max(1, int(round(new_width)))
        new_height = max(1, int(round(new_height)))

        print(f"Original image tensor (NCHW) shape: {image_tensor_nchw.shape}")
        print(f"Target dimensions (H, W): ({new_height}, {new_width})")

        if resampling == "bislerp":
            if supersampling:
                supersample_scale = 2
                temp_width = max(1, int(round(new_width * supersample_scale)))
                temp_height = max(1, int(round(new_height * supersample_scale)))

                print(f"Bislerp supersampling to: ({temp_height}, {temp_width})")
                temp_resized_nchw = standalone_bislerp(image_tensor_nchw, width=temp_width, height=temp_height)
                print(f"Bislerp downscaling to: ({new_height}, {new_width})")
                final_resized_nchw = standalone_bislerp(temp_resized_nchw, width=new_width, height=new_height)
            else:
                print(f"Bislerp resizing to: ({new_height}, {new_width})")
                final_resized_nchw = standalone_bislerp(image_tensor_nchw, width=new_width, height=new_height)

            # Преобразуем обратно в (B, H, W, C)
            output_tensor = final_resized_nchw.permute(0, 2, 3, 1) # B,C,H,W -> B,H,W,C

        else: # Другие методы ресемплинга (используем PIL, как и раньше)
            # Преобразуем тензор Torch в изображение PIL
            # image_tensor уже в формате B,H,W,C
            image_np = image[0].cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)

            if image_np.shape[2] == 1: # Монохромное
                image_pil = Image.fromarray(image_np.squeeze(2), mode='L').convert('RGB')
            else:
                image_pil = Image.fromarray(image_np)

            # Определяем PIL методы ресемплинга
            resample_methods_pil = {
                "nearest": Image.NEAREST,
                "bilinear": Image.BILINEAR,
                "bicubic": Image.BICUBIC,
                "lanczos": Image.LANCZOS,
                "area": Image.BOX, # Image.BOX часто используется для 'area'
                "nearest-exact": Image.NEAREST, # PIL NEAREST и есть nearest-exact
            }
            resample_method_pil = resample_methods_pil.get(resampling, Image.BILINEAR)


            if supersampling:
                supersample_scale = 2
                temp_width_pil = max(1, int(round(new_width * supersample_scale)))
                temp_height_pil = max(1, int(round(new_height * supersample_scale)))

                image_temp_pil = image_pil.resize((temp_width_pil, temp_height_pil), resample=resample_method_pil)
                image_resized_pil = image_temp_pil.resize((new_width, new_height), resample=resample_method_pil)
            else:
                image_resized_pil = image_pil.resize((new_width, new_height), resample=resample_method_pil)

            # Преобразуем обратно в тензор Torch
            image_resized_np = np.array(image_resized_pil).astype(np.float32) / 255.0
            if image_resized_np.ndim == 2:
                image_resized_np = np.expand_dims(image_resized_np, axis=2)
            if image_resized_np.shape[2] == 1:
                image_resized_np = np.repeat(image_resized_np, 3, axis=2)

            output_tensor = torch.from_numpy(image_resized_np).unsqueeze(0).to(image.device)

        print(f"Resized image tensor (BHWC) shape: {output_tensor.shape}")

        return (output_tensor,)
