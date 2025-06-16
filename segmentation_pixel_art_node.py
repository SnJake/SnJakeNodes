import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from skimage.segmentation import slic
from skimage.color import label2rgb

class SegmentationPixelArtNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "cell_size": ("INT", {"default":4, "min":1, "max":64, "step":1, "display":"slider"}),
                "n_segments": ("INT", {"default":200, "min":10, "max":2000, "step":10}),
                "compactness": ("FLOAT", {"default":10.0, "min":0.1, "max":100.0, "step":0.1})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "😎 SnJake/PixelArt"

    def process(self, image: torch.Tensor, cell_size: int, n_segments: int, compactness: float):
        # image: [B,H,W,C], typically B=1
        # Преобразуем тензор в PIL для удобной обработки
        # Тензор в диапазоне [0,1]
        # Формат: [B, H, W, C], C=3
        if image.shape[0] != 1:
            raise ValueError("Only a single image in the batch is supported.")

        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)  # [H,W,C]
        pil_img = Image.fromarray(img_np)

        # Исходные размеры
        w, h = pil_img.size

        # Масштабируем изображение вниз
        # Чем больше cell_size, тем меньше итоговое разрешение, следовательно крупнее "пиксели".
        new_w = max(1, w // cell_size)
        new_h = max(1, h // cell_size)
        small_img = pil_img.resize((new_w, new_h), Image.BICUBIC)  # можно попробовать BILINEAR/BICUBIC

        # Преобразуем в numpy для сегментации
        small_np = np.array(small_img)  # shape [H',W',3]

        # Применяем SLIC сегментацию
        # n_segments - количество суперпикселей, compactness - баланс цвет/пространство
        labels = slic(small_np, n_segments=n_segments, compactness=compactness, start_label=0)

        # label2rgb заполнит каждый сегмент средним цветом сегмента
        # Важно: label2rgb по умолчанию смешивает с оригиналом. Укажем bg_label и alpha=1
        quantized_img = label2rgb(labels, small_np, kind='avg', bg_label=-1, alpha=1)

        # quantized_img сейчас в float64 [0,1], преобразуем обратно в uint8
        quantized_img = (quantized_img * 255).astype(np.uint8)

        # Масштабируем обратно до исходного размера через nearest neighbor (чтобы получить пикселизацию)
        final_img = Image.fromarray(quantized_img).resize((w, h), Image.NEAREST)

        # Преобразуем обратно в тензор формата [B,H,W,C] с диапазоном [0,1]
        final_np = np.array(final_img).astype(np.float32) / 255.0
        final_tensor = torch.from_numpy(final_np)[None,]  # [1,H,W,C]

        return (final_tensor,)

# Регистрируем узел
NODE_CLASS_MAPPINGS = {
    "SegmentationPixelArtNode": SegmentationPixelArtNode
}
