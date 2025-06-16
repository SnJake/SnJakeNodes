import torch

class ExpandImageRight:
    """
    Узел для расширения изображения вправо.
    
    Если входное изображение имеет форму [B, H, W, C], на выходе получается изображение [B, H, 2W, C],
    где левая половина — исходное изображение, а правая заполнена белым (значение 1).
    
    Одновременно создаётся маска того же пространственного размера, где:
      - слева (исходное изображение) — чёрная (0),
      - справа (новая область) — белая (1).
      
    Маска приводится к форме [B, 1, H, 2W] (для типа MASK).
    """

    CATEGORY = "Custom/Images"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_in": ("IMAGE", {"tooltip": "Входное изображение [B,H,W,C]"})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image_out", "mask_out")
    FUNCTION = "expand_image"

    def expand_image(self, image_in):
        # Извлекаем размеры входного изображения: [B, H, W, C]
        B, H, W, C = image_in.shape

        # Создаём белую область того же размера, что и исходное изображение
        white_area = torch.ones((B, H, W, C), dtype=image_in.dtype, device=image_in.device)

        # Конкатенируем исходное изображение и белую область по оси ширины (axis=2)
        image_out = torch.cat([image_in, white_area], dim=2)  # Результат: [B, H, 2W, C]

        # Генерируем маску:
        # Для левой части (исходное изображение) — нули (чёрная)
        # Для правой части (добавленная область) — единицы (белая)
        mask_left = torch.zeros((B, H, W), dtype=image_in.dtype, device=image_in.device)
        mask_right = torch.ones((B, H, W), dtype=image_in.dtype, device=image_in.device)
        mask = torch.cat([mask_left, mask_right], dim=2)  # Получается [B, H, 2W]

        # Приводим маску к форме [B, 1, H, 2W] (соответствует [B,C,H,W], где C=1)
        mask_out = mask.unsqueeze(1)

        return (image_out, mask_out)
