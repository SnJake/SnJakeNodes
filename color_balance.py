import torch
from typing import List
import numpy as np

class ColorBalance:
    """
    Узел Color Balance (Lift / Gamma / Gain)
    (аналог узла Color Balance из Blender).

    Принимает изображение тензора [B, H, W, 3] (RGB) и по каналам параметры:
      - Lift: смещение (обычно диапазон около [0, 2]),
      - Gamma: коррекция гаммой (значения > 0; используем обратное значение в степени),
      - Gain: масштабирование (коэффициент усиления).
    Также имеется параметр fac для линейного смешивания с исходным изображением.

    Формула: 
      out = (image * gain)^(1/gamma) + lift - 1
      final = fac * out + (1 - fac) * image
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Lift для каждого канала (поддерживается диапазон [0,2])
                "lift_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "lift_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "lift_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                # Gamma для каждого канала (значения > 0)
                "gamma_r": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.01}),
                "gamma_g": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.01}),
                "gamma_b": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.01}),
                # Gain для каждого канала
                "gain_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "gain_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "gain_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                # Фактор смешивания (от 0 до 1)
                "fac": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_balance"
    CATEGORY = "😎 SnJake/Adjustment"

    def color_balance(self, image: torch.Tensor,
                      lift_r: float, lift_g: float, lift_b: float,
                      gamma_r: float, gamma_g: float, gamma_b: float,
                      gain_r: float, gain_g: float, gain_b: float,
                      fac: float) -> List[torch.Tensor]:
        # Проверяем, что изображение имеет форму [B, H, W, 3]
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError("Входное изображение должно иметь форму [B, H, W, 3] (RGB)")

        # Создаем тензоры параметров для каждого эффекта
        lift = torch.tensor([lift_r, lift_g, lift_b], dtype=image.dtype, device=image.device).view(1, 1, 1, 3)
        gamma = torch.tensor([gamma_r, gamma_g, gamma_b], dtype=image.dtype, device=image.device).view(1, 1, 1, 3)
        gain = torch.tensor([gain_r, gain_g, gain_b], dtype=image.dtype, device=image.device).view(1, 1, 1, 3)

        # Избегаем деления на ноль – гамма не может быть равной 0
        gamma = torch.clamp(gamma, min=1e-8)

        # Применяем формулу: out = (image * gain)^(1/gamma) + lift - 1
        out = torch.pow(image * gain, 1.0 / gamma)
        out = out + lift - 1.0

        # Смешиваем с исходным изображением по фактору fac
        fac_tensor = torch.tensor(fac, dtype=image.dtype, device=image.device).view(1, 1, 1, 1)
        final = fac_tensor * out + (1 - fac_tensor) * image

        # Ограничиваем результат диапазоном [0, 1]
        final = torch.clamp(final, 0.0, 1.0)
        return (final,)
