import torch
import numpy as np
import cv2
from PIL import Image, ImageEnhance

class ImageAdjustmentNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Параметр температуры позволяет смещать баланс между красным и синим.
                # Теперь поддерживаются как положительные (согрев), так и отрицательные (охлаждение) значения.
                "temperature": ("FLOAT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 1
                }),
                "hue": ("FLOAT", {
                    "default": 0,
                    "min": -180,
                    "max": 180,
                    "step": 1
                }),
                "brightness": ("FLOAT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 1
                }),
                "contrast": ("FLOAT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 1
                }),
                "saturation": ("FLOAT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 1
                }),
                "gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1
                }),
                # Дополнительные параметры для цветового баланса средних тонов
                "midtone_red": ("FLOAT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 1
                }),
                "midtone_green": ("FLOAT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 1
                }),
                "midtone_blue": ("FLOAT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_correct"
    CATEGORY = "😎 SnJake/Adjustment"

    def color_correct(self, image: torch.Tensor, temperature: float, hue: float, brightness: float, contrast: float,
                      saturation: float, gamma: float, midtone_red: float, midtone_green: float, midtone_blue: float):
        # Проверяем, что изображение имеет форму [B, H, W, 3]
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError("Входное изображение должно иметь форму [B, H, W, 3] (RGB)")

        original_device = image.device
        image = image.cpu().float()  # Работаем на CPU

        batch_size, height, width, channels = image.shape
        result = torch.zeros_like(image)

        # Приводим параметры к удобным коэффициентам
        brightness_factor = (brightness / 100.0) + 1.0  # [0, 2]
        contrast_factor = (contrast / 100.0) + 1.0        # [0, 2]
        saturation_factor = (saturation / 100.0) + 1.0      # [0, 2]
        hue_shift = hue  # в градусах
        gamma = max(gamma, 1e-8)  # избегаем 0 или отрицательных

        # Цветовой баланс для средних тонов (переводим [-100,100] в [-1,1])
        midtone_red   = midtone_red   / 100.0
        midtone_green = midtone_green / 100.0
        midtone_blue  = midtone_blue  / 100.0

        for b in range(batch_size):
            # Переводим тензор в numpy-массив и масштабируем в диапазон [0, 255]
            tensor_image = image[b].numpy()
            tensor_image = np.clip(tensor_image, 0, 1)
            tensor_image = (tensor_image * 255).astype(np.uint8)

            # Преобразуем в PIL Image для коррекции яркости и контраста
            pil_image = Image.fromarray(tensor_image)

            # Коррекция яркости
            brightness_factor_clipped = max(brightness_factor, 0.0)
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(brightness_factor_clipped)

            # Коррекция контраста
            contrast_factor_clipped = max(contrast_factor, 0.0)
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(contrast_factor_clipped)

            # Возвращаем в numpy-массив в диапазоне [0, 1]
            modified_image = np.array(pil_image).astype(np.float32) / 255.0

            # --- Улучшенная коррекция температуры ---
            # Новая логика: применяем мультипликативный коэффициент для каналов R и B.
            # Для положительной температуры (теплее) увеличиваем R и уменьшаем B,
            # для отрицательной (холоднее) – наоборот.
            if temperature != 0:
                # Выбираем масштаб. Делим на 200, чтобы при максимуме ±100 коэффициенты были не слишком экстремальными.
                t = temperature / 200.0  
                # Применяем корректировку: умножаем красный и синий каналы соответственно.
                r, g, b_channel = cv2.split(modified_image)
                r = np.clip(r * (1 + t), 0, 1)
                b_channel = np.clip(b_channel * (1 - t), 0, 1)
                modified_image = cv2.merge((r, g, b_channel))
            # -----------------------------------------

            # Гамма-коррекция (применяем степень gamma)
            if abs(gamma - 1.0) > 1e-3:
                modified_image = np.power(modified_image, gamma)
                modified_image = np.clip(modified_image, 0, 1)

            # Коррекция насыщенности и оттенка в пространстве HSV
            hsv_img = cv2.cvtColor(modified_image, cv2.COLOR_RGB2HSV)
            # Насыщенность
            if saturation != 0:
                hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] * saturation_factor, 0, 1)
            # Оттенок (hue). Поскольку hue хранится в [0,1], смещаем на hue/360
            if hue != 0:
                hsv_img[:, :, 0] = (hsv_img[:, :, 0] + (hue_shift / 360.0)) % 1.0
            modified_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
            modified_image = np.clip(modified_image, 0, 1)

            # Коррекция цветового баланса для средних тонов
            if any([midtone_red, midtone_green, midtone_blue]):
                # Вычисляем яркость (luminance) и создаем маску для средних тонов
                luminance = cv2.cvtColor(modified_image, cv2.COLOR_RGB2GRAY)
                midtone_mask = self.create_midtone_mask(luminance)
                midtone_mask = np.stack([midtone_mask] * 3, axis=2)
                # Формируем сдвиг по каждому каналу
                color_shift = np.zeros_like(modified_image)
                color_shift[:, :, 0] = midtone_red
                color_shift[:, :, 1] = midtone_green
                color_shift[:, :, 2] = midtone_blue
                modified_image = np.clip(modified_image + color_shift * midtone_mask, 0, 1)

            # Переводим результат обратно в тензор
            result[b] = torch.from_numpy(modified_image).to(torch.float32)

        # Возвращаем результат на исходном устройстве
        result = result.to(original_device)
        return (result,)

    def create_midtone_mask(self, luminance: np.ndarray) -> np.ndarray:
        """
        Создает маску для средних тонов на основе яркости.
        Используется гауссова функция, центрированная на 0.5.
        """
        luminance = np.clip(luminance, 0, 1)
        midtone_mask = np.exp(-4 * ((luminance - 0.5) ** 2))
        return midtone_mask
