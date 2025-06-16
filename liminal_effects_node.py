import torch
import math

class LiminalEffectsNode:
    """
    Нода, совмещающая:
      1) Ч/б шум (noise_strength)
      2) Хроматическую аберрацию (chromatic_shift)
      3) Выцветание (fade_strength)
      4) Виньетку (vignette_strength)
      5) Туман (fog_strength, fog_color)
      6) VHS-эффект (vhs_strength)
      7) Случайные глитчи (glitch_strength)
    """

    CATEGORY = "😎 SnJake/Effects"
    FUNCTION = "apply_effects"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_in": ("IMAGE", {}),

                # старые эффекты:
                "noise_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Сила ч/б шума"
                }),
                "chromatic_shift": ("INT", {
                    "default": 2, "min": 0, "max": 20,
                    "tooltip": "Смещение каналов RGB (пиксели)"
                }),
                "fade_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Выцветание (0=нет, 1=серое)"
                }),

                # новые эффекты:
                "vignette_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Виньетка: 0=нет, 1=макс."
                }),
                "fog_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Туман: 0=нет, 1=полностью белое"
                }),
                "fog_color": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Цвет тумана (серый 0..1), 1=белое, 0=чёрное"
                }),
                "vhs_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "VHS-эффект (0=выкл, 1=макс)"
                }),
                "glitch_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Случайные глитчи (0=нет, 1=макс)"
                }),

                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 999999,
                    "tooltip": "Зерно шума и глитча (0 => не фиксировать)"
                }),
            }
        }

    def apply_effects(self,
                      image_in,
                      noise_strength,
                      chromatic_shift,
                      fade_strength,
                      vignette_strength,
                      fog_strength,
                      fog_color,
                      vhs_strength,
                      glitch_strength,
                      seed=0):
        """
        image_in:  [B, H, W, C]
        Все strength-параметры: float
        seed: int (если =0, seed не фиксируем)
        """

        # Приводим [B,H,W,C] => [B,C,H,W]
        out = image_in.permute(0, 3, 1, 2).clone()

        # Если надо зафиксировать поведение шумов/глитчей
        if seed != 0:
            torch.manual_seed(seed)

        # 1. Ч/б шум
        if noise_strength > 0.0:
            out = self.add_grayscale_noise(out, noise_strength)

        # 2. Хроматическая аберрация
        if chromatic_shift > 0:
            out = self.chromatic_aberration(out, chromatic_shift)

        # 3. Выцветание
        if fade_strength > 0.0:
            out = self.fade_image(out, fade_strength)

        # 4. Виньетка
        if vignette_strength > 0.0:
            out = self.add_vignette(out, vignette_strength)

        # 5. Туман
        if fog_strength > 0.0:
            out = self.add_fog(out, fog_strength, fog_color)

        # 6. VHS
        if vhs_strength > 0.0:
            out = self.add_vhs_effect(out, vhs_strength)

        # 7. Случайные глитчи
        if glitch_strength > 0.0:
            out = self.add_random_glitches(out, glitch_strength)

        # Клипим и возвращаем в [B,H,W,C]
        out = out.clamp(0.0, 1.0)
        out = out.permute(0, 2, 3, 1).contiguous()
        return (out,)

    # ------------------------------------------------------------------------
    # Ниже — вспомогательные методы

    def add_grayscale_noise(self, img, strength):
        """
        img: [B,C,H,W]
        Генерируем шум (1 канал) и добавляем к img
        """
        B, C, H, W = img.shape
        noise = torch.randn(B, 1, H, W, device=img.device, dtype=img.dtype)
        noise = noise * strength
        # дублируем в C каналов
        noise = noise.expand(-1, C, -1, -1)
        return img + noise

    def chromatic_aberration(self, img, shift):
        """
        Сдвигаем R/G/B по-разному
        img: [B,3,H,W]
        shift: int
        """
        B, C, H, W = img.shape
        # На всякий случай проверим, чтобы C>=3
        if C < 3:
            return img

        r = torch.roll(img[:, 0:1, :, :], shifts= shift,    dims=3)  # вправо
        g = torch.roll(img[:, 1:2, :, :], shifts= shift//2, dims=2)  # вниз
        b = torch.roll(img[:, 2:3, :, :], shifts=-shift,    dims=3)  # влево
        return torch.cat([r, g, b], dim=1)

    def fade_image(self, img, fade_strength):
        """
        Смесь исходника с его grayscale
        fade_strength=1 => полностью серое
        """
        gray = img.mean(dim=1, keepdim=True)  # [B,1,H,W]
        return img*(1.0 - fade_strength) + gray*fade_strength

    def add_vignette(self, img, vignette_strength):
        """
        Простейшая виньетка: умножаем пиксели ближе к краям на (1 - factor).
        factor ~ растёт от 0 в центре до vignette_strength по краю.
        """
        B, C, H, W = img.shape
        # координаты x,y в диапазоне [-1..1]
        yy = torch.linspace(-1, 1, steps=H, device=img.device)
        xx = torch.linspace(-1, 1, steps=W, device=img.device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')  # [H,W]

        # радиальное расстояние от центра (0,0) до (x,y)
        rr = torch.sqrt(grid_x*grid_x + grid_y*grid_y)  # [H,W], 0..sqrt(2)
        # нормируем к 1.0 = край
        # max_dist ~ sqrt(2) => делаем rr_n = rr / sqrt(2)
        rr_n = rr / math.sqrt(2)

        # factor = rr_n, но масштабируем на vignette_strength
        factor = rr_n * vignette_strength
        # clip to [0,1]
        factor = factor.clamp(0.0, 1.0)

        # factor сейчас [H,W]. Расширим до [B,1,H,W]
        factor = factor.unsqueeze(0).unsqueeze(1).expand(B, 1, H, W)

        # затем умножим: out = img * (1 - factor)
        out = img * (1.0 - factor)
        return out

    def add_fog(self, img, fog_strength, fog_color):
        """
        Линейно смешиваем изображение с неким цветом (fog_color).
        fog_color — grayscale [0..1]
        fog_strength — доля fog_color
        """
        B, C, H, W = img.shape
        # fog_color => [C], допустим C=3, но если C>3, просто возьмём первые каналы
        # Для простоты считаем, что C=3 или 1.  
        # Мы используем один float => делаем серый для каждого канала
        color_tensor = img.new_ones(C) * fog_color  # shape [C]
        # затем превращаем [C] => [1,C,1,1], чтобы broadcast'ить
        color_tensor = color_tensor.view(1, C, 1, 1)

        out = img*(1.0 - fog_strength) + color_tensor*(fog_strength)
        return out

    def add_vhs_effect(self, img, vhs_strength):
        """
        1) «Разбиваем» каждую строку случайным сдвигом.
        2) Добавляем «scan lines» (тёмные полосы).
        vhs_strength: 0..1
        """
        B, C, H, W = img.shape

        # Максимальный пиксельный сдвиг
        max_shift = int(5 * vhs_strength)

        # Генерируем для каждой строки случайный сдвиг
        # shape [H], в диапазоне [-max_shift..+max_shift]
        if max_shift > 0:
            shifts = torch.randint(low=-max_shift, high=max_shift+1, size=(H,), device=img.device)
        else:
            shifts = torch.zeros(H, device=img.device, dtype=torch.int)

        # Применяем строчный сдвиг. 
        # Простейший способ — циклом по строкам (не оптимально, но наглядно).
        out = img.clone()
        for row_idx in range(H):
            shift_val = int(shifts[row_idx].item())
            if shift_val != 0:
                # сдвигаем (roll) строку row_idx вдоль ширины (dim=3)
                out[:, :, row_idx, :] = torch.roll(out[:, :, row_idx, :], shifts=shift_val, dims=2)

        # Добавим «scan lines» — каждый 2-й (или 3-й) ряд темнее
        # intensity = 0.95..0.90, зависящая от vhs_strength
        line_dark = 1.0 - 0.15*vhs_strength  # ~0.85..1.0
        # Допустим, затемняем чётные строки
        out[:, :, 0::2, :] *= line_dark

        return out

    def add_random_glitches(self, img, glitch_strength):
        """
        Случайные блочные сдвиги
        glitch_strength ~ [0..1]
        """
        B, C, H, W = img.shape
        out = img.clone()

        # Сколько блоков? Допустим, 0..5
        max_blocks = 5
        num_blocks = int(max_blocks * glitch_strength)

        if num_blocks < 1:
            return out

        for b in range(B):
            # Для каждого изображения в батче
            for _ in range(num_blocks):
                # Случайная ширина/высота блока:
                block_w = max(8, int(W*0.1*torch.rand(1).item()))  # ~ до 10% ширины
                block_h = max(8, int(H*0.1*torch.rand(1).item()))  # ~ до 10% высоты

                # Случайная позиция блока (x0, y0)
                x0 = torch.randint(0, max(W-block_w, 1), (1,)).item()
                y0 = torch.randint(0, max(H-block_h, 1), (1,)).item()

                # Случайный сдвиг
                shift_x = torch.randint(-10, 11, (1,)).item()  # -10..10
                shift_y = 0  # Можно тоже random при желании

                # Выделим блок
                block = out[b, :, y0:y0+block_h, x0:x0+block_w]
                # сдвинем roll
                block = torch.roll(block, shifts=shift_x, dims=2)
                # При желании ещё вертикальный shift
                # block = torch.roll(block, shifts=shift_y, dims=1)

                # Запишем обратно
                out[b, :, y0:y0+block_h, x0:x0+block_w] = block

        return out
