import math


class HunyuanImageResolutionCalc:
    """
    Расчет разрешения под HunyuanImage-2.1.

    Особенности:
      - Только нативные 2K-разрешения, рекомендованные авторами модели.
      - prefer_supported=True -> возвращает фиксированные размеры под выбранный AR.
      - prefer_supported=False -> рассчитывает по мегапикселям и кратности (использовать на свой риск).

    Параметры:
      - aspect_ratio: соотношение сторон или 'custom' (в custom используются custom_width/custom_height как ориентиp AR).
      - swap_wh: поменять местами ширину/высоту в конце.
      - divisible_by: кратность сторон (обычно 8/16/32).

    Возвращает:
      - width (INT), height (INT), resolution (STRING), info (STRING)
    """

    # Нативные 2K разрешения HunyuanImage-2.1 (из README):
    # - 1:1   -> 2048x2048
    # - 16:9  -> 2560x1536
    # - 9:16  -> 1536x2560
    # - 4:3   -> 2304x1792
    # - 3:4   -> 1792x2304
    # - 3:2   -> 2304x1536 (логичное дополнение для 2K, кратно 256)
    # - 2:3   -> 1536x2304 (логичное дополнение для 2K, кратно 256)
    SUPPORTED = {
        "1:1":  (2048, 2048),
        "16:9": (2560, 1536),
        "9:16": (1536, 2560),
        "4:3":  (2304, 1792),
        "3:4":  (1792, 2304),
        "3:2":  (2304, 1536),
        "2:3":  (1536, 2304),
    }

    CATEGORY = "😎 SnJake/Utils"
    RETURN_TYPES = ("INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("width", "height", "resolution", "info")
    FUNCTION = "calc"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "custom"], {"default": "1:1"}),
                "prefer_supported": ("BOOLEAN", {"default": True}),
                "megapixel": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.05}),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "swap_wh": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "custom_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 1}),
                "custom_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 1}),
            },
        }

    def _round_to_multiple(self, value, base):
        return max(base, int(round(value / base) * base))

    def _calc_by_megapixel(self, mp, ratio_w, ratio_h, divisible_by):
        total_pixels = mp * 1_000_000.0
        r = ratio_w / ratio_h
        width = math.sqrt(total_pixels * r)
        height = width / r
        w = self._round_to_multiple(width, divisible_by)
        h = self._round_to_multiple(height, divisible_by)
        return int(w), int(h)

    def calc(
        self,
        aspect_ratio,
        prefer_supported=True,
        megapixel=1.0,
        divisible_by=8,
        swap_wh=False,
        custom_width=1024,
        custom_height=1024,
    ):
        # Если выбраны нативные пресеты Hunyuan 2K — используем их.
        if prefer_supported and aspect_ratio != "custom":
            w, h = self.SUPPORTED[aspect_ratio]
        else:
            # Свободный расчет (не рекомендуется для Hunyuan 2.1, возможны артефакты на 1K)
            if aspect_ratio == "custom":
                ratio_w, ratio_h = float(custom_width), float(custom_height)
            else:
                rw, rh = self.SUPPORTED[aspect_ratio]
                ratio_w, ratio_h = float(rw), float(rh)

            g = math.gcd(int(ratio_w), int(ratio_h))
            ratio_w /= g
            ratio_h /= g

            w, h = self._calc_by_megapixel(megapixel, ratio_w, ratio_h, divisible_by)

        if swap_wh:
            w, h = h, w

        res = f"{w}x{h}"
        native = (aspect_ratio != "custom") and (w, h) == self.SUPPORTED.get(aspect_ratio, (None, None))
        info = (
            f"Hunyuan-2.1 | AR={aspect_ratio} | {res} | ~{(w*h)/1e6:.2f} MP | "
            f"{'native-2K' if native else f'div{divisible_by}'}"
        )
        return (int(w), int(h), res, info)

