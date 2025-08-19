import math

class QwenImageResolutionCalc:
    """
    Режимы:
      - prefer_supported=True -> отдаёт нативные пресеты Qwen-Image (строго фиксированные размеры).
      - prefer_supported=False -> считает по мегапикселям и аспекту с округлением до кратности (divisible_by).
    Опции:
      - aspect_ratio: выбор пресета или 'custom' (тогда используем custom_width/custom_height как базовый аспект).
      - swap_wh: быстро поменять местами ширину/высоту.
      - divisible_by: кратность для вычисляемого режима (обычно 8/16/32).
    Выходы:
      - width (INT), height (INT), resolution (STRING), info (STRING)
    """

    # Нативные размеры, предоставленные пользователем (можешь расширить по мере надобности)
    SUPPORTED = {
        "1:1":  (1328, 1328),
        "16:9": (1664,  928),
        "9:16": ( 928, 1664),
        "4:3":  (1472, 1140),
        "3:4":  (1140, 1472),
    }

    CATEGORY = "😎 SnJake/Utils
    RETURN_TYPES = ("INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("width", "height", "resolution", "info")
    FUNCTION = "calc"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4", "custom"], {"default": "1:1"}),
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
        # Округление к ближайшему кратному base
        return max(base, int(round(value / base) * base))

    def _calc_by_megapixel(self, mp, ratio_w, ratio_h, divisible_by):
        total_pixels = mp * 1_000_000.0
        # width = sqrt(P * r), height = width / r, где r = ratio_w/ratio_h
        r = ratio_w / ratio_h
        width = math.sqrt(total_pixels * r)
        height = width / r
        # Привязываем к кратности (обычно 8/16/32)
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
        # 1) Режим нативных пресетов Qwen-Image — строго фиксированные размеры
        if prefer_supported and aspect_ratio != "custom":
            w, h = self.SUPPORTED[aspect_ratio]
        else:
            # 2) Вычисляемые размеры по МП + аспект
            if aspect_ratio == "custom":
                ratio_w, ratio_h = float(custom_width), float(custom_height)
            else:
                rw, rh = self.SUPPORTED[aspect_ratio]
                ratio_w, ratio_h = float(rw), float(rh)

            # Нормализуем аспект до соотношения сторон (без масштаба)
            # Берём наименьшие целые пропорции
            g = math.gcd(int(ratio_w), int(ratio_h))
            ratio_w /= g
            ratio_h /= g

            w, h = self._calc_by_megapixel(megapixel, ratio_w, ratio_h, divisible_by)

        if swap_wh:
            w, h = h, w

        res = f"{w}x{h}"
        info = f"AR={aspect_ratio} | {res} | ~{(w*h)/1e6:.2f} MP | {'native' if prefer_supported and aspect_ratio!='custom' else f'div{divisible_by}'}"
        return (int(w), int(h), res, info)
