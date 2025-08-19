import math

class QwenImageResolutionCalc:
    """
    Режимы:
      - prefer_supported=True -> отдаёт нативные пресеты Qwen-Image (фиксированные размеры).
      - prefer_supported=False -> считает по мегапикселям и аспекту с округлением до кратности (divisible_by),
        КРОМЕ режима aspect_ratio="custom": там megapixel игнорируется, берём custom_width/custom_height.
    Опции:
      - aspect_ratio: выбор пресета или 'custom'.
      - divisible_by: кратность (обычно 8/16/32).
      - swap_wh: поменять местами ширину/высоту.
    Выходы:
      - width (INT), height (INT), resolution (STRING), info (STRING)
    """

    SUPPORTED = {
        "1:1":  (1328, 1328),
        "16:9": (1664,  928),
        "9:16": ( 928, 1664),
        "4:3":  (1472, 1140),
        "3:4":  (1140, 1472),
    }

    CATEGORY = "😎 SnJake/Utils"
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
        label = ""

        # 1) Нативные пресеты Qwen-Image — строго фиксированные размеры
        if prefer_supported and aspect_ratio != "custom":
            w, h = self.SUPPORTED[aspect_ratio]
            label = "native"

        else:
            # 2) Пользовательский режим (megapixel игнорируется)
            if aspect_ratio == "custom":
                w = self._round_to_multiple(custom_width, divisible_by)
                h = self._round_to_multiple(custom_height, divisible_by)
                label = "custom"
            else:
                # 3) Вычисление по мегапикселям для выбранного аспекта
                rw, rh = self.SUPPORTED[aspect_ratio]
                # нормализуем аспект до целых пропорций
                g = math.gcd(int(rw), int(rh))
                ratio_w = rw / g
                ratio_h = rh / g
                w, h = self._calc_by_megapixel(megapixel, ratio_w, ratio_h, divisible_by)
                label = f"div{divisible_by}"

        if swap_wh:
            w, h = h, w

        res = f"{w}x{h}"
        info = f"AR={aspect_ratio} | {res} | ~{(w*h)/1e6:.2f} MP | {label}"
        return (int(w), int(h), res, info)
