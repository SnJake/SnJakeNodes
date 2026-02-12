import comfy.utils


class SnJakeResizeIfLarger:
    """
    Resize image only when enabled conditions are met:
    - enable_larger: if any side is larger than target_resolution
    - enable_smaller: if any side is smaller than target_resolution
    """

    FUNCTION = "resize_if_larger"
    CATEGORY = "😎 SnJake/Utils"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("resized_image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_resolution": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "step": 8,
                    },
                ),
                "enable_larger": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "enabled",
                        "label_off": "disabled",
                    },
                ),
                "enable_smaller": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "enabled",
                        "label_off": "disabled",
                    },
                ),
                "keep_aspect_ratio": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "enabled",
                        "label_off": "disabled",
                    },
                ),
                "upscale_method": (
                    ["lanczos", "bicubic", "bilinear", "nearest-exact", "area"],
                    {"default": "lanczos"},
                ),
            }
        }

    def resize_if_larger(
        self,
        image,
        target_resolution,
        enable_larger,
        enable_smaller,
        keep_aspect_ratio,
        upscale_method,
    ):
        # image shape: [B, H, W, C]
        _batch, height, width, _channels = image.shape

        has_larger = height > target_resolution or width > target_resolution
        has_smaller = height < target_resolution or width < target_resolution

        if not enable_larger and not enable_smaller:
            print(
                "SnJake Resize Warning: Both conditions are disabled "
                "(enable_larger=False, enable_smaller=False). Skipping."
            )
            return (image,)

        trigger_larger = enable_larger and has_larger
        trigger_smaller = enable_smaller and has_smaller
        should_resize = trigger_larger or trigger_smaller

        if not should_resize:
            print(
                f"SnJake Resize: Image is {width}x{height}, conditions not met for target "
                f"{target_resolution}px (enable_larger={enable_larger}, enable_smaller={enable_smaller}). "
                "Skipping."
            )
            return (image,)

        if keep_aspect_ratio:
            if trigger_larger and not trigger_smaller:
                selected_condition = "larger"
                reference_side = max(width, height)
            elif trigger_smaller and not trigger_larger:
                selected_condition = "smaller"
                reference_side = min(width, height)
            else:
                larger_ratio = max(width, height) / target_resolution
                smaller_ratio = target_resolution / min(width, height)
                if larger_ratio >= smaller_ratio:
                    selected_condition = "larger"
                    reference_side = max(width, height)
                else:
                    selected_condition = "smaller"
                    reference_side = min(width, height)

            scale_factor = target_resolution / reference_side
            new_width = max(1, int(round(width * scale_factor)))
            new_height = max(1, int(round(height * scale_factor)))
        else:
            selected_condition = "direct_resize"
            new_width = target_resolution
            new_height = target_resolution

        print(
            f"SnJake Resize: Resizing image from {width}x{height} to {new_width}x{new_height} "
            f"(selected_condition: {selected_condition}, enable_larger={enable_larger}, "
            f"enable_smaller={enable_smaller}) using {upscale_method}"
        )

        # comfy.utils expects [B, C, H, W]
        img_bchw = image.permute(0, 3, 1, 2)

        resized_img = comfy.utils.common_upscale(
            img_bchw,
            new_width,
            new_height,
            upscale_method,
            "disabled",
        )

        # back to [B, H, W, C]
        resized_img_bhwc = resized_img.permute(0, 2, 3, 1)
        return (resized_img_bhwc,)
