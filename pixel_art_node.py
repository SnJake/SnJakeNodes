import traceback

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from . import pixelart
except ImportError as e:
    raise ImportError("PixelArt submodules not found or failed to import.") from e


PALETTE_NODE_TYPE = "*"


class PixelArtNode:
    @classmethod
    def INPUT_TYPES(cls):
        quant_methods = ["kmeans", "median_cut", "octree", "SQ"]
        try:
            predefined_palette_names = list(pixelart.palettes.PREDEFINED_PALETTES.keys())
        except AttributeError:
            predefined_palette_names = ["EGA", "C64"]

        downscale_methods = [
            "Nearest",
            "Bilinear (Blurry)",
            "Bicubic (Blurry)",
            "Lanczos (Blurry)",
            "PixelOE Contrast-Aware",
        ]

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image for pixelation."}),
                "use_pixelo_preprocessing": ("BOOLEAN", {"default": False, "tooltip": "Enable PixelOE-inspired preprocessing (Outline Expansion and/or Contrast-Aware Downsampling)."}),
                "pixelo_outline_expansion": ("BOOLEAN", {"default": True, "tooltip": "[PixelOE] Apply contrast-aware outline expansion before downsampling."}),
                "pixelo_median_kernel_size": ("INT", {"default": 5, "min": 3, "max": 15, "step": 2, "tooltip": "[PixelOE Outline] Kernel size for local median/min/max calculation."}),
                "pixelo_morph_kernel_size": ("INT", {"default": 3, "min": 3, "max": 9, "step": 2, "tooltip": "[PixelOE Outline] Kernel size for morphological closing/opening."}),
                "downscale_method": (downscale_methods, {"default": "Nearest", "tooltip": "Method used to reduce resolution initially. 'PixelOE' uses contrast-aware logic. 'Lanczos' is a high-quality interpolator."}),
                "pixel_size": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1, "display": "slider", "tooltip": "Size of the square 'pixel' block."}),
                "reduce_color_palette": ("BOOLEAN", {"default": True, "tooltip": "Reduce the image's color count."}),
                "use_predefined_palette": ("BOOLEAN", {"default": False, "tooltip": "Use a standard palette. Overrides 'num_colors' and 'auto_num_colors'."}),
                "predefined_palette": (predefined_palette_names, {"default": "EGA", "tooltip": "Select a standard color palette."}),
                "use_custom_palette": ("BOOLEAN", {"default": False, "tooltip": "Use a custom palette from 'custom_palette_hex'. Overrides 'num_colors' and 'auto_num_colors'."}),
                "custom_palette_hex": ("STRING", {"default": "#FFFFFF, #000000", "multiline": False, "tooltip": "Comma-separated hex palette."}),
                "num_colors": ("INT", {"default": 16, "min": 2, "max": 256, "step": 1, "display": "slider", "tooltip": "Target number of colors."}),
                "auto_num_colors": ("BOOLEAN", {"default": False, "tooltip": "Automatically determine K using DBI."}),
                "auto_k_range": ("INT", {"default": 16, "min": 4, "max": 48, "step": 1, "display": "slider", "tooltip": "Max K for DBI search."}),
                "min_pixel_area": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1, "display": "slider", "tooltip": "Minimum pixel count per color cluster."}),
                "color_quantization_method": (quant_methods, {"default": "kmeans", "tooltip": "Color quantization algorithm."}),
                "quantize_in": (["RGB", "LAB", "YCbCr", "HSV"], {"default": "RGB", "tooltip": "Color space for distance calculations."}),
                "max_iter": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1, "display": "slider", "tooltip": "Max iterations for K-Means."}),
                "sq_iterations_factor": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1, "tooltip": "SQ iterations factor."}),
                "sq_learning_rate_initial": ("FLOAT", {"default": 0.1, "min": 1e-4, "max": 1.0, "step": 1e-3, "round": 0.0001, "tooltip": "Initial SQ learning rate."}),
                "sq_learning_rate_decay_time": ("INT", {"default": 10000, "min": 100, "max": 1000000, "step": 100, "tooltip": "SQ learning rate decay time constant."}),
                "apply_dithering": ("BOOLEAN", {"default": False, "tooltip": "Enable dithering."}),
                "dither_pattern": (list(pixelart.dithering.DIFFUSION_PATTERNS.keys()) + list(pixelart.dithering.ORDERED_PATTERNS.keys()) + ["WhiteNoise"], {"default": "Floyd-Steinberg", "tooltip": "Dithering pattern."}),
                "dither_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Dithering strength."}),
                "color_distance_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Error diffusion threshold."}),
                "pixel_perfect": ("BOOLEAN", {"default": False, "tooltip": "Disable smoothing and force nearest upscaling."}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1, "tooltip": "Final image scaling factor."}),
                "scale_method": (["nearest", "bilinear", "bicubic"], {"default": "nearest", "tooltip": "Interpolation for final scaling."}),
                "smoothing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Smoothing strength after upscaling."}),
                "smoothing_filter": (["gaussian", "median", "bilateral"], {"default": "gaussian", "tooltip": "Smoothing filter type."}),
                "advanced_filter": (["none", "guided", "kuwahara"], {"default": "none", "tooltip": "Additional post-smoothing filter."}),
                "edge_preservation": ("BOOLEAN", {"default": False, "tooltip": "Preserve edges during smoothing."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", PALETTE_NODE_TYPE)
    RETURN_NAMES = ("pixelated_image", "color_palette_hex", "palette_tensor")
    FUNCTION = "execute"
    CATEGORY = "😎 SnJake/PixelArt"

    @staticmethod
    def _interpolate(image, size_hw, mode):
        kwargs = {"size": size_hw, "mode": mode}
        if mode in ("bilinear", "bicubic"):
            kwargs["align_corners"] = False
            kwargs["antialias"] = True
        return F.interpolate(image, **kwargs)

    @staticmethod
    def _downscale_lanczos_pil(image_bchw, target_h, target_w, device):
        batches = []
        for i in range(image_bchw.shape[0]):
            pil_img = Image.fromarray((image_bchw[i].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8))
            pil_resized = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            tensor_resized = torch.from_numpy(np.asarray(pil_resized).astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
            batches.append(tensor_resized)
        return torch.stack(batches, dim=0)

    def execute(
        self,
        image,
        use_pixelo_preprocessing,
        pixelo_outline_expansion,
        pixelo_median_kernel_size,
        pixelo_morph_kernel_size,
        downscale_method,
        pixel_size,
        reduce_color_palette,
        use_predefined_palette,
        predefined_palette,
        use_custom_palette,
        custom_palette_hex,
        num_colors,
        auto_num_colors,
        auto_k_range,
        min_pixel_area,
        color_quantization_method,
        quantize_in,
        max_iter,
        sq_iterations_factor,
        sq_learning_rate_initial,
        sq_learning_rate_decay_time,
        apply_dithering,
        dither_pattern,
        dither_strength,
        color_distance_threshold,
        pixel_perfect,
        scale_factor,
        scale_method,
        smoothing,
        smoothing_filter,
        advanced_filter,
        edge_preservation,
    ):
        if use_predefined_palette and use_custom_palette:
            return (image, "Input Validation Error: select only one palette source", None)

        if auto_num_colors and (use_predefined_palette or use_custom_palette):
            auto_num_colors = False

        if pixel_perfect:
            smoothing = 0.0
            advanced_filter = "none"

        pixel_size = max(1, int(pixel_size))
        num_colors = max(2, int(num_colors))
        auto_k_range = max(2, int(auto_k_range))
        min_pixel_area = max(1, int(min_pixel_area))
        scale_factor = max(0.1, float(scale_factor))
        pixelo_median_kernel_size = max(3, int(pixelo_median_kernel_size) | 1)
        pixelo_morph_kernel_size = max(3, int(pixelo_morph_kernel_size) | 1)

        try:
            image_in = image.clone()
            batch_size, height, width, channels = image_in.shape
            if channels == 4:
                image_in = image_in[:, :, :, :3]
            elif channels == 1:
                image_in = image_in.repeat(1, 1, 1, 3)
            elif channels != 3:
                raise ValueError(f"Unsupported channel count: {channels}")

            original_dtype = image_in.dtype
            device = image_in.device
            image_proc = image_in.permute(0, 3, 1, 2).float().clamp(0.0, 1.0)
            original_size = (image_proc.shape[2], image_proc.shape[3])
        except Exception as e:
            traceback.print_exc()
            return (image, f"Error: Image Prep Failed - {e}", None)

        image_to_downscale = image_proc
        if use_pixelo_preprocessing and pixelo_outline_expansion:
            try:
                image_to_downscale = pixelart.pixelo_ops.contrast_aware_outline_expansion(
                    image_to_downscale,
                    median_kernel_size=pixelo_median_kernel_size,
                    morph_kernel_size=pixelo_morph_kernel_size,
                )
            except Exception:
                traceback.print_exc()

        try:
            target_h = max(1, original_size[0] // pixel_size)
            target_w = max(1, original_size[1] // pixel_size)
            target_size_hw = (target_h, target_w)

            if downscale_method == "PixelOE Contrast-Aware":
                if use_pixelo_preprocessing:
                    downscaled_image = pixelart.pixelo_ops.contrast_aware_downsample(image_to_downscale, pixel_size)
                else:
                    downscaled_image = self._interpolate(image_to_downscale, target_size_hw, "nearest")
            elif downscale_method == "Nearest":
                downscaled_image = self._interpolate(image_to_downscale, target_size_hw, "nearest")
            elif downscale_method == "Bilinear (Blurry)":
                downscaled_image = self._interpolate(image_to_downscale, target_size_hw, "bilinear")
            elif downscale_method == "Bicubic (Blurry)":
                downscaled_image = self._interpolate(image_to_downscale, target_size_hw, "bicubic")
            elif downscale_method == "Lanczos (Blurry)":
                downscaled_image = self._downscale_lanczos_pil(image_to_downscale, target_h, target_w, device)
            else:
                downscaled_image = self._interpolate(image_to_downscale, target_size_hw, "nearest")

            downscaled_image = downscaled_image.float().clamp(0.0, 1.0)
        except Exception as e:
            traceback.print_exc()
            fallback = image_to_downscale.permute(0, 2, 3, 1).clamp(0.0, 1.0).to(original_dtype)
            return (fallback, f"Error: Downscale Failed - {e}", None)

        active_palette_rgb = None
        palette_source = "none"
        final_num_colors = num_colors

        if use_custom_palette:
            try:
                active_palette_rgb = pixelart.palettes.parse_custom_palette(custom_palette_hex, device)
                if active_palette_rgb is not None and active_palette_rgb.shape[0] > 0:
                    palette_source = "custom"
                    final_num_colors = int(active_palette_rgb.shape[0])
                    reduce_color_palette = True
            except Exception:
                traceback.print_exc()

        if palette_source == "none" and use_predefined_palette:
            try:
                active_palette_rgb = pixelart.palettes.get_predefined_palette(predefined_palette, device)
                if active_palette_rgb is not None and active_palette_rgb.shape[0] > 0:
                    palette_source = "predefined"
                    final_num_colors = int(active_palette_rgb.shape[0])
                    reduce_color_palette = True
            except Exception:
                traceback.print_exc()

        if palette_source == "none" and reduce_color_palette:
            palette_source = "quantized"

        processing_space = quantize_in
        metric_sensitive = palette_source in {"custom", "predefined"} or (
            palette_source == "quantized" and color_quantization_method.strip().lower() in {"kmeans", "median_cut", "sq"}
        )
        if quantize_in == "HSV" and metric_sensitive:
            processing_space = "RGB"

        quantized_image = downscaled_image.clone()
        effective_palette_in_space = None
        labels_for_filtering = None

        if reduce_color_palette:
            try:
                image_for_quant = pixelart.color_utils.to_quantize_space(downscaled_image, processing_space)

                if palette_source in {"custom", "predefined"}:
                    palette_in_space = pixelart.color_utils.to_quantize_space(
                        active_palette_rgb.unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
                        processing_space,
                    ).squeeze()
                    if palette_in_space.ndim == 1:
                        palette_in_space = palette_in_space.unsqueeze(0)

                    quantized_image_in_space, labels_batch0 = pixelart.quantization.apply_fixed_palette_get_labels(
                        image_for_quant,
                        palette_in_space,
                    )
                    effective_palette_in_space = palette_in_space.clone()
                    labels_for_filtering = labels_batch0
                elif palette_source == "quantized":
                    quantized_image_in_space, final_centroids_in_space = pixelart.quantization.run_color_quantization(
                        image_for_quant,
                        num_colors=final_num_colors,
                        method=color_quantization_method,
                        min_pixel_area=min_pixel_area,
                        processing_space=processing_space,
                        auto_num_colors=auto_num_colors,
                        auto_k_range=auto_k_range,
                        kmeans_max_iter=max_iter,
                        sq_iterations_factor=sq_iterations_factor,
                        sq_learning_rate_initial=sq_learning_rate_initial,
                        sq_learning_rate_decay_time=sq_learning_rate_decay_time,
                    )
                    effective_palette_in_space = final_centroids_in_space
                else:
                    quantized_image_in_space = image_for_quant

                if (
                    palette_source in {"custom", "predefined"}
                    and min_pixel_area > 1
                    and labels_for_filtering is not None
                    and effective_palette_in_space is not None
                    and image_for_quant.shape[0] > 0
                ):
                    pixels_flat = image_for_quant[0].permute(1, 2, 0).reshape(-1, image_for_quant.shape[1])
                    _, filtered_palette = pixelart.quantization.filter_palette_by_usage(
                        pixels_flat,
                        labels_for_filtering,
                        effective_palette_in_space,
                        min_pixel_area,
                    )
                    if filtered_palette is not None and filtered_palette.shape[0] > 0 and filtered_palette.shape[0] < effective_palette_in_space.shape[0]:
                        quantized_image_in_space = pixelart.quantization.apply_fixed_palette(image_for_quant, filtered_palette)
                        effective_palette_in_space = filtered_palette

                if processing_space != "RGB":
                    quantized_image = pixelart.color_utils.from_quantize_space(quantized_image_in_space, processing_space)
                else:
                    quantized_image = quantized_image_in_space
            except Exception as e:
                traceback.print_exc()
                quantized_image = downscaled_image.clone()
                reduce_color_palette = False
                effective_palette_in_space = None
        else:
            quantized_image = downscaled_image.clone()

        effective_palette_rgb = None
        if effective_palette_in_space is not None and effective_palette_in_space.shape[0] > 0:
            try:
                effective_palette_rgb = pixelart.color_utils.from_quantize_space(
                    effective_palette_in_space.unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
                    processing_space,
                ).squeeze()
                if effective_palette_rgb.ndim == 1:
                    effective_palette_rgb = effective_palette_rgb.unsqueeze(0)
                effective_palette_rgb = effective_palette_rgb.clamp(0.0, 1.0)
            except Exception:
                traceback.print_exc()
                effective_palette_rgb = None

        final_quantized_image_rgb = quantized_image.clamp(0.0, 1.0)

        if (
            apply_dithering
            and reduce_color_palette
            and effective_palette_rgb is not None
            and effective_palette_rgb.shape[0] > 0
            and dither_strength > 0
        ):
            try:
                source_image_for_dither_rgb = downscaled_image.clamp(0.0, 1.0)
                final_quantized_image_rgb = pixelart.dithering.apply_dithering(
                    source_image_for_dither_rgb,
                    effective_palette_rgb,
                    dither_pattern,
                    dither_strength,
                    color_distance_threshold,
                    pixelart.quantization.apply_fixed_palette,
                )
            except Exception:
                traceback.print_exc()
                final_quantized_image_rgb = pixelart.quantization.apply_fixed_palette(
                    downscaled_image.clamp(0.0, 1.0),
                    effective_palette_rgb,
                )

        processed_image = final_quantized_image_rgb
        try:
            mode_to_orig = "nearest" if pixel_perfect else scale_method
            processed_image = self._interpolate(processed_image, original_size, mode_to_orig)

            if not pixel_perfect and (smoothing > 0 or advanced_filter != "none"):
                processed_image = pixelart.filters.apply_smoothing(
                    processed_image,
                    smoothing,
                    smoothing_filter,
                    edge_preservation,
                    advanced_filter,
                )

            scaled_image = processed_image
            if abs(scale_factor - 1.0) > 1e-4:
                final_scale_mode = "nearest" if pixel_perfect else scale_method
                scaled_h = max(1, int(round(original_size[0] * scale_factor)))
                scaled_w = max(1, int(round(original_size[1] * scale_factor)))
                scaled_image = self._interpolate(processed_image, (scaled_h, scaled_w), final_scale_mode)
        except Exception as e:
            traceback.print_exc()
            out = processed_image.permute(0, 2, 3, 1).clamp(0.0, 1.0).to(original_dtype)
            return (out, f"Error: Final Processing Failed - {e}", None)

        output_image = scaled_image.permute(0, 2, 3, 1).clamp(0.0, 1.0).to(original_dtype)
        output_palette_str = "N/A"
        output_palette_tensor = None

        if reduce_color_palette and effective_palette_rgb is not None:
            output_palette_str = pixelart.color_utils.convert_palette_to_string(effective_palette_rgb)
            if effective_palette_rgb.ndim == 2 and effective_palette_rgb.shape[1] == 3:
                output_palette_tensor = effective_palette_rgb.float().to(device)
            else:
                output_palette_str += " (Tensor Invalid Shape)"
        elif not reduce_color_palette:
            output_palette_str = "Original colors (not reduced)"
        else:
            output_palette_str = "Color reduction failed or produced no valid RGB palette"

        return (output_image, output_palette_str, output_palette_tensor)
