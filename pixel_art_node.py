# --- START OF FILE pixel_art_node.py ---

import torch
import torch.nn.functional as F
import kornia # Убедитесь, что Kornia установлен
import kornia.geometry.transform as KGT # Explicit import for resize
import numpy as np
import math
import traceback
# import torchvision.transforms.functional as TF # No longer primary for Lanczos
from PIL import Image # For Lanczos fallback

# --- Импорт из сабмодулей ---
try:
    # Используем относительный импорт, предполагая, что структура папок верна
    from . import pixelart # Import the package
    # Access submodules via pixelart.*
    print("Successfully imported PixelArt submodules")
except ImportError as e:
     # Выводим более подробное сообщение об ошибке
     print(f"\n\n*****\nError importing PixelArt submodules: {e}\n"
           f"Please ensure the 'pixelart' folder exists inside the 'SnJakeNodes' custom_nodes directory "
           f"and contains all required .py files (__init__.py must also exist in both folders).\n"
           f"Full traceback: {traceback.format_exc()}\n*****\n")
     # Лучше вызвать исключение, чтобы ComfyUI не загрузил нерабочий узел
     raise ImportError("PixelArt submodules not found or failed to import.") from e


# --- Константы и Определение Узла ---
PALETTE_NODE_TYPE = "*" # Используется для типа выхода палитры

class PixelArtNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Список методов квантования
        quant_methods = ["kmeans", "median_cut", "wu", "octree", "SQ"]

        # Список стандартных палитр
        try:
            predefined_palette_names = list(pixelart.palettes.PREDEFINED_PALETTES.keys())
        except AttributeError:
             print("Warning: Could not load predefined palettes list from palettes.py. Using default.")
             predefined_palette_names = ["EGA", "C64"]

        # Downscale methods including PixelOE and Lanczos
        downscale_methods = ["Nearest", "Bilinear (Blurry)", "Bicubic (Blurry)", "Lanczos (Blurry)", "PixelOE Contrast-Aware"]

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image for pixelation."}),
                "use_pixelo_preprocessing": ("BOOLEAN", {"default": False, "tooltip": "Enable PixelOE-inspired preprocessing (Outline Expansion and/or Contrast-Aware Downsampling)."}),
                "pixelo_outline_expansion": ("BOOLEAN", {"default": True, "tooltip": "[PixelOE] Apply contrast-aware outline expansion before downsampling."}),
                "pixelo_median_kernel_size": ("INT", {"default": 5, "min": 3, "max": 15, "step": 2, "tooltip": "[PixelOE Outline] Kernel size for local median/min/max calculation."}),
                "pixelo_morph_kernel_size": ("INT", {"default": 3, "min": 3, "max": 9, "step": 2, "tooltip": "[PixelOE Outline] Kernel size for morphological closing/opening."}),
                "downscale_method": (downscale_methods, {"default": "Nearest", "tooltip": "Method used to reduce resolution initially. 'PixelOE' uses contrast-aware logic. 'Lanczos' is a high-quality interpolator."}),
                "pixel_size": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1, "display": "slider", "tooltip": "Size of the square 'pixel' block (used for Nearest/PixelOE/Blurry downsampling)."}),
                "reduce_color_palette": ("BOOLEAN", {"default": True, "tooltip": "Reduce the image's color count using the chosen method/palette."}),
                "use_predefined_palette": ("BOOLEAN", {"default": False, "tooltip": "Use a standard palette (EGA, C64, etc.). Overrides 'num_colors' and 'auto_num_colors'."}),
                "predefined_palette": (predefined_palette_names, {"default": "EGA", "tooltip": "Select a standard color palette."}),
                "use_custom_palette": ("BOOLEAN", {"default": False, "tooltip": "Use a custom palette specified in 'custom_palette_hex'. Overrides 'num_colors' and 'auto_num_colors'."}),
                "custom_palette_hex": ("STRING", {"default": "#FFFFFF, #000000", "multiline": False, "tooltip": "Define a custom palette as comma-separated hex codes (e.g., #FF0000, #00FF00, #0000FF)."}),
                "num_colors": ("INT", {"default": 16, "min": 2, "max": 256, "step": 1, "display": "slider", "tooltip": "Target number of colors if not using predefined/custom palette or auto_num_colors."}),
                "auto_num_colors": ("BOOLEAN", {"default": False, "tooltip": "Automatically determine optimal K using Davies-Bouldin Index (can be slow)."}),
                "auto_k_range": ("INT", {"default": 16, "min": 4, "max": 48, "step": 1, "display": "slider", "tooltip": "Max K to search when 'auto_num_colors' is enabled."}),
                "min_pixel_area": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1, "display": "slider", "tooltip": "Min pixels for a color cluster/palette color. Smaller groups get merged/reassigned. (1 = disabled)"}),
                "color_quantization_method": (quant_methods, {"default": "kmeans", "tooltip": "Algorithm for color reduction (quantization). SQ added."}),
                "quantize_in": (["RGB", "LAB", "YCbCr", "HSV"], {"default": "RGB", "tooltip": "Color space for distance calculations. SQ/KMeans/MedianCut benefit from perceptual spaces like LAB/YCbCr. HSV uses RGB internally for most methods."}),
                "max_iter": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1, "display": "slider", "tooltip": "Max iterations ONLY for K-Means."}),
                "sq_iterations_factor": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1, "tooltip": "SQ Iterations = factor * number_of_pixels..."}),
                "sq_learning_rate_initial": ("FLOAT", {"default": 0.1, "min": 1e-4, "max": 1.0, "step": 1e-3, "round": 0.0001, "tooltip": "Initial learning rate (step size) for SQ..."}),
                "sq_learning_rate_decay_time": ("INT", {"default": 10000, "min": 100, "max": 1000000, "step": 100, "tooltip": "Time constant (t0) for SQ learning rate decay..."}),
                "apply_dithering": ("BOOLEAN", {"default": False, "tooltip": "Enable dithering to simulate more colors than in the palette."}),
                "dither_pattern": (list(pixelart.dithering.DIFFUSION_PATTERNS.keys()) + list(pixelart.dithering.ORDERED_PATTERNS.keys()) + ["WhiteNoise"], {"default": "Floyd-Steinberg", "tooltip": "Dithering algorithm/pattern."}),
                "dither_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Intensity of the dithering effect."}),
                "color_distance_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Error Diffusion only: Don't diffuse error if original color distance > threshold (0=always diffuse)."}),
                "pixel_perfect": ("BOOLEAN", {"default": False, "tooltip": "Ensures sharp blocks: disables smoothing/filters, uses 'nearest' upscale."}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1, "tooltip": "Final image scaling factor relative to original size."}),
                "scale_method": (["nearest", "bilinear", "bicubic"], {"default": "nearest", "tooltip": "Interpolation for final scaling. 'nearest' recommended for pixel art, especially with 'pixel_perfect'."}),
                "smoothing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Smoothing (blur) strength applied after upscaling (ignored if 'pixel_perfect' is on)."}),
                "smoothing_filter": (["gaussian", "median", "bilateral"], {"default": "gaussian", "tooltip": "Type of basic smoothing filter (ignored if 'pixel_perfect' is on)."}),
                "advanced_filter": (["none", "guided", "kuwahara"], {"default": "none", "tooltip": "Additional filter applied after smoothing (ignored if 'pixel_perfect' is on)."}),
                "edge_preservation": ("BOOLEAN", {"default": False, "tooltip": "Attempt to preserve edges when smoothing (ignored if 'pixel_perfect' is on)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", PALETTE_NODE_TYPE)
    RETURN_NAMES = ("pixelated_image", "color_palette_hex", "palette_tensor")
    FUNCTION = "execute"
    CATEGORY = "😎 SnJake/PixelArt"

    def execute(self,
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

        print("--- Starting PixelArtNode Execution ---")

        valid = True
        if use_predefined_palette and use_custom_palette:
             print("ERROR: Select only one palette type: predefined OR custom.")
             valid = False
        if auto_num_colors and (use_predefined_palette or use_custom_palette):
             auto_num_colors = False
        if pixel_perfect:
             smoothing = 0
             advanced_filter = "none"
             # scale_method = "nearest" # Already handled by pixel_perfect logic later
        if pixel_size <= 0: pixel_size = 1
        if scale_factor <= 0: scale_factor = 1.0
        if num_colors < 2: num_colors = 2
        if auto_k_range < 2: auto_k_range = 2
        if min_pixel_area < 1: min_pixel_area = 1
        if pixelo_median_kernel_size % 2 == 0: pixelo_median_kernel_size += 1
        if pixelo_morph_kernel_size % 2 == 0: pixelo_morph_kernel_size += 1

        if not valid:
             return (image, "Input Validation Error", None)

        try:
            image_in = image.clone()
            batch_size, height, width, channels = image_in.shape
            if channels != 3:
                if channels == 4: image_in = image_in[:, :, :, :3]
                elif channels == 1: image_in = image_in.repeat(1, 1, 1, 3)
                else: raise ValueError(f"Cannot handle input image with {channels} channels.")
                batch_size, height, width, channels = image_in.shape

            original_dtype = image_in.dtype
            device = image_in.device
            image_proc = image_in.permute(0, 3, 1, 2).float() # (B, C, H, W), float
            original_size = (height, width)

        except Exception as e:
            print(f"Error during image preparation: {e}")
            traceback.print_exc()
            return (image, f"Error: Image Prep Failed - {e}", None)

        image_to_downscale = image_proc
        if use_pixelo_preprocessing and pixelo_outline_expansion:
            try:
                image_to_downscale = pixelart.pixelo_ops.contrast_aware_outline_expansion(
                    image_to_downscale,
                    median_kernel_size=pixelo_median_kernel_size,
                    morph_kernel_size=pixelo_morph_kernel_size
                )
            except Exception as e:
                print(f"Error during PixelOE Outline Expansion: {e}. Skipping.")
                traceback.print_exc()

        downscaled_image = None
        try:
            target_h = max(1, height // pixel_size)
            target_w = max(1, width // pixel_size)
            target_size_hw = (target_h, target_w) # PIL and TF.resize use (H,W) for target_size

            print(f"Downscaling to {target_w}x{target_h} using method '{downscale_method}' (pixel size: {pixel_size})")

            if downscale_method == "PixelOE Contrast-Aware":
                 if use_pixelo_preprocessing:
                     downscaled_image = pixelart.pixelo_ops.contrast_aware_downsample(
                         image_to_downscale, pixel_size
                     )
                 else:
                     print("Warning: PixelOE Downsampling selected, but 'use_pixelo_preprocessing' is OFF. Falling back to Nearest.")
                     downscaled_image = F.interpolate(image_to_downscale, size=target_size_hw, mode='nearest')
            elif downscale_method == "Nearest":
                downscaled_image = F.interpolate(image_to_downscale, size=target_size_hw, mode='nearest')
            elif downscale_method == "Bilinear (Blurry)":
                 downscaled_image = F.interpolate(image_to_downscale, size=target_size_hw, mode='bilinear', align_corners=False, antialias=True)
            elif downscale_method == "Bicubic (Blurry)":
                 downscaled_image = F.interpolate(image_to_downscale, size=target_size_hw, mode='bicubic', align_corners=False, antialias=True)
            elif downscale_method == "Lanczos (Blurry)":
                print(f"Using PIL Lanczos for downscaling. Input shape: {image_to_downscale.shape}, Target size (H,W): {target_size_hw}")
                processed_batches = []
                # image_to_downscale is (B, C, H, W) float [0,1]
                for i in range(image_to_downscale.shape[0]):
                    single_image_chw = image_to_downscale[i] # C, H, W
                    # Convert to PIL Image: C,H,W [0,1] -> H,W,C uint8 [0,255]
                    pil_img = Image.fromarray(
                        (single_image_chw.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    )
                    # PIL resize expects (W,H)
                    pil_resized = pil_img.resize((target_size_hw[1], target_size_hw[0]), Image.Resampling.LANCZOS)
                    
                    # Convert back to Tensor: H,W,C uint8 -> C,H,W float [0,1]
                    tensor_resized_chw = torch.from_numpy(
                        np.array(pil_resized).astype(np.float32) / 255.0
                    ).permute(2, 0, 1).to(device)
                    processed_batches.append(tensor_resized_chw)
                downscaled_image = torch.stack(processed_batches)
                print(f"PIL Lanczos output shape: {downscaled_image.shape}")
            else: 
                 print(f"Warning: Unknown downscale_method '{downscale_method}'. Using Nearest.")
                 downscaled_image = F.interpolate(image_to_downscale, size=target_size_hw, mode='nearest')

            downscaled_image = downscaled_image.float().clamp(0.0, 1.0)

        except Exception as e:
            print(f"Error during downscaling: {e}")
            traceback.print_exc()
            output_fallback = image_to_downscale.permute(0, 2, 3, 1).clamp(0, 1).to(original_dtype)
            return (output_fallback, f"Error: Downscale Failed - {e}", None)

        # --- Palette Determination ---
        active_palette_rgb = None
        palette_source = "none"
        final_num_colors = num_colors

        if palette_source == "none": 
             if use_custom_palette:
                 try:
                     active_palette_rgb = pixelart.palettes.parse_custom_palette(custom_palette_hex, device)
                     if active_palette_rgb is not None:
                         final_num_colors = active_palette_rgb.shape[0]
                         reduce_color_palette = True 
                         palette_source = "custom"
                         print(f"Using custom palette with {final_num_colors} colors.")
                 except Exception as e:
                     print(f"Error parsing custom palette: {e}")

             if palette_source == "none" and use_predefined_palette:
                 try:
                     active_palette_rgb = pixelart.palettes.get_predefined_palette(predefined_palette, device)
                     final_num_colors = active_palette_rgb.shape[0]
                     reduce_color_palette = True
                     palette_source = "predefined"
                     print(f"Using predefined palette '{predefined_palette}' with {final_num_colors} colors.")
                 except Exception as e:
                     print(f"Error loading predefined palette: {e}")

             if palette_source == "none" and reduce_color_palette:
                 palette_source = "quantized"
                 print(f"Using color quantization method '{color_quantization_method}' (Target K={num_colors}, Auto={auto_num_colors}).")

        # --- Determine Processing Space ---
        processing_space = quantize_in
        is_metric_sensitive_method = (
            palette_source in ["custom", "predefined", "optional"] or
            (palette_source == "quantized" and color_quantization_method in ["kmeans", "median_cut"])
        )
        if quantize_in == "HSV" and is_metric_sensitive_method:
             print(f"Warning: HSV space selected, but method is metric-sensitive. Forcing to RGB.")
             processing_space = "RGB"
        elif quantize_in == "HSV" and color_quantization_method in ["wu", "octree"]:
             print(f"Note: Method '{color_quantization_method}' operates on RGB internally for HSV.")

        # --- Color Reduction / Palette Application ---
        quantized_image = downscaled_image.clone() 
        effective_palette_in_processing_space = None
        labels_for_filtering = None

        if reduce_color_palette:
            print(f"Reducing colors in space: {processing_space} (User selected: {quantize_in})")
            try:
                image_for_quant = pixelart.color_utils.to_quantize_space(downscaled_image, processing_space)

                if palette_source in ["custom", "predefined", "optional"]:
                    active_palette_in_processing_space = pixelart.color_utils.to_quantize_space(
                        active_palette_rgb.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), processing_space
                    ).squeeze()
                    if active_palette_in_processing_space.ndim == 1:
                        active_palette_in_processing_space = active_palette_in_processing_space.unsqueeze(0)

                    quantized_image_in_space, labels_batch0 = pixelart.quantization.apply_fixed_palette_get_labels(
                        image_for_quant, active_palette_in_processing_space
                    )
                    effective_palette_in_processing_space = active_palette_in_processing_space.clone()
                    labels_for_filtering = labels_batch0

                elif palette_source == "quantized":
                    quant_params = {
                        "num_colors": final_num_colors,
                        "method": color_quantization_method,
                        "min_pixel_area": min_pixel_area,
                        "processing_space": processing_space,
                        "auto_num_colors": auto_num_colors,
                        "auto_k_range": auto_k_range,
                        "kmeans_max_iter": max_iter,
                        "sq_iterations_factor": sq_iterations_factor,
                        "sq_learning_rate_initial": sq_learning_rate_initial,
                        "sq_learning_rate_decay_time": sq_learning_rate_decay_time
                    }
                    quantized_image_in_space, final_centroids_in_space = pixelart.quantization.run_color_quantization(
                        image_for_quant, **quant_params
                    )
                    effective_palette_in_processing_space = final_centroids_in_space
                
                if palette_source != "quantized" and min_pixel_area > 1 and labels_for_filtering is not None and effective_palette_in_processing_space is not None:
                     # This condition should be: palette_source IN ["custom", "predefined"]
                     # and effective_palette_in_processing_space is not None
                     if palette_source in ["custom", "predefined", "optional"]: # "optional" kept for legacy if ever used
                        print("Applying min_pixel_area filter to fixed palette application results...")
                        # Ensure image_for_quant[0] is used for single batch processing for filtering context
                        pixels_flat = image_for_quant[0].permute(1, 2, 0).reshape(-1, image_for_quant.shape[1])
                        
                        filtered_labels, filtered_palette = pixelart.quantization.filter_palette_by_usage(
                            pixels_flat, labels_for_filtering, effective_palette_in_processing_space, min_pixel_area
                        )
                        if filtered_palette.shape[0] < effective_palette_in_processing_space.shape[0]:
                            quantized_image_in_space = pixelart.quantization.apply_fixed_palette(image_for_quant, filtered_palette) # Re-apply with new palette
                            effective_palette_in_processing_space = filtered_palette
                        else:
                            print("Info: min_pixel_area filter did not remove any palette colors.")


                if processing_space != "RGB":
                    quantized_image = pixelart.color_utils.from_quantize_space(quantized_image_in_space, processing_space)
                else:
                    quantized_image = quantized_image_in_space

            except Exception as e:
                print(f"ERROR during color reduction in space '{processing_space}': {e}")
                traceback.print_exc()
                quantized_image = pixelart.color_utils.to_quantize_space(downscaled_image, "RGB") 
                reduce_color_palette = False
                effective_palette_in_processing_space = None
        else:
            quantized_image = pixelart.color_utils.to_quantize_space(downscaled_image, "RGB")

        # --- Dithering ---
        effective_palette_rgb = None
        if effective_palette_in_processing_space is not None and effective_palette_in_processing_space.shape[0] > 0:
            try:
                effective_palette_rgb = pixelart.color_utils.from_quantize_space(
                    effective_palette_in_processing_space.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), processing_space
                ).squeeze()
                if effective_palette_rgb.ndim == 1: effective_palette_rgb = effective_palette_rgb.unsqueeze(0)
                effective_palette_rgb = effective_palette_rgb.clamp(0,1)
            except Exception as e:
                 print(f"Warning: Error converting final palette from {processing_space} to RGB for dithering: {e}")
                 effective_palette_rgb = None

        final_quantized_image_rgb = quantized_image.clamp(0, 1) 

        if apply_dithering and reduce_color_palette and effective_palette_rgb is not None and effective_palette_rgb.shape[0] > 1 and dither_strength > 0:
            print(f"Applying dithering ('{dither_pattern}')...")
            source_image_for_dither_rgb = pixelart.color_utils.to_quantize_space(downscaled_image, "RGB").clamp(0,1)
            try:
                 final_quantized_image_rgb = pixelart.dithering.apply_dithering(
                     source_image_for_dither_rgb,
                     effective_palette_rgb,
                     dither_pattern,
                     dither_strength,
                     color_distance_threshold,
                     pixelart.quantization.apply_fixed_palette 
                 )
            except Exception as e:
                 print(f"Error during dithering: {e}")
                 traceback.print_exc()
                 final_quantized_image_rgb = pixelart.quantization.apply_fixed_palette(source_image_for_dither_rgb, effective_palette_rgb)
        elif apply_dithering:
             print("Warning: Dithering skipped. Conditions not met.")

        # --- Final Processing Steps (Upscale, Smooth, Scale) ---
        processed_image = final_quantized_image_rgb

        try:
            # Determine upscale method for the first upscale to original size
            # `original_size` is (H,W)
            # `F.interpolate` size expects (H,W) if 4D tensor (B,C,H,W)
            # or (D,H,W) if 5D tensor (B,C,D,H,W)
            # Our `processed_image` is (B,C,h_small,w_small)
            # `original_size` is correct tuple (orig_H, orig_W)

            upscale_mode_to_orig = 'nearest' if pixel_perfect else scale_method # scale_method from input is nearest/bilinear/bicubic
            
            # Kornia's resize `interpolation` can be different if we want to use lanczos for upscaling
            # For now, stick to F.interpolate options for this main upscale
            
            print(f"Upscaling to {original_size[1]}x{original_size[0]} using '{upscale_mode_to_orig}'...")
            antialias_upscale = True if upscale_mode_to_orig != 'nearest' else None # F.interpolate antialias
            align_corners_upscale = True if upscale_mode_to_orig in ['bilinear', 'bicubic'] else None # For F.interpolate

            processed_image = F.interpolate(
                processed_image, 
                size=original_size, # Target (H_orig, W_orig)
                mode=upscale_mode_to_orig,
                align_corners=align_corners_upscale, 
                antialias=antialias_upscale
            )

            if not pixel_perfect and (smoothing > 0 or advanced_filter != "none"):
                print(f"Applying smoothing/filters (smooth={smoothing}, filter={smoothing_filter}, advanced={advanced_filter})...")
                processed_image = pixelart.filters.apply_smoothing(
                    processed_image, smoothing, smoothing_filter, edge_preservation, advanced_filter
                )

            scaled_image = processed_image
            if abs(scale_factor - 1.0) > 1e-4:
                final_scale_mode = 'nearest' if pixel_perfect else scale_method # scale_method is nearest/bilinear/bicubic
                
                scaled_h = max(1, int(round(original_size[0] * scale_factor)))
                scaled_w = max(1, int(round(original_size[1] * scale_factor)))
                final_target_size_hw = (scaled_h, scaled_w) # (H,W)

                print(f"Applying final scale factor {scale_factor} to {scaled_w}x{scaled_h} using '{final_scale_mode}'...")
                antialias_final = True if final_scale_mode != 'nearest' else None
                align_corners_final = True if final_scale_mode in ['bilinear', 'bicubic'] else None
                scaled_image = F.interpolate(
                    processed_image, 
                    size=final_target_size_hw, 
                    mode=final_scale_mode,
                    align_corners=align_corners_final, 
                    antialias=antialias_final
                )

        except Exception as e:
            print(f"Error during final processing (upscale/smooth/scale): {e}")
            traceback.print_exc()
            scaled_image = processed_image 
            # Convert back to original ComfyUI format
            output_image = scaled_image.permute(0, 2, 3, 1).clamp(0, 1).to(original_dtype)
            return (output_image, f"Error: Final Processing Failed - {e}", None)


        # --- Output Formatting ---
        output_image = scaled_image.permute(0, 2, 3, 1).clamp(0, 1).to(original_dtype)
        output_palette_str = "N/A"
        output_palette_tensor = None

        if reduce_color_palette and effective_palette_rgb is not None:
            output_palette_str = pixelart.color_utils.convert_palette_to_string(effective_palette_rgb)
            if effective_palette_rgb.ndim == 2 and effective_palette_rgb.shape[1] == 3:
                 output_palette_tensor = effective_palette_rgb.float().to(device)
            else:
                 print(f"Warning: Final RGB palette has unexpected shape {effective_palette_rgb.shape}. Cannot return tensor.")
                 output_palette_str += " (Tensor Invalid Shape)"
        elif not reduce_color_palette:
            output_palette_str = "Original colors (not reduced)"
        elif reduce_color_palette and effective_palette_rgb is None:
             output_palette_str = "Color reduction failed or produced no valid RGB palette"

        print("--- PixelArtNode Execution Finished ---")
        return (output_image, output_palette_str, output_palette_tensor)