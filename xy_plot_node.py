import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
import json
import math
import time
import random
import hashlib
import uuid
import shutil
import logging

import comfy.utils # Импортируем comfy.utils для доступа к lanczos
import comfy.sample
import comfy.sd
import comfy.model_management
import folder_paths
import latent_preview
import nodes
from comfy.comfy_types import ComfyNodeABC, InputTypeDict, IO


# Импорт bislerp из подпапки utils_snjake (ваш локальный bislerp)
try:
    from .utils_snjake.bislerp_standalone import bislerp as local_bislerp
except ImportError:
    logging.error("[XY Plot Advanced] Failed to import local_bislerp relatively. Attempting direct import strategy (may fail in ComfyUI).")
    try:
        import SnJakeNodes.utils_snjake.bislerp_standalone
        local_bislerp = SnJakeNodes.utils_snjake.bislerp_standalone.bislerp
        logging.info("[XY Plot Advanced] Successfully imported local_bislerp via SnJakeNodes path.")
    except ImportError as e_direct:
        logging.error(f"[XY Plot Advanced] CRITICAL: Could not import local_bislerp. Hires Fix with bislerp will not work. Error: {e_direct}")
        def local_bislerp(samples, width, height): # Fallback local_bislerp
            logging.error("local_bislerp function is not available. Using F.interpolate as fallback.")
            return torch.nn.functional.interpolate(samples, size=(height, width), mode='bilinear', align_corners=False)

# Default Font Search Paths
DEFAULT_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", # Linux common
    "/Library/Fonts/Arial.ttf",                       # macOS common
    "C:/Windows/Fonts/arial.ttf",                     # Windows common
]

class XYPlotAdvanced:
    AXIS_TYPES = ["Nothing", "Seed", "Steps", "CFG", "Sampler", "Scheduler", "Prompt S/R"]
    CONCAT_DIRECTIONS = ["X (Rows First)", "Y (Columns First)"]
    HIRES_RESCALE_METHODS = ["bislerp", "lanczos"] 

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        try:
            sampler_list = comfy.samplers.KSampler.SAMPLERS
            scheduler_list = comfy.samplers.KSampler.SCHEDULERS
            if not sampler_list: sampler_list = ["euler"]
            if not scheduler_list: scheduler_list = ["normal"]
        except AttributeError:
            logging.warning("[XY Plot Advanced] Could not get sampler/scheduler lists from comfy.samplers. Using defaults.")
            sampler_list = ["euler", "dpm_2", "lcm"]
            scheduler_list = ["normal", "karras", "simple"]
        except Exception as e:
            logging.error(f"[XY Plot Advanced] Unexpected error getting sampler/scheduler lists: {e}. Using defaults.")
            sampler_list = ["euler", "dpm_2", "lcm"]
            scheduler_list = ["normal", "karras", "simple"]

        return {
            "required": {
                "model": (IO.MODEL,),
                "clip": (IO.CLIP,),
                "vae": (IO.VAE,),
                "positive": (IO.STRING, {"multiline": True, "default": ""}),
                "negative": (IO.STRING, {"multiline": True, "default": ""}),
                "latent_image": (IO.LATENT, {"tooltip": "Initial latent image. Defines dimensions."}),
                "seed": (IO.INT, {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": (IO.INT, {"default": 20, "min": 1, "max": 10000}),
                "cfg": (IO.FLOAT, {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (sampler_list,),
                "scheduler": (scheduler_list,),

                "x_axis_type": (cls.AXIS_TYPES, {"default": "Prompt S/R"}),
                "x_axis_values": (IO.STRING, {"multiline": True, "default": "cat, dog, car", "tooltip": "Comma-separated values for the X-axis."}),
                "y_axis_type": (cls.AXIS_TYPES, {"default": "Seed"}),
                "y_axis_values": (IO.STRING, {"multiline": True, "default": "1, 2, 3", "tooltip": "Comma-separated values for the Y-axis."}),

                "placeholder": (IO.STRING, {"default": "XYZ_PROMPT", "tooltip": "Placeholder in prompts to replace for 'Prompt S/R' axis type."}),
                "concat_direction": (cls.CONCAT_DIRECTIONS, {"default": "X (Rows First)"}),
                "header_font_size": (IO.INT, {"default": 24, "min": 8, "max": 128}),
                "header_height": (IO.INT, {"default": 60, "min": 0, "max": 500, "tooltip": "Height of the X-axis header bar. Set to 0 to disable."}), # Allow 0

                "enable_hires_fix": ("BOOLEAN", {"default": False, "tooltip": "Enable second pass Hires Fix."}),
                "upscaler": (IO.UPSCALE_MODEL, {"tooltip":"Upscale model used for Hires Fix."}),
                "hires_target_upscale_factor": (IO.FLOAT, {"default": 1.5, "min": 0.5, "max": 8.0, "step": 0.05, "tooltip": "Target upscale factor for Hires Fix (relative to original size)."}),
                "hires_rescale_method": (cls.HIRES_RESCALE_METHODS, {"default": "bislerp", "tooltip": "Method for rescaling the image after model upscaling in Hires Fix."}),
                "hires_steps": (IO.INT, {"default": 15, "min": 1, "max": 10000, "tooltip":"Steps for the Hires Fix pass."}),
                "hires_denoise": (IO.FLOAT, {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":"Denoise strength for the Hires Fix pass."}),
                "hires_sampler_name": (sampler_list, {"default": "euler", "tooltip":"Sampler for the Hires Fix pass."}),
                "hires_scheduler": (scheduler_list, {"default": "normal", "tooltip":"Scheduler for the Hires Fix pass."}),
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("xy_plot_image",)
    FUNCTION = "generate_plot"
    CATEGORY = "😎 SnJake/XY Plot"

    def _find_font(self, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        # ... (no changes to this helper) ...
        for font_path in DEFAULT_FONT_PATHS:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, font_size)
                except IOError as e:
                    logging.warning(f"[XY Plot Advanced] IOError loading font {font_path}: {e}")
                    continue
                except Exception as e: 
                    logging.warning(f"[XY Plot Advanced] General error loading font {font_path}: {e}")
                    continue
        logging.warning("[XY Plot Advanced] Could not find common fonts. Using PIL default.")
        try:
             return ImageFont.load_default()
        except IOError:
             logging.error("[XY Plot Advanced] Could not load even the default PIL font.")
             return ImageFont.ImageFont()

    def _measure_text(self, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, font_size_fallback: int) -> tuple[int, int]:
        # ... (no changes to this helper) ...
        if not text: return 0, 0
        try:
            dummy_image = Image.new('RGB', (1, 1)) 
            draw_context = ImageDraw.Draw(dummy_image)
            if hasattr(draw_context, 'textbbox'): 
                bbox = draw_context.textbbox((0, 0), text, font=font, anchor="lt") 
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                return width, height
            elif hasattr(font, 'getbbox'): 
                bbox = font.getbbox(text)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1] 
                return width, height
            elif hasattr(draw_context, 'textlength'): 
                width = draw_context.textlength(text, font=font)
                if hasattr(font, 'getmetrics'): 
                    ascent, descent = font.getmetrics()
                    height = ascent + descent
                else: 
                    try: height = font.size
                    except AttributeError: height = font_size_fallback 
                return int(width), int(height)
            else: 
                if hasattr(font, 'size'):
                    est_char_width = font.size * 0.6 
                    height = font.size
                else:
                    est_char_width = font_size_fallback * 0.6
                    height = font_size_fallback
                return int(len(text) * est_char_width), height
        except Exception as e:
            logging.warning(f"[XY Plot Advanced] Error measuring text '{text}': {e}. Using fallback estimate.")
            return int(len(text) * font_size_fallback * 0.6), font_size_fallback

    def _create_text_image(self, text_lines: list[str], image_width: int, fixed_header_height: int, font_size: int) -> torch.Tensor:
        # ... (no changes to this helper, fixed_header_height is used directly) ...
        img = Image.new('RGB', (image_width, fixed_header_height), color='white')
        draw = ImageDraw.Draw(img)
        font = self._find_font(font_size)
        total_text_block_height = 0
        line_dimensions_data = [] 
        padding_between_lines = 2 
        for i, line in enumerate(text_lines):
            line_w, line_h = self._measure_text(line, font, font_size)
            line_dimensions_data.append({'text': line, 'width': line_w, 'height': line_h})
            total_text_block_height += line_h
            if i < len(text_lines) - 1: total_text_block_height += padding_between_lines
        current_y = max(5, (fixed_header_height - total_text_block_height) / 2) 
        for line_data in line_dimensions_data:
            line_text = line_data['text']
            line_width = line_data['width']
            line_height = line_data['height']
            text_x = max(5, (image_width - line_width) / 2) 
            try:
                draw.text((text_x, current_y), line_text, fill='black', font=font, anchor="lt")
            except Exception as e:
                logging.error(f"[XY Plot Advanced] Error drawing text line '{line_text}' in top header: {e}")
                err_font = self._find_font(max(8, font_size // 2)) 
                draw.text((5, current_y), "Text Error", fill='red', font=err_font, anchor="lt") 
            current_y += line_height + padding_between_lines
        image_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(image_np)[None,].cpu()


    def _create_vertical_text_image(self, text_content: str, target_image_height: int, font_size: int, fixed_y_label_width: int) -> torch.Tensor:
        # ... (no changes to this helper, target_image_height and fixed_y_label_width are used) ...
        font = self._find_font(font_size)
        img = Image.new('RGB', (fixed_y_label_width, target_image_height), color='white')
        draw = ImageDraw.Draw(img)
        if text_content: 
            text_w, text_h = self._measure_text(text_content, font, font_size)
            text_x = max(0, (fixed_y_label_width - text_w) / 2) 
            text_y = max(0, (target_image_height - text_h) / 2) 
            try:
                draw.text((text_x, text_y), text_content, fill='black', font=font, anchor="lt") 
            except Exception as e:
                logging.error(f"[XY Plot Advanced] Error drawing text for vertical header '{text_content}': {e}")
                err_font_size = max(8, font_size // 2)
                err_font = self._find_font(err_font_size)
                err_text_w, _ = self._measure_text("Text Error", err_font, err_font_size)
                err_text_x = max(0, (fixed_y_label_width - err_text_w) / 2)
                draw.text((err_text_x , 5), "Text Error", fill='red', font=err_font, anchor="lt")
        image_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(image_np)[None,].cpu()

    def generate_plot(self, model, clip, vae,
                      positive, negative, latent_image,
                      seed, steps, cfg, sampler_name, scheduler,
                      x_axis_type, x_axis_values,
                      y_axis_type, y_axis_values,
                      placeholder, concat_direction, header_font_size, header_height, # header_height is for X-axis
                      enable_hires_fix, upscaler,
                      hires_target_upscale_factor,
                      hires_rescale_method, 
                      hires_steps, hires_denoise,
                      hires_sampler_name, hires_scheduler):

        x_values_raw = [v.strip() for v in x_axis_values.split(',') if v.strip()]
        y_values_raw = [v.strip() for v in y_axis_values.split(',') if v.strip()]

        if x_axis_type == "Nothing": x_values_iter = ["N/A_X_internal"] 
        else: x_values_iter = x_values_raw if x_values_raw else ["(empty X)"]
        
        if y_axis_type == "Nothing": y_values_iter = ["N/A_Y_internal"]
        else: y_values_iter = y_values_raw if y_values_raw else ["(empty Y)"]
        
        num_x = len(x_values_iter)
        num_y = len(y_values_iter)
        total_images = num_x * num_y

        try:
            valid_sampler_list = comfy.samplers.KSampler.SAMPLERS
            valid_scheduler_list = comfy.samplers.KSampler.SCHEDULERS
        except AttributeError: 
            valid_sampler_list = ["euler", "dpm_2", "lcm"] 
            valid_scheduler_list = ["normal", "karras", "simple"] 

        logging.info(f"[XY Plot Advanced] Starting generation of {total_images} images ({num_x} X values, {num_y} Y values). Hires Fix: {'Enabled' if enable_hires_fix else 'Disabled'}.")

        plot_temp_path = ""
        try:
            temp_dir = folder_paths.get_temp_directory()
            if not os.path.isdir(temp_dir): os.makedirs(temp_dir, exist_ok=True)
            unique_folder_name = f"xy_plot_temp_{uuid.uuid4()}" 
            plot_temp_path = os.path.join(temp_dir, unique_folder_name)
            os.makedirs(plot_temp_path, exist_ok=True)
        except Exception as e:
            logging.error(f"[XY Plot Advanced] Could not create temp directory: {e}. Aborting.")
            h_fallback = latent_image['samples'].shape[2] * 8 if 'samples' in latent_image else 64
            w_fallback = latent_image['samples'].shape[3] * 8 if 'samples' in latent_image else 64
            return (torch.ones([1, h_fallback, w_fallback, 3], device=comfy.model_management.intermediate_device()) * 0.5,)

        # Calculate max_y_label_width for Y-axis labels
        max_y_label_width = 0 
        y_label_active = y_axis_type != "Nothing"
        if y_label_active:
            temp_font = self._find_font(header_font_size)
            current_max_w = 0
            y_label_padding = 10 
            for y_val_disp in y_values_iter:
                if str(y_val_disp) and str(y_val_disp) != "N/A_Y_internal":
                    text_w, _ = self._measure_text(str(y_val_disp), temp_font, header_font_size)
                    current_max_w = max(current_max_w, text_w)
            if current_max_w > 0: max_y_label_width = int(current_max_w + 2 * y_label_padding)
            elif y_values_iter and any(val != "N/A_Y_internal" and val for val in y_values_iter):
                 max_y_label_width = int(2 * y_label_padding) 
        if not y_label_active: max_y_label_width = 0 # Force 0 if Y-axis is Nothing

        x_header_active = (x_axis_type != "Nothing") or (x_axis_type == "Nothing" and y_axis_type == "Nothing") # X-Header shown if X active OR if both Nothing
        effective_header_height = header_height if x_header_active and header_height > 0 else 0


        raw_image_paths = [[None for _ in range(num_x)] for _ in range(num_y)]
        # Store (H, W) of each raw generated image (after Hires if applicable)
        image_actual_dims = [[(0,0) for _ in range(num_x)] for _ in range(num_y)] 
        params_for_0_0_header_display = {} 

        pbar = comfy.utils.ProgressBar(total_images)
        device = comfy.model_management.get_torch_device()
        intermediate_device = comfy.model_management.intermediate_device()
        
        try:
            # --- Stage 1: Generate all raw images ---
            current_image_idx_overall = 0
            for iy, y_iter_val in enumerate(y_values_iter):
                for ix, x_iter_val in enumerate(x_values_iter):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    current_image_idx_overall += 1

                    current_seed, current_steps, current_cfg = seed, steps, cfg
                    current_sampler, current_scheduler = sampler_name, scheduler
                    current_positive, current_negative = positive, negative 
                    current_latent_dict = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in latent_image.items()}
                    
                    # Apply Y-Axis Parameter 
                    if y_axis_type != "Nothing":
                        if y_axis_type == "Prompt S/R":
                            if placeholder:
                                current_positive = current_positive.replace(placeholder, str(y_iter_val))
                                current_negative = current_negative.replace(placeholder, str(y_iter_val))
                        elif y_axis_type == "Seed":
                            try: current_seed = int(y_iter_val)
                            except ValueError: current_seed = seed
                        # ... (other Y-axis types similarly) ...
                        elif y_axis_type == "Steps":
                            try: current_steps = max(1, int(y_iter_val))
                            except ValueError: current_steps = steps
                        elif y_axis_type == "CFG":
                            try: current_cfg = float(y_iter_val)
                            except ValueError: current_cfg = cfg
                        elif y_axis_type == "Sampler":
                            current_sampler = str(y_iter_val) if str(y_iter_val) in valid_sampler_list else sampler_name
                        elif y_axis_type == "Scheduler":
                            current_scheduler = str(y_iter_val) if str(y_iter_val) in valid_scheduler_list else scheduler

                    # Apply X-Axis Parameter
                    if x_axis_type != "Nothing":
                        if x_axis_type == "Prompt S/R":
                            if placeholder: 
                                current_positive = current_positive.replace(placeholder, str(x_iter_val))
                                current_negative = current_negative.replace(placeholder, str(x_iter_val))
                        elif x_axis_type == "Seed":
                            try: current_seed = int(x_iter_val) 
                            except ValueError: pass # Y value or default retained
                        # ... (other X-axis types similarly) ...
                        elif x_axis_type == "Steps":
                            try: current_steps = max(1, int(x_iter_val))
                            except ValueError: pass
                        elif x_axis_type == "CFG":
                            try: current_cfg = float(x_iter_val)
                            except ValueError: pass
                        elif x_axis_type == "Sampler":
                            if str(x_iter_val) in valid_sampler_list: current_sampler = str(x_iter_val)
                        elif x_axis_type == "Scheduler":
                            if str(x_iter_val) in valid_scheduler_list: current_scheduler = str(x_iter_val)
                    
                    if iy == 0 and ix == 0: # Store params for cell (0,0) if needed for "Nothing/Nothing" header
                        params_for_0_0_header_display = {
                            'seed': current_seed, 'steps': current_steps, 'cfg': current_cfg,
                            'sampler': current_sampler, 'scheduler': current_scheduler
                        }

                    # Generate image (Pass 1)
                    # ... (same ksampler and VAE decode logic as before to get first_pass_image_tensor)
                    try:
                        tokens_positive = clip.tokenize(current_positive); cond_positive = clip.encode_from_tokens_scheduled(tokens_positive)
                        tokens_negative = clip.tokenize(current_negative); cond_negative = clip.encode_from_tokens_scheduled(tokens_negative)
                    except Exception as e_encode:
                        logging.error(f"Text Encoding Error: {e_encode}"); cond_positive=clip.encode_from_tokens_scheduled(clip.tokenize("")); cond_negative=clip.encode_from_tokens_scheduled(clip.tokenize(""))
                    
                    logging.info(f"Generating raw image {current_image_idx_overall}/{total_images}...")
                    latent_out_tuple = nodes.common_ksampler(model, current_seed, current_steps, current_cfg, current_sampler, current_scheduler, cond_positive, cond_negative, current_latent_dict, denoise=1.0, disable_noise=False, force_full_denoise=not enable_hires_fix)
                    first_pass_image_tensor = vae.decode(latent_out_tuple[0]["samples"][:1].to(intermediate_device)).to(device)
                    
                    image_to_be_processed_further = first_pass_image_tensor
                    
                    # Hires Fix (if enabled) - UNCHANGED LOGIC BLOCK
                    if enable_hires_fix and upscaler is not None:
                        # ... (Exact Hires Fix block from your original code, updating image_to_be_processed_further)
                        logging.info(f"  Applying Hires Fix for raw image {current_image_idx_overall}...")
                        upscaled_by_model_bchw, rescaled_image_bchw, rescaled_image_bhwc = None, None, None
                        image_bchw_for_upscale = None
                        try:
                            original_h_pass1 = image_to_be_processed_further.shape[1]; original_w_pass1 = image_to_be_processed_further.shape[2]
                            upscaler.to(device); image_bchw_for_upscale = image_to_be_processed_further.permute(0, 3, 1, 2).to(device)
                            tile = 512; overlap = 32; oom = True
                            while oom:
                                try:
                                    upscaled_by_model_bchw = comfy.utils.tiled_scale(image_bchw_for_upscale, lambda a: upscaler(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscaler.scale, output_device=device)
                                    oom = False
                                except torch.cuda.OutOfMemoryError as e_oom_hires:
                                    tile //= 2; comfy.model_management.soft_empty_cache()
                                    if tile < 64: logging.error("Hires OOM"); raise e_oom_hires
                                    else: logging.warning(f"Hires OOM, tile to {tile}")
                                except Exception as e_scale: logging.error(f"Tiled Scale Error: {e_scale}"); raise e_scale
                            upscaler.to("cpu"); comfy.model_management.soft_empty_cache()
                            target_h = max(8, (int(round(original_h_pass1 * hires_target_upscale_factor)) // 8) * 8)
                            target_w = max(8, (int(round(original_w_pass1 * hires_target_upscale_factor)) // 8) * 8)
                            image_for_rescale = upscaled_by_model_bchw.to(device)
                            if image_for_rescale.shape[2] == target_h and image_for_rescale.shape[3] == target_w: rescaled_image_bchw = image_for_rescale.clone()
                            else:
                                if hires_rescale_method == "bislerp": rescaled_image_bchw = local_bislerp(image_for_rescale, target_w, target_h)
                                elif hires_rescale_method == "lanczos": rescaled_image_bchw = comfy.utils.lanczos(image_for_rescale, target_w, target_h)
                                else: rescaled_image_bchw = torch.nn.functional.interpolate(image_for_rescale, size=(target_h, target_w), mode='bilinear', align_corners=False)
                            rescaled_image_bhwc = torch.clamp(rescaled_image_bchw.permute(0, 2, 3, 1), min=0, max=1.0)
                            hires_latent_dict = {"samples": vae.encode(rescaled_image_bhwc.to(intermediate_device)).to(device)}
                            latent_hires_out_tuple = nodes.common_ksampler(model, current_seed, hires_steps, current_cfg, hires_sampler_name, hires_scheduler, cond_positive, cond_negative, hires_latent_dict, denoise=hires_denoise, disable_noise=False, force_full_denoise=True)
                            image_to_be_processed_further = vae.decode(latent_hires_out_tuple[0]["samples"][:1].to(intermediate_device)).to(device)
                        except Exception as e_hires: logging.error(f"Hires Fix Error: {e_hires}", exc_info=True)
                        finally: # Hires cleanup
                            del image_bchw_for_upscale
                            if upscaled_by_model_bchw is not None: del upscaled_by_model_bchw
                            if rescaled_image_bchw is not None: del rescaled_image_bchw
                            if rescaled_image_bhwc is not None: del rescaled_image_bhwc
                            if 'latent_hires_result_dict' in locals(): del locals()['latent_hires_result_dict']
                            if 'hires_latent_dict' in locals(): del locals()['hires_latent_dict']
                            comfy.model_management.soft_empty_cache()
                    
                    image_actual_dims[iy][ix] = (image_to_be_processed_further.shape[1], image_to_be_processed_further.shape[2])
                    temp_raw_img_path = os.path.join(plot_temp_path, f"raw_img_{iy}_{ix}.pt")
                    torch.save(image_to_be_processed_further.cpu(), temp_raw_img_path)
                    raw_image_paths[iy][ix] = temp_raw_img_path
                    
                    del image_to_be_processed_further, first_pass_image_tensor, latent_out_tuple
                    comfy.model_management.soft_empty_cache()
                    pbar.update(1)

            # --- Stage 2: Create X-Header Row (if active) ---
            final_x_header_row_tensor = None
            if effective_header_height > 0: # Only create if height > 0
                x_header_segments_cpu = []
                if x_axis_type != "Nothing":
                    for ix_h, x_val_h in enumerate(x_values_iter):
                        # Use width of the image in row 0, column ix_h for this header segment
                        segment_w = image_actual_dims[0][ix_h][1] 
                        seg_tensor_cpu = self._create_text_image([str(x_val_h)], segment_w, effective_header_height, header_font_size)
                        x_header_segments_cpu.append(seg_tensor_cpu)
                elif x_axis_type == "Nothing" and y_axis_type == "Nothing" and params_for_0_0_header_display: # Single cell (0,0) full info
                    p = params_for_0_0_header_display
                    info_lines = [f"Seed:{p['seed']}", f"St:{p['steps']},CFG:{p['cfg']:.1f}", f"{p['sampler']}/{p['scheduler']}"]
                    segment_w_00 = image_actual_dims[0][0][1]
                    x_header_segments_cpu.append(self._create_text_image(info_lines, segment_w_00, effective_header_height, header_font_size))
                
                if x_header_segments_cpu:
                    # Ensure all segments are on device before cat
                    x_header_segments_dev = [s.to(device) for s in x_header_segments_cpu]
                    try:
                        final_x_header_row_tensor = torch.cat(x_header_segments_dev, dim=2) # Concatenate horizontally
                    except Exception as e_cat_xh:
                        logging.error(f"[XY Plot Advanced] Error concatenating X-header segments: {e_cat_xh}. X-Header might be missing or incomplete.")
                        final_x_header_row_tensor = None # Fallback
                    del x_header_segments_dev, x_header_segments_cpu


            # --- Stage 3: Create Y-Label Column (if active) ---
            final_y_label_column_tensor = None
            if max_y_label_width > 0: # Only create if width > 0
                y_label_segments_cpu = []
                for iy_l, y_val_l in enumerate(y_values_iter):
                    # Use height of the image in row iy_l, column 0 for this label segment
                    segment_h = image_actual_dims[iy_l][0][0] 
                    seg_tensor_cpu = self._create_vertical_text_image(str(y_val_l), segment_h, header_font_size, max_y_label_width)
                    y_label_segments_cpu.append(seg_tensor_cpu)
                
                if y_label_segments_cpu:
                    y_label_segments_dev = [s.to(device) for s in y_label_segments_cpu]
                    try:
                        final_y_label_column_tensor = torch.cat(y_label_segments_dev, dim=1) # Concatenate vertically
                    except Exception as e_cat_yl:
                        logging.error(f"[XY Plot Advanced] Error concatenating Y-label segments: {e_cat_yl}. Y-Labels might be missing or incomplete.")
                        final_y_label_column_tensor = None # Fallback
                    del y_label_segments_dev, y_label_segments_cpu

            # --- Stage 4: Assemble Core Image Grid ---
            logging.info("[XY Plot Advanced] Assembling core image grid...")
            image_block_tensor = None
            # Concatenation logic (X Rows First or Y Columns First) from original script, applied to raw images
            if concat_direction == "X (Rows First)":
                grid_rows_tensors = []
                for iy_g in range(num_y):
                    current_row_images = [torch.load(raw_image_paths[iy_g][ix_g], map_location=device) for ix_g in range(num_x)]
                    try: grid_rows_tensors.append(torch.cat(current_row_images, dim=2))
                    except Exception as e: logging.error(f"Error cat row {iy_g}: {e}"); image_block_tensor=None; break
                    del current_row_images
                if grid_rows_tensors and image_block_tensor is not False: # image_block_tensor could be None if loop broke
                    try: image_block_tensor = torch.cat(grid_rows_tensors, dim=1)
                    except Exception as e: logging.error(f"Error cat final rows: {e}"); image_block_tensor=None
                del grid_rows_tensors
            else: # "Y (Columns First)"
                grid_cols_tensors = []
                for ix_g in range(num_x):
                    current_col_images = [torch.load(raw_image_paths[iy_g][ix_g], map_location=device) for iy_g in range(num_y)]
                    try: grid_cols_tensors.append(torch.cat(current_col_images, dim=1))
                    except Exception as e: logging.error(f"Error cat col {ix_g}: {e}"); image_block_tensor=None; break
                    del current_col_images
                if grid_cols_tensors and image_block_tensor is not False:
                    try: image_block_tensor = torch.cat(grid_cols_tensors, dim=2)
                    except Exception as e: logging.error(f"Error cat final cols: {e}"); image_block_tensor=None
                del grid_cols_tensors
            
            if image_block_tensor is None: # Fallback if grid assembly failed
                logging.error("[XY Plot Advanced] Failed to assemble core image grid. Returning error placeholder.")
                h_fb = latent_image['samples'].shape[2]*8; w_fb = latent_image['samples'].shape[3]*8
                return (torch.ones([1, h_fb, w_fb, 3], device=intermediate_device) * 0.3,)


            # --- Stage 5: Combine Image Grid with Headers ---
            final_image_assembly = image_block_tensor
            
            # Add X-Header row if it exists
            if final_x_header_row_tensor is not None:
                try:
                    final_image_assembly = torch.cat((final_x_header_row_tensor, final_image_assembly), dim=1)
                except Exception as e_cat_xh_final:
                    logging.error(f"[XY Plot Advanced] Error attaching X-header row: {e_cat_xh_final}")


            # Add Y-Label column if it exists (and potentially a top-left blank corner)
            if final_y_label_column_tensor is not None:
                left_column_to_add = final_y_label_column_tensor
                if effective_header_height > 0: # If X-headers existed, Y-labels need a blank space above them
                    blank_corner_h = effective_header_height
                    blank_corner_w = max_y_label_width
                    top_left_blank_tensor = torch.ones([1, blank_corner_h, blank_corner_w, 3], 
                                                       dtype=final_y_label_column_tensor.dtype, device=device) # White
                    try:
                        left_column_to_add = torch.cat((top_left_blank_tensor, final_y_label_column_tensor), dim=1)
                    except Exception as e_cat_yl_corner:
                         logging.error(f"[XY Plot Advanced] Error creating full Y-label column with corner: {e_cat_yl_corner}")
                         # Proceed with final_y_label_column_tensor as is, might cause misalignment if X-header exists

                try:
                    final_image_assembly = torch.cat((left_column_to_add, final_image_assembly), dim=2)
                except Exception as e_cat_yl_final:
                    logging.error(f"[XY Plot Advanced] Error attaching Y-label column: {e_cat_yl_final}")
            
            logging.info("[XY Plot Advanced] Final grid assembly finished.")
            return (final_image_assembly.to(intermediate_device),)

        finally:
             if plot_temp_path and os.path.exists(plot_temp_path):
                 try:
                     shutil.rmtree(plot_temp_path)
                     logging.info(f"[XY Plot Advanced] Cleaned up temp folder: {plot_temp_path}")
                 except Exception as e_cleanup:
                     logging.error(f"[XY Plot Advanced] Failed to clean up temp folder '{plot_temp_path}': {e_cleanup}.")