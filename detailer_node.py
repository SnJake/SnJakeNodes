import torch
import numpy as np
import math
import comfy.utils
import comfy.samplers
import comfy.sample
import nodes
import latent_preview
from PIL import Image
import torchvision.transforms.functional as F
from scipy.ndimage import gaussian_filter, label, find_objects, grey_dilation, binary_closing, binary_fill_holes

# region: --- Helper functions from InpaintStitchImproved ---
# Мы импортируем все необходимые и хорошо написанные вспомогательные функции.
# Это лучший подход, чем писать свои реализации.

def rescale_i(samples, width, height, algorithm: str):
    """Rescales an image tensor (BHWC)."""
    samples = samples.movedim(-1, 1)
    # getattr(Image, algorithm.upper()) -> Image.BICUBIC etc.
    rescale_pil_algorithm = getattr(Image, algorithm.upper())
    pil_img = F.to_pil_image(samples[0].cpu())
    rescaled_pil = pil_img.resize((width, height), rescale_pil_algorithm)
    rescaled_tensor = F.to_tensor(rescaled_pil).unsqueeze(0)
    return rescaled_tensor.movedim(1, -1)

def rescale_m(samples, width, height, algorithm: str):
    """Rescales a mask tensor (BHW)."""
    samples = samples.unsqueeze(1)
    rescale_pil_algorithm = getattr(Image, algorithm.upper())
    pil_img = F.to_pil_image(samples[0].cpu())
    rescaled_pil = pil_img.resize((width, height), rescale_pil_algorithm)
    rescaled_tensor = F.to_tensor(rescaled_pil).unsqueeze(0)
    return rescaled_tensor.squeeze(1)

def blur_m(samples, pixels):
    """Blurs a mask tensor."""
    if pixels == 0:
        return samples
    mask = samples.squeeze(0)
    sigma = pixels / 4.0
    mask_np = mask.cpu().numpy()
    blurred_mask = gaussian_filter(mask_np, sigma=sigma)
    blurred_mask = torch.from_numpy(blurred_mask).float()
    blurred_mask = torch.clamp(blurred_mask, 0.0, 1.0)
    return blurred_mask.unsqueeze(0)

def expand_m(mask, pixels):
    """Expands a mask tensor using grey dilation."""
    if pixels == 0:
        return mask
    sigma = pixels / 4.0
    mask_np = mask.squeeze(0).cpu().numpy()
    kernel_size = math.ceil(sigma * 1.5 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated_mask = grey_dilation(mask_np, footprint=kernel)
    dilated_mask = torch.clamp(torch.from_numpy(dilated_mask.astype(np.float32)), 0.0, 1.0)
    return dilated_mask.unsqueeze(0)
    
def pad_to_multiple(value, multiple):
    """Calculates the padded value to be a multiple of a number."""
    return int(math.ceil(value / multiple) * multiple)

def crop_magic_im(image, mask, x, y, w, h, target_w, target_h, padding, downscale_algorithm, upscale_algorithm):
    """
    Core cropping logic. Creates a canvas, adjusts for aspect ratio,
    and returns the cropped sections and coordinates.
    """
    image = image.clone()
    mask = mask.clone()

    if target_w <= 0 or target_h <= 0 or w <= 0 or h <= 0:
        return image, 0, 0, image.shape[2], image.shape[1], image, mask, 0, 0, image.shape[2], image.shape[1]

    if padding > 1:
        target_w = pad_to_multiple(target_w, padding)
        target_h = pad_to_multiple(target_h, padding)

    target_aspect_ratio = target_w / target_h
    B, image_h, image_w, C = image.shape
    context_aspect_ratio = w / h

    new_x, new_y, new_w, new_h = x, y, w, h

    if context_aspect_ratio < target_aspect_ratio:
        new_w = int(h * target_aspect_ratio)
        new_x = x - (new_w - w) // 2
    else:
        new_h = int(w / target_aspect_ratio)
        new_y = y - (new_h - h) // 2
    
    # Simple boundary clamp for Detailer's purpose
    # The original has more complex logic to shift the box if it overflows,
    # but for Detailer, we just expand and let it create a canvas.
    
    up_padding = max(0, -new_y)
    down_padding = max(0, (new_y + new_h) - image_h)
    left_padding = max(0, -new_x)
    right_padding = max(0, (new_x + new_w) - image_w)

    expanded_image_h = image_h + up_padding + down_padding
    expanded_image_w = image_w + left_padding + right_padding
    
    canvas_image = torch.zeros((B, expanded_image_h, expanded_image_w, C), device=image.device)
    canvas_mask = torch.ones((B, expanded_image_h, expanded_image_w), device=mask.device)

    img_bchw = image.movedim(-1, 1)
    canvas_bchw = canvas_image.movedim(-1, 1)

    canvas_bchw[:, :, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = img_bchw

    # Edge pixel padding
    if up_padding > 0:
        canvas_bchw[:, :, :up_padding, left_padding:left_padding + image_w] = img_bchw[:, :, 0:1, :]
    if down_padding > 0:
        canvas_bchw[:, :, -down_padding:, left_padding:left_padding + image_w] = img_bchw[:, :, -1:, :]
    if left_padding > 0:
        canvas_bchw[:, :, :, :left_padding] = canvas_bchw[:, :, :, left_padding:left_padding+1]
    if right_padding > 0:
        canvas_bchw[:, :, :, -right_padding:] = canvas_bchw[:, :, :, -right_padding-1:-right_padding]

    canvas_image = canvas_bchw.movedim(1, -1)
    canvas_mask[:, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = mask

    cto_x, cto_y, cto_w, cto_h = left_padding, up_padding, image_w, image_h
    ctc_x, ctc_y, ctc_w, ctc_h = new_x + left_padding, new_y + up_padding, new_w, new_h

    cropped_image = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
    cropped_mask = canvas_mask[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
    
    rescale_algo = upscale_algorithm if target_w > ctc_w or target_h > ctc_h else downscale_algorithm
    
    final_cropped_image = rescale_i(cropped_image, target_w, target_h, rescale_algo)
    final_cropped_mask = rescale_m(cropped_mask, target_w, target_h, "nearest") # mask is always nearest

    return canvas_image, cto_x, cto_y, cto_w, cto_h, final_cropped_image, final_cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h

def stitch_magic_im(canvas_image, inpainted_image, mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm):
    """
    Core stitching logic. Resizes the inpainted result, blends it onto
    the canvas, and crops back to original dimensions.
    """
    canvas_image = canvas_image.clone()
    
    rescale_algo = upscale_algorithm if ctc_w > inpainted_image.shape[2] or ctc_h > inpainted_image.shape[1] else downscale_algorithm

    resized_image = rescale_i(inpainted_image, ctc_w, ctc_h, rescale_algo)
    # Mask should be rescaled with a smooth algorithm for good blending
    resized_mask = rescale_m(mask, ctc_w, ctc_h, "bicubic")

    resized_mask = resized_mask.clamp(0, 1).unsqueeze(-1)
    canvas_crop = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

    blended = resized_mask * resized_image + (1.0 - resized_mask) * canvas_crop
    canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w] = blended

    output_image = canvas_image[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w]
    return output_image

# endregion

class DetailerForEachMask:
    """
    Эта нода последовательно детализирует области на изображении, указанные масками.
    Она перебирает каждую маску, вырезает соответствующую область с контекстом,
    применяет семплер для детализации, а затем вшивает результат обратно.
    Идеально подходит для улучшения лиц, объектов или других деталей с помощью BBOX или сегментационных масок.
    РЕФАКТОРИНГ: Использует улучшенную и более надежную логику кропа/сшивки из ноды InpaintStitchImproved.
    """
    
    upscale_methods = ["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "image": ("IMAGE", {"tooltip": "Изображение (холст), на котором будет производиться детализация."}),
                "masks": ("MASK", {"tooltip": "Маска с одной или несколькими областями (например, от BBOX) для детализации."}),

                # Sampler settings
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Начальное случайное зерно для шума. Будет увеличиваться на 1 для каждой следующей маски."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Количество шагов семплирования для каждой области."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "tooltip": "Сила влияния промпта на результат (Classifier-Free Guidance)."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Алгоритм семплера, который будет использоваться для детализации."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Планировщик шагов для семплера."}),
                "denoise": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Сила обесшумливания. 1.0 — полное изменение области, <1.0 — сохранение части исходной структуры."}),

                # Crop & Stitch settings
                "context_expand_pixels": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": "На сколько пикселей расширить область контекста вокруг каждой маски."}),
                "blur_mask_pixels": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 64.0, "step": 0.1, "tooltip": "Размытие маски для инпейнтинга (до VAE). Помогает создать более плавные края в латенте."}),
                "blend_pixels": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 64.0, "step": 0.1, "tooltip": "Радиус размытия для плавного смешивания обработанного участка с исходным изображением."}),
                "grow_mask_by": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1, "tooltip": "На сколько пикселей расширить маску в латентном пространстве для предотвращения жестких краев."}),
                
                # Rescale settings
                "force_width": ("INT", {"default": 512, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8, "tooltip": "Принудительная ширина области для семплирования. 0 = авто."}),
                "force_height": ("INT", {"default": 512, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8, "tooltip": "Принудительная высота области для семплирования. 0 = авто."}),
                "downscale_algorithm": (cls.upscale_methods, {"default": "bilinear", "tooltip": "Алгоритм для уменьшения масштаба."}),
                "upscale_algorithm": (cls.upscale_methods, {"default": "bicubic", "tooltip": "Алгоритм для увеличения масштаба."}),
                "padding": ([8, 16, 32, 64, 128, 256], {"default": 32, "tooltip": "Выравнивание размера вырезанной области. Ее ширина и высота будут кратны этому значению."}),

                # Mask processing order
                "mask_process_order": (["сверху-вниз", "снизу-вверх", "слева-направо", "справа-налево", "от большей к меньшей", "от меньшей к большей", "случайно"],
                                       {"default": "сверху-вниз", "tooltip": "Порядок, в котором будут обрабатываться маски, если их несколько."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "MASK")
    RETURN_NAMES = ("image", "latent", "processed_masks")
    FUNCTION = "detail_sequentially"
    CATEGORY = "😎 SnJake/Detailer"

    def detail_sequentially(self, model, positive, negative, vae, image, masks,
                            noise_seed, steps, cfg, sampler_name, scheduler, denoise,
                            context_expand_pixels, blur_mask_pixels, blend_pixels, grow_mask_by,
                            force_width, force_height, downscale_algorithm, upscale_algorithm, padding,
                            mask_process_order):
        
        if masks.numel() == 0 or masks.max() == 0:
            print("Маски не найдены или пусты. Возвращается исходное изображение.")
            # Исправлено: VAE ожидает BCHW, а не BHWC
            final_latent_tensor = vae.encode(image[:,:,:,:3].movedim(-1, 1))
            return (image, {"samples": final_latent_tensor}, torch.zeros_like(masks))

        mask_np = masks.cpu().numpy().squeeze()
        labeled_array, num_features = label(mask_np > 0.5)
        if num_features == 0:
            # Исправлено: VAE ожидает BCHW, а не BHWC
            final_latent_tensor = vae.encode(image[:,:,:,:3].movedim(-1, 1))
            return (image, {"samples": final_latent_tensor}, torch.zeros_like(masks))
            
        found_objects = find_objects(labeled_array)
        decorated_masks = [{'slice': slc, 'label': i + 1, 'area': np.sum(labeled_array[slc] == i + 1),
                            'center_y': slc[0].start + (slc[0].stop - slc[0].start) / 2,
                            'center_x': slc[1].start + (slc[1].stop - slc[1].start) / 2}
                           for i, slc in enumerate(found_objects) if slc is not None]

        sort_key_map = {"слева-направо": "center_x", "справа-налево": "center_x", "сверху-вниз": "center_y", "снизу-вверх": "center_y", "от большей к меньшей": "area", "от меньшей к большей": "area"}
        if mask_process_order in sort_key_map:
            decorated_masks.sort(key=lambda m: m[sort_key_map[mask_process_order]], reverse=(mask_process_order in {"справа-налево", "снизу-вверх", "от большей к меньшей"}))
        elif mask_process_order == "случайно":
            import random
            random.shuffle(decorated_masks)

        image_to_process = image.clone()
        original_height, original_width = image.shape[1], image.shape[2]
        final_processed_mask = torch.zeros((1, original_height, original_width), device=image.device)
        pbar = comfy.utils.ProgressBar(num_features)

        for i, mask_info in enumerate(decorated_masks):
            print(f"Обработка маски {i+1}/{num_features}...")
            
            current_mask_np = (labeled_array == mask_info['label']).astype(np.float32)
            current_mask_tensor = torch.from_numpy(current_mask_np).to(image.device).unsqueeze(0)
            
            if blur_mask_pixels > 0:
                inpaint_mask = blur_m(current_mask_tensor, blur_mask_pixels)
            else:
                inpaint_mask = current_mask_tensor

            slc = mask_info['slice']
            y_min, y_max = slc[0].start, slc[0].stop
            x_min, x_max = slc[1].start, slc[1].stop

            x = x_min - context_expand_pixels
            y = y_min - context_expand_pixels
            w = (x_max - x_min) + 2 * context_expand_pixels
            h = (y_max - y_min) + 2 * context_expand_pixels

            target_w = force_width if force_width > 0 else w
            target_h = force_height if force_height > 0 else h

            canvas_image, cto_x, cto_y, cto_w, cto_h, \
            cropped_image, cropped_mask, \
            ctc_x, ctc_y, ctc_w, ctc_h = crop_magic_im(
                image_to_process, inpaint_mask, x, y, w, h,
                target_w, target_h, padding, downscale_algorithm, upscale_algorithm
            )
            
            if cropped_image.shape[1] == 0 or cropped_image.shape[2] == 0: continue

            latent_mask_for_inpaint = cropped_mask.clone()
            pixels_for_concat = cropped_image * (1.0 - latent_mask_for_inpaint.round().unsqueeze(-1))
            
            # ИСПРАВЛЕНО: vae.encode() возвращает BCHW, второй movedim не нужен
            concat_latent = vae.encode(pixels_for_concat.movedim(-1, 1))
            initial_latent_samples = vae.encode(cropped_image.movedim(-1, 1))

            latent_for_sampler = {"samples": initial_latent_samples}
            mask_for_sampler = latent_mask_for_inpaint.reshape((-1, 1, latent_mask_for_inpaint.shape[-2], latent_mask_for_inpaint.shape[-1]))
            
            if grow_mask_by > 0:
                grown_mask = expand_m(mask_for_sampler.squeeze(0), grow_mask_by*2).unsqueeze(0)
            else:
                grown_mask = mask_for_sampler

            latent_h, latent_w = initial_latent_samples.shape[2], initial_latent_samples.shape[3]
            latent_for_sampler["noise_mask"] = torch.nn.functional.interpolate(grown_mask, size=(latent_h, latent_w), mode="bilinear")

            positive_inpaint = [[c[0], {**c[1], 'concat_latent_image': concat_latent, 'concat_mask': mask_for_sampler}] for c in positive]
            negative_inpaint = [[c[0], {**c[1], 'concat_latent_image': concat_latent, 'concat_mask': mask_for_sampler}] for c in negative]
            
            latent_out = nodes.common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive_inpaint, negative_inpaint, latent_for_sampler, denoise=denoise)[0]
            noise_seed += 1
            
            # ИСПРАВЛЕНО: latent_out["samples"] уже BCHW. vae.decode() возвращает BCHW, который надо перевести в BHWC.
            decoded_crop = vae.decode(latent_out["samples"]).movedim(1, -1)

            blend_mask = blur_m(cropped_mask, blend_pixels)
            
            image_to_process = stitch_magic_im(
                canvas_image, decoded_crop, blend_mask,
                ctc_x, ctc_y, ctc_w, ctc_h,
                cto_x, cto_y, cto_w, cto_h,
                downscale_algorithm, upscale_algorithm
            )
            
            temp_mask_canvas = torch.zeros_like(canvas_image[:,:,:,0])
            blend_mask_resized = rescale_m(blend_mask, ctc_w, ctc_h, "bicubic")
            temp_mask_canvas[:, ctc_y:ctc_y+ctc_h, ctc_x:ctc_x+ctc_w] = blend_mask_resized
            processed_part_mask = temp_mask_canvas[:, cto_y:cto_y+cto_h, cto_x:cto_x+cto_w]
            
            final_processed_mask = torch.max(final_processed_mask, processed_part_mask)
            pbar.update(1)
        
        final_processed_mask.clamp_(0.0, 1.0)
        
        # ИСПРАВЛЕНО: VAE ожидает BCHW. Результат нужно обернуть в словарь.
        final_latent_tensor = vae.encode(image_to_process[:,:,:,:3].movedim(-1, 1))
        
        return (image_to_process, {"samples": final_latent_tensor}, final_processed_mask.squeeze(0))
