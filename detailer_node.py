import torch
import numpy as np
import math
import comfy.utils
import comfy.samplers
import comfy.sample
import nodes
from PIL import Image
import torchvision.transforms.functional as F
from scipy.ndimage import gaussian_filter, label, find_objects, grey_dilation, binary_closing, binary_fill_holes

# ==================================================================================
# == Вспомогательные функции, перенесенные из InpaintStitchImproved ==
# ==================================================================================

def rescale_i(samples, width, height, algorithm: str):
    """Rescales an image tensor."""
    samples = samples.movedim(-1, 1)
    rescale_pil_algorithm = getattr(Image, algorithm.upper())
    rescaled_tensors = []
    for sample in samples:
        pil_img = F.to_pil_image(sample.cpu())
        rescaled_pil = pil_img.resize((width, height), rescale_pil_algorithm)
        rescaled_tensors.append(F.to_tensor(rescaled_pil))
    
    output = torch.stack(rescaled_tensors).to(samples.device)
    return output.movedim(1, -1)

def rescale_m(samples, width, height, algorithm: str):
    """Rescales a mask tensor, now correctly handling batches."""
    samples = samples.unsqueeze(1)
    rescale_pil_algorithm = getattr(Image, algorithm.upper())
    rescaled_tensors = []
    for sample in samples:
        pil_img = F.to_pil_image(sample.cpu())
        rescaled_pil = pil_img.resize((width, height), rescale_pil_algorithm)
        rescaled_tensors.append(F.to_tensor(rescaled_pil))
        
    output = torch.stack(rescaled_tensors).to(samples.device)
    return output.squeeze(1)

def hipassfilter_m(samples, threshold):
    """Filters mask values lower than a threshold."""
    filtered_mask = samples.clone()
    filtered_mask[filtered_mask < threshold] = 0
    return filtered_mask

def expand_m(mask, pixels):
    """Expands a mask by a given number of pixels using grey dilation."""
    if pixels == 0:
        return mask
    sigma = pixels / 4
    mask_np = mask.squeeze(0).cpu().numpy()
    kernel_size = math.ceil(sigma * 1.5 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated_mask = grey_dilation(mask_np, footprint=kernel)
    dilated_mask = dilated_mask.astype(np.float32)
    dilated_mask = torch.from_numpy(dilated_mask).to(mask.device)
    dilated_mask = torch.clamp(dilated_mask, 0.0, 1.0)
    return dilated_mask.unsqueeze(0)

def blur_m(samples, pixels):
    """Blurs a mask using a Gaussian filter."""
    if pixels == 0:
        return samples
    mask = samples.squeeze(0)
    sigma = pixels / 4 
    mask_np = mask.cpu().numpy()
    blurred_mask = gaussian_filter(mask_np, sigma=sigma)
    blurred_mask = torch.from_numpy(blurred_mask).float().to(samples.device)
    blurred_mask = torch.clamp(blurred_mask, 0.0, 1.0)
    return blurred_mask.unsqueeze(0)

def findcontextarea_m(mask):
    """Finds the bounding box of the non-zero area in a mask."""
    mask_squeezed = mask[0]
    non_zero_indices = torch.nonzero(mask_squeezed)

    if non_zero_indices.numel() == 0:
        return None, -1, -1, -1, -1

    y_min, x_min = torch.min(non_zero_indices, dim=0).values.tolist()
    y_max, x_max = torch.max(non_zero_indices, dim=0).values.tolist()
    
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    
    context = mask[:, y_min:y_min+h, x_min:x_min+w]
    return context, x_min, y_min, w, h

def pad_to_multiple(value, multiple):
    """Pads a value to be a multiple of another value."""
    return int(math.ceil(value / multiple) * multiple)

# --- ИСПРАВЛЕННАЯ ВЕРСИЯ CROP_MAGIC_IM ---
def crop_magic_im(image, mask, x, y, w, h, target_w, target_h, padding, downscale_algorithm, upscale_algorithm):
    """
    Core cropping function. Determines the right context area, grows the image canvas if needed,
    and crops/resizes the area to the target dimensions.
    """
    image = image.clone()
    mask = mask.clone()

    if target_w <= 0 or target_h <= 0 or w <= 0 or h <= 0:
        return image, 0, 0, image.shape[2], image.shape[1], image, mask, 0, 0, image.shape[2], image.shape[1]

    # Step 1: Pad target dimensions to be multiples of padding
    if padding > 1:
        target_w = pad_to_multiple(target_w, padding)
        target_h = pad_to_multiple(target_h, padding)

    # Step 2: Calculate target aspect ratio and grow context to match
    target_aspect_ratio = target_w / target_h
    B, image_h, image_w, C = image.shape
    context_aspect_ratio = w / h

    if context_aspect_ratio < target_aspect_ratio:
        # Grow width
        new_w = int(h * target_aspect_ratio)
        new_h = h
        new_x = x - (new_w - w) // 2
        new_y = y
    else:
        # Grow height
        new_w = w
        new_h = int(w / target_aspect_ratio)
        new_x = x
        new_y = y - (new_h - h) // 2

    # Step 3: Grow the image canvas to accommodate the new context area if it overflows
    up_padding, down_padding, left_padding, right_padding = 0, 0, 0, 0
    if new_x < 0:
        left_padding = -new_x
    if new_y < 0:
        up_padding = -new_y
    if new_x + new_w > image_w:
        right_padding = (new_x + new_w) - image_w
    if new_y + new_h > image_h:
        down_padding = (new_y + new_h) - image_h
    
    expanded_image_w = image_w + left_padding + right_padding
    expanded_image_h = image_h + up_padding + down_padding

    canvas_image = torch.zeros((B, expanded_image_h, expanded_image_w, C), device=image.device)
    canvas_mask = torch.ones((B, expanded_image_h, expanded_image_w), device=mask.device)

    # Place original image and mask onto the new canvas
    canvas_image[:, up_padding:up_padding + image_h, left_padding:left_padding + image_w, :] = image
    canvas_mask[:, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = mask

    # Step 4: Fill the new extended areas with replicated edge pixels
    if up_padding > 0: canvas_image[:, :up_padding, :, :] = canvas_image[:, up_padding:up_padding+1, :, :].repeat(1, up_padding, 1, 1)
    if down_padding > 0: canvas_image[:, -down_padding:, :, :] = canvas_image[:, -down_padding-1:-down_padding, :, :].repeat(1, down_padding, 1, 1)
    if left_padding > 0: canvas_image[:, :, :left_padding, :] = canvas_image[:, :, left_padding:left_padding+1, :].repeat(1, 1, left_padding, 1)
    if right_padding > 0: canvas_image[:, :, -right_padding:, :] = canvas_image[:, :, -right_padding:, :].clone()[:, :, -right_padding-1:-right_padding, :].repeat(1, 1, right_padding, 1)

    # Step 5: Define coordinate systems
    # cto: canvas to original
    cto_x, cto_y, cto_w, cto_h = left_padding, up_padding, image_w, image_h
    # ctc: cropped to canvas
    ctc_x, ctc_y, ctc_w, ctc_h = new_x + left_padding, new_y + up_padding, new_w, new_h

    # Step 6: Crop the context area from the canvas
    cropped_image = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
    cropped_mask = canvas_mask[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

    # Step 7: Resize cropped area to the final target size
    rescale_algorithm = upscale_algorithm if target_w > ctc_w or target_h > ctc_h else downscale_algorithm
    
    if cropped_image.shape[1] > 0 and cropped_image.shape[2] > 0:
        cropped_image = rescale_i(cropped_image, target_w, target_h, rescale_algorithm)
        cropped_mask = rescale_m(cropped_mask, target_w, target_h, "nearest")
    else:
        cropped_image = torch.zeros((B, target_h, target_w, C), device=image.device)
        cropped_mask = torch.zeros((B, target_h, target_w), device=mask.device)

    return canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h
# --- КОНЕЦ ИСПРАВЛЕННОЙ ВЕРСИИ ---

def stitch_magic_im(canvas_image, inpainted_image, mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm):
    """
    Core stitching function. Resizes the inpainted result, blends it into the canvas, and
    crops the canvas back to the original image dimensions.
    """
    canvas_image = canvas_image.clone()
    inpainted_image = inpainted_image.clone()
    mask = mask.clone()

    rescale_algorithm = upscale_algorithm if ctc_w > inpainted_image.shape[2] or ctc_h > inpainted_image.shape[1] else downscale_algorithm
    resized_image = rescale_i(inpainted_image, ctc_w, ctc_h, rescale_algorithm)
    resized_mask = rescale_m(mask, ctc_w, ctc_h, "bilinear") # bilinear for smooth blend

    resized_mask = resized_mask.clamp(0, 1).unsqueeze(-1)
    
    canvas_crop = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
    
    blended = resized_mask * resized_image + (1.0 - resized_mask) * canvas_crop
    canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w] = blended

    output_image = canvas_image[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w]
    return output_image


# ==================================================================================
# ==                         ОБНОВЛЕННАЯ НОДА DETAILER                            ==
# ==================================================================================

class DetailerForEachMask:
    """
    Эта нода последовательно детализирует области на изображении, указанные масками.
    Использует улучшенный пайплайн кропа и сшивания для более надежной и гибкой обработки.
    Она перебирает каждую маску, вырезает соответствующую область с контекстом,
    применяет семплер для детализации, а затем вшивает результат обратно.
    """
    
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

                # Crop & Stitch settings (from InpaintStitchImproved)
                "mask_expand_pixels": ("INT", {"default": 32, "min": 0, "max": 512, "step": 1, "tooltip": "Расширить каждую маску на указанное количество пикселей для создания контекста."}),
                "mask_blend_pixels": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "Размытие краев итоговой маски для плавного смешивания."}),
                "mask_hipass_filter": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.01, "tooltip": "Игнорировать значения в маске ниже этого порога. 0 = выключено."}),
                
                # Rescale settings
                "force_width": ("INT", {"default": 512, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8, "tooltip": "Принудительная ширина области для семплирования. 0 = авто."}),
                "force_height": ("INT", {"default": 512, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8, "tooltip": "Принудительная высота области для семплирования. 0 = авто."}),
                "downscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos", "area"], {"default": "bilinear"}),
                "upscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos"], {"default": "bicubic"}),
                "padding": ([8, 16, 32, 64, 128, 256], {"default": 32, "tooltip": "Выравнивание размера вырезанной области. Ее ширина и высота будут кратны этому значению."}),

                # Mask processing order
                "mask_process_order": (["сверху-вниз", "снизу-вверх", "слева-направо", "справа-налево", "от большей к меньшей", "от меньшей к большей", "случайно"],
                                       {"default": "сверху-вниз", "tooltip": "Порядок, в котором будут обрабатываться маски, если их несколько."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "MASK")
    RETURN_NAMES = ("image", "latent", "processed_masks")
    OUTPUT_TOOLTIPS = (
        "Детализированное изображение.",
        "Латент финального изображения.",
        "Комбинированная маска всех обработанных областей с учетом размытия для смешивания."
    )
    FUNCTION = "detail_sequentially"
    CATEGORY = "😎 SnJake/Detailer"


    def detail_sequentially(self, model, positive, negative, vae, image, masks,
                            noise_seed, steps, cfg, sampler_name, scheduler, denoise,
                            mask_expand_pixels, mask_blend_pixels, mask_hipass_filter,
                            force_width, force_height, downscale_algorithm, upscale_algorithm, padding,
                            mask_process_order):
        
        if masks.numel() == 0 or masks.max() == 0:
            print("DetailerForEachPipe: Маски не найдены или пусты.")
            latent = vae.encode(image[:,:,:,:3])
            return (image, {"samples": latent}, torch.zeros_like(masks[:,:,:]))

        mask_np = masks.cpu().numpy().squeeze()
        labeled_array, num_features = label(mask_np > 0.5)
        if num_features == 0: 
            print("DetailerForEachPipe: Не найдено отдельных областей в масках.")
            return (image, {"samples": vae.encode(image[:,:,:,:3])}, torch.zeros_like(masks[:,:,:]))

        found_objects = find_objects(labeled_array)
        decorated_masks = []
        for i, slc in enumerate(found_objects):
            if slc is None: continue
            mask_label = i + 1; coords = np.argwhere(labeled_array[slc] == mask_label)
            if coords.size == 0: continue
            center_y, center_x = np.mean(coords, axis=0); center_y += slc[0].start; center_x += slc[1].start
            decorated_masks.append({'slice': slc, 'center_x': center_x, 'center_y': center_y, 'area': len(coords), 'label': mask_label})

        sort_key_map = {"слева-направо": "center_x", "справа-налево": "center_x", "сверху-вниз": "center_y", "снизу-вверх": "center_y", "от большей к меньшей": "area", "от меньшей к большей": "area"}
        if mask_process_order in sort_key_map:
            decorated_masks.sort(key=lambda m: m[sort_key_map[mask_process_order]], reverse=(mask_process_order in {"справа-налево", "снизу-вверх", "от большей к меньшей"}))
        elif mask_process_order == "случайно":
            import random; random.shuffle(decorated_masks)

        image_to_process = image.clone()
        original_height, original_width = image.shape[1], image.shape[2]
        final_processed_mask = torch.zeros((1, original_height, original_width), device=image.device)
        pbar = comfy.utils.ProgressBar(num_features)

        for i, mask_info in enumerate(decorated_masks):
            print(f"DetailerForEachPipe: Обработка маски {i+1}/{num_features}...")
            
            # 1. Разделяем маски: одна для формы объекта, вторая для контекста
            individual_mask = torch.from_numpy((labeled_array == mask_info['label']).astype(np.float32)).to(image.device).unsqueeze(0)
            if mask_hipass_filter > 0.0:
                individual_mask = hipassfilter_m(individual_mask, mask_hipass_filter)
            
            context_mask = expand_m(individual_mask, mask_expand_pixels)

            # 2. Находим bbox для кропа из РАСШИРЕННОЙ маски (контекста)
            _, x, y, w, h = findcontextarea_m(context_mask)
            if x == -1: continue

            # 3. Выполняем кроп. `cropped_mask` будет содержать форму контекста для VAE.
            target_w = force_width if force_width > 0 else w
            target_h = force_height if force_height > 0 else h
            (canvas_image, cto_x, cto_y, cto_w, cto_h, 
             cropped_image, cropped_context_mask, 
             ctc_x, ctc_y, ctc_w, ctc_h) = crop_magic_im(
                image_to_process, context_mask, x, y, w, h, target_w, target_h, 
                padding, downscale_algorithm, upscale_algorithm
             )
            
            if cropped_image.shape[1] == 0 or cropped_image.shape[2] == 0: continue

            # --- НАЧАЛО ИСПРАВЛЕНИЯ ---
            # 4. Создаем ПРАВИЛЬНУЮ маску для смешивания. Она должна иметь форму ИСХОДНОГО объекта, а не контекста.
            # Создаем пустой холст с такими же размерами, как и `canvas_image`.
            object_shape_canvas = torch.zeros((1, canvas_image.shape[1], canvas_image.shape[2]), device=image.device)
            # Размещаем на нем маску исходного объекта (individual_mask) со смещением `cto_x`, `cto_y`.
            object_shape_canvas[:, cto_y:cto_y+original_height, cto_x:cto_x+original_width] = individual_mask
            # Вырезаем из этого холста область по тем же координатам `ctc...`, что и основной кроп.
            cropped_object_mask = object_shape_canvas[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
            # Размываем именно эту, правильную по форме, маску.
            blending_mask = blur_m(cropped_object_mask, mask_blend_pixels)
            # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
            
            # 5. Подготовка к семплированию (используем `cropped_context_mask` для VAE)
            pixels_for_concat = cropped_image.clone()
            m = (1.0 - cropped_context_mask.round()).unsqueeze(-1)
            pixels_for_concat = (pixels_for_concat - 0.5) * m + 0.5
            
            concat_latent = vae.encode(pixels_for_concat)
            initial_latent_samples = vae.encode(cropped_image)
            latent_for_sampler = {"samples": initial_latent_samples}
            mask_for_sampler = cropped_context_mask.reshape((-1, 1, cropped_context_mask.shape[-2], cropped_context_mask.shape[-1]))
            latent_h, latent_w = initial_latent_samples.shape[2], initial_latent_samples.shape[3]
            latent_for_sampler["noise_mask"] = torch.nn.functional.interpolate(mask_for_sampler, size=(latent_h, latent_w), mode="bilinear").squeeze(1)
            
            def create_inpaint_cond(cond_list): return [[c[0], {**c[1], 'concat_latent_image': concat_latent, 'concat_mask': mask_for_sampler}] for c in cond_list]
            positive_inpaint, negative_inpaint = create_inpaint_cond(positive), create_inpaint_cond(negative)
            
            # 6. Семплирование
            latent_out = nodes.common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive_inpaint, negative_inpaint, latent_for_sampler, denoise=denoise)
            noise_seed += 1

            # 7. Декодирование и сшивание с использованием ИСПРАВЛЕННОЙ `blending_mask`
            decoded_crop = vae.decode(latent_out[0]["samples"])
            image_to_process = stitch_magic_im(
                canvas_image, decoded_crop, blending_mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm
            )
            
            # Аккумулируем итоговую маску обработанных областей для вывода
            temp_canvas_mask = torch.zeros_like(canvas_image[:,:,:,0])
            blending_mask_resized = rescale_m(blending_mask, ctc_w, ctc_h, "bilinear")
            temp_canvas_mask[:, ctc_y:ctc_y+ctc_h, ctc_x:ctc_x+ctc_w] = blending_mask_resized
            final_processed_mask += temp_canvas_mask[:, cto_y:cto_y+cto_h, cto_x:cto_x+cto_w]
            pbar.update(1)
        
        final_processed_mask.clamp_(0.0, 1.0)
        final_latent = vae.encode(image_to_process[:,:,:,:3])

        return (image_to_process, {"samples": final_latent}, final_processed_mask.squeeze(0))
