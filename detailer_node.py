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
from scipy.ndimage import gaussian_filter, label, find_objects

class DetailerForEachMask:
    """
    Эта нода последовательно детализирует области на изображении, указанные масками.
    Она перебирает каждую маску, вырезает соответствующую область с контекстом,
    применяет семплер для детализации, а затем вшивает результат обратно.
    Идеально подходит для улучшения лиц, объектов или других деталей с помощью BBOX или сегментационных масок.
    """
    
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp", "lanczos"]

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
                "rescale_algorithm": (cls.upscale_methods, {"default": "bicubic", "tooltip": "Алгоритм, используемый для масштабирования области до и после семплирования."}),
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
        "Комбинированная маска всех обработанных областей с учетом размытия для смешивания (blend_pixels)."
    )
    FUNCTION = "detail_sequentially"
    CATEGORY = "😎 SnJake/Detailer"

    def rescale(self, image_tensor, width, height, algorithm):
        if image_tensor.dim() == 4:
            samples = image_tensor.movedim(-1, 1)
        else:
            samples = image_tensor.unsqueeze(0).unsqueeze(0)
        
        if algorithm == "bislerp": algorithm = "bicubic"
        rescale_pil_algorithm = getattr(Image, algorithm.upper())
        
        rescaled_tensors = []
        for sample in samples:
            pil_img = F.to_pil_image(sample.cpu())
            rescaled_pil = pil_img.resize((width, height), rescale_pil_algorithm)
            rescaled_tensors.append(F.to_tensor(rescaled_pil))
            
        output = torch.stack(rescaled_tensors).to(image_tensor.device)

        if image_tensor.dim() == 4:
            return output.movedim(1, -1)
        else:
            return output.squeeze(0).squeeze(0)

    def apply_padding(self, min_val, max_val, max_boundary, padding):
        original_range = max_val - min_val
        if original_range % padding == 0: return min_val, max_val
        midpoint = (min_val + max_val) // 2
        new_range = ((original_range // padding) + 1) * padding
        new_min_val = max(midpoint - new_range // 2, 0)
        new_max_val = new_min_val + new_range
        if new_max_val > max_boundary:
            new_max_val = max_boundary
            new_min_val = max(new_max_val - new_range, 0)
        return int(new_min_val), int(new_max_val)

    def blur_inpaint_mask(self, mask, blur_pixels):
        if blur_pixels <= 0: return mask
        mask_np = mask.cpu().numpy()
        sigma = blur_pixels / 4.0
        blurred_mask_np = gaussian_filter(mask_np, sigma=sigma)
        return torch.from_numpy(blurred_mask_np).to(mask.device)

    def composite(self, destination, source, x, y, mask):
        destination = destination.clone()
        source = source.to(destination.device)
        mask = mask.to(destination.device)
        if mask.dim() == 2: mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3: mask = mask.unsqueeze(1)
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])
        mask = torch.nn.functional.interpolate(mask, size=(source.shape[2], source.shape[3]), mode="bilinear")
        left, top, right, bottom = x, y, x + source.shape[3], y + source.shape[2]
        dest_top, dest_bottom = max(0, top), min(destination.shape[2], bottom)
        dest_left, dest_right = max(0, left), min(destination.shape[3], right)
        if dest_top >= dest_bottom or dest_left >= dest_right: return destination
        src_top, src_bottom = dest_top - top, dest_bottom - top
        src_left, src_right = dest_left - left, dest_right - left
        destination_slice = destination[:, :, dest_top:dest_bottom, dest_left:dest_right]
        source_slice = source[:, :, src_top:src_bottom, src_left:src_right]
        mask_slice = mask[:, :, src_top:src_bottom, src_left:src_right]
        blended_slice = source_slice * mask_slice + destination_slice * (1.0 - mask_slice)
        destination[:, :, dest_top:dest_bottom, dest_left:dest_right] = blended_slice
        return destination

    def detail_sequentially(self, model, positive, negative, vae, image, masks,
                            noise_seed, steps, cfg, sampler_name, scheduler, denoise,
                            context_expand_pixels, blur_mask_pixels, blend_pixels, grow_mask_by,
                            force_width, force_height, rescale_algorithm, padding,
                            mask_process_order):
        if masks.numel() == 0 or masks.max() == 0:
            print("Маски не найдены или пусты. Возвращается исходное изображение.")
            latent = vae.encode(image[:,:,:,:3])
            return (image, {"samples": latent}, torch.zeros_like(masks))

        mask_np = masks.cpu().numpy().squeeze()
        labeled_array, num_features = label(mask_np > 0.5)
        if num_features == 0: return (image, {"samples": vae.encode(image[:,:,:,:3])}, torch.zeros_like(masks))

        found_objects = find_objects(labeled_array)
        decorated_masks = []
        for i, slc in enumerate(found_objects):
            if slc is None: continue
            mask_label = i + 1
            coords = np.argwhere(labeled_array[slc] == mask_label)
            if coords.size == 0: continue
            center_y, center_x = np.mean(coords, axis=0)
            center_y += slc[0].start
            center_x += slc[1].start
            area = len(coords)
            decorated_masks.append({'slice': slc, 'center_x': center_x, 'center_y': center_y, 'area': area, 'label': mask_label})

        sort_key_map = {"слева-направо": "center_x", "справа-налево": "center_x", "сверху-вниз": "center_y", "снизу-вверх": "center_y", "от большей к меньшей": "area", "от меньшей к большей": "area"}
        reverse_map = {"справа-налево", "снизу-вверх", "от большей к меньшей"}
        if mask_process_order in sort_key_map:
            decorated_masks.sort(key=lambda m: m[sort_key_map[mask_process_order]], reverse=(mask_process_order in reverse_map))
        elif mask_process_order == "случайно":
            import random
            random.shuffle(decorated_masks)

        image_to_process = image.clone()
        original_height, original_width = image.shape[1], image.shape[2]
        final_processed_mask = torch.zeros((original_height, original_width), device=image.device)
        pbar = comfy.utils.ProgressBar(num_features)

        for i, mask_info in enumerate(decorated_masks):
            print(f"Обработка маски {i+1}/{num_features}...")
            current_mask_np = (labeled_array == mask_info['label']).astype(np.float32)
            current_mask_tensor = torch.from_numpy(current_mask_np).to(image.device)
            if blur_mask_pixels > 0: current_mask_tensor = self.blur_inpaint_mask(current_mask_tensor, blur_mask_pixels)

            slc = mask_info['slice']
            y_min, y_max, x_min, x_max = slc[0].start, slc[0].stop, slc[1].start, slc[1].stop
            y_min_exp, y_max_exp = max(y_min - context_expand_pixels, 0), min(y_max + context_expand_pixels, original_height)
            x_min_exp, x_max_exp = max(x_min - context_expand_pixels, 0), min(x_max + context_expand_pixels, original_width)
            if padding > 1:
                x_min_exp, x_max_exp = self.apply_padding(x_min_exp, x_max_exp, original_width, padding)
                y_min_exp, y_max_exp = self.apply_padding(y_min_exp, y_max_exp, original_height, padding)

            cropped_image = image_to_process[:, y_min_exp:y_max_exp, x_min_exp:x_max_exp, :]
            cropped_mask_for_stitch = current_mask_tensor[y_min_exp:y_max_exp, x_min_exp:x_max_exp]
            if cropped_image.shape[1] == 0 or cropped_image.shape[2] == 0: continue
            
            original_crop_height, original_crop_width = cropped_image.shape[1], cropped_image.shape[2]
            rescaled = force_width > 0 and force_height > 0
            image_for_inpaint, mask_for_inpaint = (self.rescale(cropped_image, force_width, force_height, rescale_algorithm), self.rescale(cropped_mask_for_stitch, force_width, force_height, "nearest")) if rescaled else (cropped_image, cropped_mask_for_stitch)
            
            pixels_for_concat = image_for_inpaint.clone()
            m = (1.0 - mask_for_inpaint.round()).unsqueeze(-1)
            pixels_for_concat = (pixels_for_concat - 0.5) * m + 0.5
            concat_latent = vae.encode(pixels_for_concat)
            initial_latent_samples = vae.encode(image_for_inpaint)
            latent_for_sampler = {"samples": initial_latent_samples}
            mask_for_sampler = mask_for_inpaint.reshape((-1, 1, mask_for_inpaint.shape[-2], mask_for_inpaint.shape[-1]))
            
            if grow_mask_by > 0:
                kernel = torch.ones((1, 1, grow_mask_by, grow_mask_by), device=image.device)
                padding_val = math.ceil((grow_mask_by - 1) / 2)
                grown_mask = torch.clamp(torch.nn.functional.conv2d(mask_for_sampler.round(), kernel, padding=padding_val), 0, 1)
            else:
                grown_mask = mask_for_sampler

            latent_h, latent_w = initial_latent_samples.shape[2], initial_latent_samples.shape[3]
            latent_for_sampler["noise_mask"] = torch.nn.functional.interpolate(grown_mask, size=(latent_h, latent_w), mode="bilinear").squeeze(1)

            def create_inpaint_cond(cond_list):
                return [[c[0], {**c[1], 'concat_latent_image': concat_latent, 'concat_mask': mask_for_sampler}] for c in cond_list]
            
            positive_inpaint, negative_inpaint = create_inpaint_cond(positive), create_inpaint_cond(negative)
            latent_out = nodes.common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive_inpaint, negative_inpaint, latent_for_sampler, denoise=denoise)
            noise_seed += 1

            decoded_crop = vae.decode(latent_out[0]["samples"])
            if rescaled: decoded_crop = self.rescale(decoded_crop, original_crop_width, original_crop_height, rescale_algorithm)

            blend_mask = self.blur_inpaint_mask(cropped_mask_for_stitch, blend_pixels)
            image_to_process = self.composite(image_to_process.movedim(-1, 1), decoded_crop.movedim(-1, 1), x_min_exp, y_min_exp, blend_mask).movedim(1, -1)
            final_processed_mask[y_min_exp:y_max_exp, x_min_exp:x_max_exp] += blend_mask
            pbar.update(1)
        
        final_processed_mask.clamp_(0.0, 1.0)
        final_latent = vae.encode(image_to_process[:,:,:,:3])
        return (image_to_process, {"samples": final_latent}, final_processed_mask)