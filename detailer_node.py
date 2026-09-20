import math
import random

import numpy as np
import torch
import torch.nn.functional as TF
from PIL import Image
from scipy.ndimage import gaussian_filter, label, find_objects, grey_dilation

import comfy.utils
import comfy.samplers
import comfy.model_management
import nodes


def resize_image(image, width, height, algorithm):
    """Resize CPU BHWC pixels without an 8-bit round trip."""
    if image.shape[1:3] == (height, width):
        return image
    if algorithm == "lanczos":
        output = torch.empty((image.shape[0], height, width, image.shape[-1]), dtype=image.dtype)
        for b in range(image.shape[0]):
            for c in range(image.shape[-1]):
                plane = Image.fromarray(image[b, :, :, c].float().numpy())
                output[b, :, :, c] = torch.from_numpy(np.array(plane.resize((width, height), Image.Resampling.LANCZOS)))
        return output.clamp_(0, 1)
    options = {"align_corners": False, "antialias": True} if algorithm in ("bilinear", "bicubic") else {}
    return TF.interpolate(image.movedim(-1, 1), size=(height, width), mode=algorithm, **options).movedim(1, -1).clamp_(0, 1)


def encode_pixels(vae, pixels, mode):
    if mode == "tiled" or (mode == "auto" and pixels.shape[1] * pixels.shape[2] > 1024 * 1024):
        return vae.encode_tiled(pixels, tile_x=512, tile_y=512, overlap=64)
    return vae.encode(pixels)


def sample_crop(model, positive, negative, vae, pixels, mask, seed, steps, cfg, sampler, scheduler, denoise, vae_mode):
    # Function scope releases sampling tensors before the next crop or final encode.
    samples = encode_pixels(vae, pixels, vae_mode)
    noise_mask = TF.interpolate(mask[None, None], size=samples.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    if "masked_image" in model.get_model_object("concat_keys"):
        masked_pixels = (pixels - 0.5) * (1.0 - mask.round()[None, :, :, None]) + 0.5
        concat = encode_pixels(vae, masked_pixels, vae_mode)
        del masked_pixels
        positive = [[c[0], {**c[1], "concat_latent_image": concat, "concat_mask": mask[None, None]}] for c in positive]
        negative = [[c[0], {**c[1], "concat_latent_image": concat, "concat_mask": mask[None, None]}] for c in negative]
    result = nodes.common_ksampler(model, seed, steps, cfg, sampler, scheduler, positive, negative,
                                  {"samples": samples, "noise_mask": noise_mask}, denoise=denoise)[0]["samples"]
    if vae_mode == "tiled" or (vae_mode == "auto" and pixels.shape[1] * pixels.shape[2] > 1024 * 1024):
        compression = vae.spacial_compression_decode()
        decoded = vae.decode_tiled(result, tile_x=512 // compression, tile_y=512 // compression, overlap=64 // compression)
    else:
        decoded = vae.decode(result)
    return decoded.cpu()


def process_region(image, processed_mask, labels, component, model, positive, negative, vae,
                   seed, steps, cfg, sampler, scheduler, denoise, expand, blend,
                   force_width, force_height, padding, downscale, upscale, vae_mode):
    height, width = image.shape[1:3]
    sy, sx = component["slice"]
    # Retain the historical expansion amount for existing workflows.
    kernel = math.ceil(expand / 4 * 1.5 + 1) if expand else 1
    margin = kernel + blend
    x0, y0 = max(0, sx.start - margin), max(0, sy.start - margin)
    x1, y1 = min(width, sx.stop + margin), min(height, sy.stop + margin)
    obj = (labels[y0:y1, x0:x1] == component["label"]).astype(np.float32)
    context = grey_dilation(obj, size=(kernel, kernel)) if expand else obj
    rows = np.flatnonzero(context.any(axis=1))
    cols = np.flatnonzero(context.any(axis=0))
    x, y = x0 + int(cols[0]), y0 + int(rows[0])
    w, h = int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)
    target_w = math.ceil((force_width or w) / padding) * padding
    target_h = math.ceil((force_height or h) / padding) * padding
    aspect = target_w / target_h
    cw, ch = (max(w, int(h * aspect)), h) if w / h < aspect else (w, max(h, int(w / aspect)))
    cx, cy = x - (cw - w) // 2, y - (ch - h) // 2
    ix0, iy0, ix1, iy1 = max(0, cx), max(0, cy), min(width, cx + cw), min(height, cy + ch)
    pads = (ix0 - cx, cx + cw - ix1, iy0 - cy, cy + ch - iy1)
    crop = image[:, iy0:iy1, ix0:ix1, :3]
    if any(pads):
        crop = TF.pad(crop.movedim(-1, 1), pads, mode="replicate").movedim(1, -1)
    algorithm = upscale if target_w > cw or target_h > ch else downscale
    pixels = resize_image(crop, target_w, target_h, algorithm)

    context_crop = np.zeros((ch, cw), dtype=np.float32)
    object_crop = np.zeros((ch, cw), dtype=np.float32)
    ox0, oy0, ox1, oy1 = max(cx, x0), max(cy, y0), min(cx + cw, x1), min(cy + ch, y1)
    dst = (slice(oy0 - cy, oy1 - cy), slice(ox0 - cx, ox1 - cx))
    src = (slice(oy0 - y0, oy1 - y0), slice(ox0 - x0, ox1 - x0))
    context_crop[dst] = context[src]
    object_crop[dst] = obj[src]
    # Outside-image context is denoised, but never stitched into the original.
    if pads[0]: context_crop[:, :pads[0]] = 1
    if pads[1]: context_crop[:, -pads[1]:] = 1
    if pads[2]: context_crop[:pads[2]] = 1
    if pads[3]: context_crop[-pads[3]:] = 1
    mask = TF.interpolate(torch.from_numpy(context_crop)[None, None], size=(target_h, target_w), mode="nearest")[0, 0]
    print(f"Sequential Mask Detailer: crop {cw}x{ch} -> {target_w}x{target_h}, VAE={vae_mode}")
    decoded = sample_crop(model, positive, negative, vae, pixels, mask, seed, steps, cfg, sampler, scheduler, denoise, vae_mode)
    algorithm = upscale if cw > decoded.shape[2] or ch > decoded.shape[1] else downscale
    resized = resize_image(decoded, cw, ch, algorithm)
    alpha = torch.from_numpy(gaussian_filter(object_crop, sigma=blend / 4) if blend else object_crop)
    ry, rx = slice(iy0 - cy, iy1 - cy), slice(ix0 - cx, ix1 - cx)
    alpha = alpha[ry, rx]
    # Only mutate our owned output canvas, never the caller's IMAGE.
    dest = image[:, iy0:iy1, ix0:ix1, :3]
    dest.lerp_(resized[:, ry, rx].to(dest.dtype), alpha[None, :, :, None].to(dest.dtype))
    processed_mask[iy0:iy1, ix0:ix1].add_(alpha).clamp_(0, 1)


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
            },
            "optional": {
                "vae_mode": (["auto", "tiled", "regular"], {"default": "auto", "tooltip": "auto: VAE по тайлам для областей больше 1024×1024; tiled: всегда по тайлам; regular: обычный VAE."}),
                "generate_final_latent": ("BOOLEAN", {"default": True, "tooltip": "Кодировать итоговое изображение в latent. Выключите, если используете только image/mask. При выключении latent = None; не подключайте этот выход к другим нодам."}),
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
                            mask_process_order, vae_mode="auto", generate_final_latent=True):
        if image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError("Sequential Mask Detailer expects a BHWC RGB image.")
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if masks.ndim != 3:
            raise ValueError("Sequential Mask Detailer expects masks shaped [B,H,W] or [H,W].")
        batch, height, width = image.shape[:3]
        if batch > 1 and masks.shape[0] not in (0, 1, batch):
            raise ValueError("For an image batch, supply one shared mask or one mask per image.")
        output = image.to(device="cpu", copy=True)
        processed = torch.zeros((batch, height, width), dtype=torch.float32)
        rng = random.Random(noise_seed)
        for b in range(batch):
            # A mask stack on a single image denotes a union of regions.
            mask = torch.zeros((height, width), dtype=torch.float32)
            sources = range(masks.shape[0]) if batch == 1 else ([0 if masks.shape[0] == 1 else b] if masks.shape[0] else [])
            for index in sources:
                current = masks[index].detach().to(device="cpu", dtype=torch.float32)
                if current.shape != (height, width):
                    current = TF.interpolate(current[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
                torch.maximum(mask, current, out=mask)
                del current
            binary = (mask.numpy() > 0.5) & (mask.numpy() >= mask_hipass_filter)
            labels, count = label(binary)
            del mask, binary
            components = []
            for ident, region in enumerate(find_objects(labels), 1):
                if region is None:
                    continue
                local = labels[region] == ident
                row_counts, col_counts = local.sum(axis=1), local.sum(axis=0)
                area = int(row_counts.sum())
                components.append({"label": ident, "slice": region, "area": area,
                                   "center_y": float(np.dot(np.arange(len(row_counts)), row_counts) / area + region[0].start),
                                   "center_x": float(np.dot(np.arange(len(col_counts)), col_counts) / area + region[1].start)})
                del local, row_counts, col_counts
            keys = {"слева-направо": "center_x", "справа-налево": "center_x", "сверху-вниз": "center_y", "снизу-вверх": "center_y", "от большей к меньшей": "area", "от меньшей к большей": "area"}
            if mask_process_order in keys:
                components.sort(key=lambda item: item[keys[mask_process_order]], reverse=mask_process_order in ("справа-налево", "снизу-вверх", "от большей к меньшей"))
            elif mask_process_order == "случайно":
                rng.shuffle(components)
            progress = comfy.utils.ProgressBar(count)
            for component in components:
                comfy.model_management.throw_exception_if_processing_interrupted()
                if denoise > 0:
                    process_region(output[b:b+1], processed[b], labels, component, model, positive, negative, vae,
                                   noise_seed, steps, cfg, sampler_name, scheduler, denoise, mask_expand_pixels,
                                   mask_blend_pixels, force_width, force_height, padding, downscale_algorithm,
                                   upscale_algorithm, vae_mode)
                noise_seed = (noise_seed + 1) & 0xffffffffffffffff
                progress.update(1)
            del labels, components
        latent = None
        if generate_final_latent:
            # Encode one image at a time even if the input is a batch.
            for b in range(batch):
                encoded = encode_pixels(vae, output[b:b+1, :, :, :3], vae_mode).cpu()
                if latent is None:
                    latent = {"samples": torch.empty((batch, *encoded.shape[1:]), dtype=encoded.dtype)}
                latent["samples"][b:b+1].copy_(encoded)
                del encoded
        return output, latent, processed
