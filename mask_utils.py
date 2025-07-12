# file: mask_utils.py

import torch
import torchvision.transforms.functional as TF
import numpy as np

class ResizeAllMasks:
    """
    Нода для изменения размера только активной области маски (белой зоны), 
    сохраняя исходный размер холста.
    """
    CATEGORY = "😎 SnJake/Masks"
    FUNCTION = "resize_content"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
            }
        }

    def get_bbox_from_mask(self, mask):
        """Находит bounding box для одной маски."""
        rows = torch.any(mask, axis=1)
        cols = torch.any(mask, axis=0)
        if not torch.any(rows):
            return None
        rmin, rmax = torch.where(rows)[0][[0, -1]]
        cmin, cmax = torch.where(cols)[0][[0, -1]]
        # +1 чтобы включить крайний пиксель
        return rmin.item(), rmax.item() + 1, cmin.item(), cmax.item() + 1

    def resize_content(self, masks, scale):
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
            
        original_h, original_w = masks.shape[1], masks.shape[2]
        output_masks = []

        for mask in masks:
            bbox = self.get_bbox_from_mask(mask)

            if bbox is None: # Если маска пустая
                output_masks.append(torch.zeros_like(mask))
                continue

            y1, y2, x1, x2 = bbox
            cropped_mask = mask[y1:y2, x1:x2]
            
            # Масштабируем вырезанную часть
            bbox_h, bbox_w = cropped_mask.shape
            new_h, new_w = int(bbox_h * scale), int(bbox_w * scale)

            if new_h == 0 or new_w == 0: # Если масштаб слишком мал
                output_masks.append(torch.zeros_like(mask))
                continue

            # TF.resize требует как минимум 3 измерения
            resized_crop = TF.resize(cropped_mask.unsqueeze(0), size=[new_h, new_w], interpolation=TF.InterpolationMode.NEAREST)
            
            # Создаем новый пустой холст
            new_canvas = torch.zeros((original_h, original_w), device=masks.device, dtype=masks.dtype)

            # Находим центр оригинального bbox
            center_y, center_x = (y1 + y2) / 2, (x1 + x2) / 2
            
            # Находим координаты для вставки, чтобы центры совпали
            paste_y1 = int(round(center_y - new_h / 2))
            paste_x1 = int(round(center_x - new_w / 2))

            # Обрезаем, если выходит за границы (клиппинг)
            target_y1 = max(0, paste_y1)
            target_x1 = max(0, paste_x1)
            target_y2 = min(original_h, paste_y1 + new_h)
            target_x2 = min(original_w, paste_x1 + new_w)

            crop_src_y1 = max(0, -paste_y1)
            crop_src_x1 = max(0, -paste_x1)
            crop_src_y2 = crop_src_y1 + (target_y2 - target_y1)
            crop_src_x2 = crop_src_x1 + (target_x2 - target_x1)

            # Вставляем отмасштабированную маску на новый холст
            if target_y1 < target_y2 and target_x1 < target_x2:
                new_canvas[target_y1:target_y2, target_x1:target_x2] = resized_crop[0, crop_src_y1:crop_src_y2, crop_src_x1:crop_src_x2]
            
            output_masks.append(new_canvas)

        return (torch.stack(output_masks),)

class BlurImageByMasks:
    """
    Нода для применения Гауссова размытия на изображение по областям масок с растушевкой краев.
    """
    CATEGORY = "😎 SnJake/Effects"
    FUNCTION = "blur"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "masks": ("MASK",),
                "blur_radius": ("INT", {"default": 25, "min": 1, "max": 201, "step": 2, "tooltip": "Сила размытия для самого изображения."}),
                "feather_amount": ("INT", {"default": 15, "min": 0, "max": 201, "step": 2, "tooltip": "Сила размытия краев маски для создания плавного перехода."}),
            }
        }

    def blur(self, image, masks, blur_radius, feather_amount):
        # 1. Готовим полностью размытое изображение
        if blur_radius % 2 == 0: blur_radius += 1
        
        image_bchw = image.permute(0, 3, 1, 2)
        blurred_image_bchw = TF.gaussian_blur(image_bchw, kernel_size=(blur_radius, blur_radius))
        blurred_image = blurred_image_bchw.permute(0, 2, 3, 1)

        # 2. Готовим маску (с растушевкой)
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
        
        # Совмещаем количество масок и изображений
        if masks.shape[0] != image.shape[0]:
            if masks.shape[0] == 1:
                masks = masks.repeat(image.shape[0], 1, 1)
            else:
                 raise ValueError("Количество масок должно соответствовать количеству изображений или быть равным 1.")

        blended_mask = masks
        if feather_amount > 0:
            if feather_amount % 2 == 0: feather_amount += 1
            # Добавляем канал для blur-функции: [B, H, W] -> [B, 1, H, W]
            feather_mask = masks.unsqueeze(1)
            blurred_mask_bchw = TF.gaussian_blur(feather_mask, kernel_size=(feather_amount, feather_amount))
            # Убираем канал обратно: [B, 1, H, W] -> [B, H, W]
            blended_mask = blurred_mask_bchw.squeeze(1)
        
        # 3. Смешиваем изображения, используя размытую маску
        mask_expanded = blended_mask.unsqueeze(-1)
        output_image = image * (1 - mask_expanded) + blurred_image * mask_expanded
        
        return (output_image,)

class OverlayImageByMasks:
    """
    Нода для наложения изображения по маске с автоматическим и ручным масштабированием, 
    прозрачностью и поддержкой альфа-канала.
    """
    CATEGORY = "😎 SnJake/Masks"
    FUNCTION = "overlay"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "masks": ("MASK",),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def get_bbox_from_mask(self, mask):
        rows = torch.any(mask, axis=1)
        cols = torch.any(mask, axis=0)
        if not torch.any(rows):
            return None
        rmin, rmax = torch.where(rows)[0][[0, -1]]
        cmin, cmax = torch.where(cols)[0][[0, -1]]
        return rmin.item(), rmax.item(), cmin.item(), cmax.item()

    def overlay(self, base_image, overlay_image, masks, scale, opacity):
        if overlay_image.shape[0] != 1:
            raise ValueError("Изображение для наложения должно быть одним (размер батча 1).")
        
        output_image = base_image.clone()
        base_has_alpha = base_image.shape[-1] == 4
        
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
            
        if masks.shape[0] != base_image.shape[0]:
            if masks.shape[0] == 1:
                masks = masks.repeat(base_image.shape[0], 1, 1)
            else:
                raise ValueError("Количество масок должно соответствовать количеству изображений или быть равным 1.")

        overlay_bchw = overlay_image.permute(0, 3, 1, 2)
        overlay_has_alpha = overlay_bchw.shape[1] == 4

        for i in range(base_image.shape[0]):
            mask = masks[i]
            bbox = self.get_bbox_from_mask(mask)
            if bbox is None:
                continue

            y1, y2, x1, x2 = bbox
            bbox_h, bbox_w = y2 - y1 + 1, x2 - x1 + 1
            
            # Применяем дополнительный масштаб
            scaled_h, scaled_w = int(bbox_h * scale), int(bbox_w * scale)
            if scaled_h == 0 or scaled_w == 0: continue

            # Масштабируем оверлей
            resized_overlay_bchw = TF.resize(overlay_bchw, size=[scaled_h, scaled_w], antialias=True)
            resized_overlay = resized_overlay_bchw.permute(0, 2, 3, 1).squeeze(0)

            # Центрируем отмасштабированный оверлей относительно центра bbox
            center_x, center_y = x1 + bbox_w // 2, y1 + bbox_h // 2
            paste_x1, paste_y1 = center_x - scaled_w // 2, center_y - scaled_h // 2
            
            # --- Логика обрезки (Clipping) ---
            # Область на базовом изображении, куда будем вставлять
            img_h, img_w = base_image.shape[1], base_image.shape[2]
            target_x1 = max(0, paste_x1)
            target_y1 = max(0, paste_y1)
            target_x2 = min(img_w, paste_x1 + scaled_w)
            target_y2 = min(img_h, paste_y1 + scaled_h)

            # Область на оверлее, которую будем вырезать
            crop_x1 = max(0, -paste_x1)
            crop_y1 = max(0, -paste_y1)
            crop_x2 = crop_x1 + (target_x2 - target_x1)
            crop_y2 = crop_y1 + (target_y2 - target_y1)

            if target_x1 >= target_x2 or target_y1 >= target_y2: continue
            
            # Вырезаем нужные части
            base_region = output_image[i, target_y1:target_y2, target_x1:target_x2, :]
            mask_region = mask[target_y1:target_y2, target_x1:target_x2].unsqueeze(-1)
            overlay_cropped = resized_overlay[crop_y1:crop_y2, crop_x1:crop_x2, :]

            # --- Логика смешивания с альфа-каналом ---
            overlay_rgb = overlay_cropped[..., :3]
            if overlay_has_alpha:
                overlay_alpha = overlay_cropped[..., 3:4]
            else:
                overlay_alpha = torch.ones_like(overlay_rgb[..., :1])
            
            # Фикс ошибки: работаем только с RGB каналами основного изображения при смешивании
            base_rgb_region = base_region[..., :3]

            final_alpha_mask = overlay_alpha * mask_region * opacity
            
            blended_rgb = base_rgb_region * (1.0 - final_alpha_mask) + overlay_rgb * final_alpha_mask

            # Если у базового изображения есть альфа-канал, сохраним его
            if base_has_alpha:
                 output_image[i, target_y1:target_y2, target_x1:target_x2, :3] = blended_rgb
                 # Можно добавить логику для смешивания альфа-каналов, если нужно
            else:
                 output_image[i, target_y1:target_y2, target_x1:target_x2, :] = blended_rgb
            
        return (output_image,)
