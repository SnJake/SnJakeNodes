# filename: pixel_art_utils.py
import torch
import numpy as np
#from PIL import Image # PIL больше не нужен здесь
import kornia.color as kc # Для конвертации в LAB/Luminance в ReplacePalette
import traceback
import math # Для math.isfinite в calculate_dbi

# Определяем устройство один раз
# Лучше определять устройство в момент выполнения узла,
# так как пользователь может переключать CPU/GPU в ComfyUI
def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # elif torch.backends.mps.is_available(): # Раскомментировать для MacOS
    #     return torch.device("mps")
    else:
        return torch.device("cpu")

# --- ИМПОРТЫ ИЗ ОБЩЕЙ ЛОГИКИ ---
# Убедитесь, что папка pixelart и файл pixel_art_logic.py находятся в той же папке, что и этот файл
LOGIC_IMPORTED = False
try:
    # Основные функции квантования и применения палитры из quantization.py
    # (он содержит COLOR_SPACE_RANGES и функции нормализации/проекции)
    from .pixelart.quantization import (
        run_color_quantization,
        apply_fixed_palette,
        apply_fixed_palette_get_labels
        # Остальные специфичные методы (kmeans, sq) вызываются через run_color_quantization
    )
    # Утилиты для работы с цветом из color_utils.py
    from .pixelart.color_utils import (
        to_quantize_space,
        from_quantize_space,
        convert_palette_to_string,
        calculate_dbi # Если используется
    )
    # Логика дизеринга из dithering.py
    from .pixelart.dithering import (
        apply_dithering,
        DIFFUSION_PATTERNS,
        ORDERED_PATTERNS
    )
    LOGIC_IMPORTED = True
    print("Successfully imported PixelArt logic modules for Utils.")
except ImportError as e:
    print("*"*80)
    print("[ERROR] Failed to import from pixelart logic modules!")
    print(f"Error details: {e}")
    print("Please ensure the 'pixelart' folder with logic files exists.")
    print("Nodes requiring this logic (ExtractPaletteNode, ApplyPaletteNode, ReplacePaletteColorsNode) will likely fail.")
    print("*"*80)
    traceback.print_exc()
    raise ImportError("Failed to load PixelArt logic modules.") from e


# --- Новый виртуальный тип ---
PALETTE = "*" # Это будет тензор [N, 3]

# --- Узел Извлечения Палитры ---
class ExtractPaletteNode:
    CATEGORY = "😎 SnJake/PixelArt"
    FUNCTION = "extract_palette"
    RETURN_TYPES = (PALETTE, "STRING") # Возвращаем тензор палитры и HEX-строку
    RETURN_NAMES = ("palette_tensor", "palette_hex")

    @classmethod
    def INPUT_TYPES(cls):
        # Не регистрируем узел, если логика не импортировалась
        if not LOGIC_IMPORTED:
             return {"required": {"error": ("STRING", {"default": "PixelArt logic failed to load", "forceInput": True, "hidden":True})}}

        # Добавим SQ в список методов
        quant_methods = ["kmeans", "median_cut", "octree", "SQ"] # Wu убран, SQ добавлен

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Изображение-донор для извлечения палитры."}),
                "num_colors": ("INT", {"default": 16, "min": 2, "max": 256, "step": 1, "display": "slider"}),
                "method": (quant_methods, {"default": "kmeans", "tooltip": "Метод квантования для извлечения палитры."}), # Обновлен список
                "color_space": (["RGB", "LAB", "YCbCr", "HSV"], {"default": "LAB", "tooltip": "Пространство для вычисления расстояний при квантовании. LAB/YCbCr часто лучше."}), # Обновлен tooltip
                "min_pixel_area": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1, "display": "slider", "tooltip": "Минимальное кол-во пикселей для цвета в палитре (1=выкл)."}), # Обновлен tooltip
                # Добавляем параметры, специфичные для методов
                "max_kmeans_iter": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "sq_iterations_factor": ("INT", {"default": 2, "min": 1, "max": 50, "step": 1,
                     "tooltip": "SQ Iterations = factor * number_of_pixels"
                }),
                "sq_learning_rate_initial": ("FLOAT", { # ИСПРАВЛЕНО
                    "default": 0.1, "min": 1e-4, "max": 1.0, "step": 1e-3, "round": 0.0001,
                    "tooltip": "Initial learning rate for SQ"
                }),
                "sq_learning_rate_decay_time": ("INT", { # ИСПРАВЛЕНО
                    "default": 10000, "min": 100, "max": 1000000, "step": 100,
                     "tooltip": "Time constant (t0) for SQ learning rate decay"
                }),
            }
        }

    def extract_palette(self, image, num_colors, method, color_space, min_pixel_area,
                        max_kmeans_iter, # Специфично для K-Means
                        sq_iterations_factor, # Специфично для SQ
                        sq_learning_rate_initial, # Специфично для SQ
                        sq_learning_rate_decay_time # Специфично для SQ
                        ):

        device = image.device if isinstance(image, torch.Tensor) else get_torch_device()
        print(f"[ExtractPaletteNode] Input image initial device: {image.device}, Target device: {device}")

        if not LOGIC_IMPORTED:
             print("[ExtractPaletteNode] ERROR: PixelArt logic modules failed to load.")
             return (None, "Error: PixelArt logic modules failed to load")
        if image is None:
            return (None, "Error: No input image")

        # --- Подготовка изображения ---
        try:
            image = image.to(device) # Переносим на целевое устройство
            print(f"[ExtractPaletteNode] Input image moved to device: {image.device}")
            image = image.clamp(0, 1)
            # Обработка батча: используем только первое изображение
            if image.dim() == 4 and image.shape[0] > 1:
                 print(f"[ExtractPaletteNode] Warning: Input batch size is {image.shape[0]}. Using only the first image for palette extraction.")
                 image_chw = image[0:1].permute(0, 3, 1, 2).float() # (1, C, H, W), используем float
            elif image.dim() == 4:
                 image_chw = image.permute(0, 3, 1, 2).float() # (1, C, H, W)
            else:
                raise ValueError(f"Unexpected input image shape: {image.shape}")

            batch_size, channels, height, width = image_chw.shape
            if channels != 3:
                 raise ValueError(f"ExtractPaletteNode requires RGB input (3 channels), got {channels}")

        except Exception as e:
             print(f"[ExtractPaletteNode] Error during image preparation: {e}")
             traceback.print_exc()
             return (None, f"Error: Image Prep Failed - {e}")


        # --- Определяем рабочее пространство ---
        # (Та же логика, что и в PixelArtNode)
        processing_space = color_space
        method_clean = method.strip().lower()
        is_metric_sensitive_method = method_clean in ["kmeans", "median_cut", "sq"] # Методы, чувствительные к метрике
        if color_space == "HSV" and is_metric_sensitive_method:
             print(f"[ExtractPaletteNode] Warning: Using HSV with '{method}' is unreliable. Forcing RGB for calculations.")
             processing_space = "RGB"
        # Octree fallback (kmeans) работает в processing_space
        elif method_clean == "octree" and color_space != "RGB":
             print(f"[ExtractPaletteNode] Note: Method '{method}' (fallback to kmeans) will use {color_space} space as requested.")
             # Fallback (kmeans) будет работать в выбранном пространстве

        print(f"[ExtractPaletteNode] Extracting palette using '{method_clean}' in {processing_space} space (User selected: {color_space}).")

        final_centroids_rgb = None
        try:
            # 1. Конвертируем изображение в рабочее пространство
            # Убедимся, что на входе float
            img_in_processing_space = to_quantize_space(image_chw.float(), processing_space)

            # 2. Собираем параметры для квантования
            quant_params = {
                "num_colors": num_colors,
                "method": method_clean,
                "min_pixel_area": min_pixel_area,
                "processing_space": processing_space,
                "auto_num_colors": False, # Не используем авто K здесь
                # Параметры конкретных методов
                "kmeans_max_iter": max_kmeans_iter,
                "sq_iterations_factor": sq_iterations_factor,
                "sq_learning_rate_initial": sq_learning_rate_initial,
                "sq_learning_rate_decay_time": sq_learning_rate_decay_time
            }

            # 3. Выполняем квантование для получения центроидов
            _, final_centroids_in_space = run_color_quantization(
                img_in_processing_space, **quant_params # Передаем параметры
            )

            # 4. Конвертируем центроиды обратно в RGB [0,1]
            if final_centroids_in_space is not None and final_centroids_in_space.shape[0] > 0:
                 # Добавляем недостающие измерения для конвертации
                 centroids_expanded = final_centroids_in_space.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                 final_centroids_rgb = from_quantize_space(centroids_expanded, processing_space)
                 final_centroids_rgb = final_centroids_rgb.squeeze().clamp(0, 1) # Убираем лишние измерения и зажимаем
                 # Убеждаемся, что результат 2D (N, C)
                 if final_centroids_rgb.ndim == 1:
                      final_centroids_rgb = final_centroids_rgb.unsqueeze(0)
            else:
                 print("[ExtractPaletteNode] Warning: Quantization did not return valid centroids.")
                 final_centroids_rgb = None

        except Exception as e:
            print(f"[ExtractPaletteNode] Error during palette extraction: {e}")
            traceback.print_exc()
            return (None, f"Error: {e}")

        # 5. Формируем выход
        if final_centroids_rgb is None or final_centroids_rgb.shape[0] == 0:
            return (None, "Error: Failed to extract palette")

        palette_tensor = final_centroids_rgb.float() # Возвращаем float
        palette_hex = convert_palette_to_string(palette_tensor) # Используем функцию из logic

        print(f"[ExtractPaletteNode] Extracted palette with {palette_tensor.shape[0]} colors.")
        return (palette_tensor, palette_hex)


# --- Узел Применения Палитры ---
class ApplyPaletteNode:
    CATEGORY = "😎 SnJake/PixelArt/Utils" # Уточнили категорию
    FUNCTION = "apply_palette"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        if not LOGIC_IMPORTED:
             return {"required": {"error": ("STRING", {"default": "PixelArt logic failed to load", "forceInput": True, "hidden":True})}}

        dither_patterns = ["No Dithering"] + list(DIFFUSION_PATTERNS.keys()) + list(ORDERED_PATTERNS.keys()) + ["WhiteNoise"]

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Целевое изображение (обычно после PixelArtNode без квантования)."}),
                "palette_tensor": (PALETTE, {"tooltip": "Палитра (тензор [N, 3] RGB 0-1) от ExtractPaletteNode."}),
                "dithering": ("BOOLEAN", {"default": True, "tooltip": "Применить дизеринг при сопоставлении с палитрой."}),
                "dither_pattern": (dither_patterns, {"default": "Floyd-Steinberg"}),
                "dither_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "color_space": (["RGB", "LAB", "YCbCr", "HSV"], {"default": "LAB", "tooltip": "Пространство для поиска ближайшего цвета при МЭППИНГЕ (дизеринг всегда в RGB). LAB/YCbCr часто лучше."}), # Уточнили tooltip
                "color_distance_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Только для Error Diffusion дизеринга. Порог евклидова расстояния для распространения ошибки (0 = всегда)."}),
            }
        }

    def apply_palette(self, image, palette_tensor, dithering, dither_pattern, dither_strength, color_space, color_distance_threshold):
        device = image.device if isinstance(image, torch.Tensor) else get_torch_device()
        print(f"[ApplyPaletteNode] Input image initial device: {image.device}, Target device: {device}")

        if not LOGIC_IMPORTED:
             print("[ApplyPaletteNode] ERROR: PixelArt logic modules failed to load.")
             return (image,) # Возвращаем оригинал

        if image is None:
            print("[ApplyPaletteNode] Error: No input image.")
            return (None,)
        if palette_tensor is None or not isinstance(palette_tensor, torch.Tensor) or palette_tensor.ndim != 2 or palette_tensor.shape[1] != 3:
            print(f"[ApplyPaletteNode] Error: Invalid palette input. Expected tensor [N, 3], got: {type(palette_tensor)}")
            return (image,)

        # --- Подготовка ---
        image = image.to(device)
        palette_tensor = palette_tensor.to(device)
        print(f"[ApplyPaletteNode] Inputs moved to device: {device}")

        image_chw = image.permute(0, 3, 1, 2).float().clamp(0, 1) # B, C, H, W - используем float
        palette_rgb = palette_tensor.float().clamp(0, 1) # N, C - используем float

        print(f"[ApplyPaletteNode] Applying palette with {palette_rgb.shape[0]} colors. Dithering: {dithering} ({dither_pattern} / Strength: {dither_strength:.2f}). Mapping space: {color_space}.")

        result_image_chw = torch.zeros_like(image_chw)

        # --- Выбор метода: Дизеринг или Прямое Сопоставление ---
        use_dither = dithering and dither_pattern != "No Dithering" and dither_strength > 0
        if use_dither:
            # --- Дизеринг (всегда работает в RGB) ---
            print("  - Applying dithering...")
            try:
                # apply_dithering ожидает RGB источник и RGB палитру
                # Передаем функцию apply_fixed_palette из логики
                result_image_chw = apply_dithering(
                    image_chw, # Источник для ошибки уже должен быть float RGB [0,1]
                    palette_rgb,
                    dither_pattern,
                    dither_strength,
                    color_distance_threshold,
                    apply_fixed_palette,
                )
            except Exception as e:
                 print(f"[ApplyPaletteNode] Error during dithering: {e}")
                 traceback.print_exc()
                 print("  - Falling back to non-dithered palette application (RGB).")
                 # Fallback на обычное применение палитры в RGB
                 color_space = "RGB" # Меняем color_space на RGB для fallback
                 use_dither = False # Отключаем флаг дизеринга для выполнения блока else

        if not use_dither:
            # --- Простое сопоставление с палитрой (без дизеринга) ---
            print(f"  - Applying palette using nearest neighbor in {color_space} space (Dithering disabled or failed).")
            try:
                # 1. Конвертируем изображение и палитру в целевое пространство
                img_in_space = to_quantize_space(image_chw, color_space)
                # Конвертируем RGB палитру в color_space
                palette_expanded = palette_rgb.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                palette_in_space = to_quantize_space(palette_expanded, color_space).squeeze()
                if palette_in_space.ndim == 1: palette_in_space = palette_in_space.unsqueeze(0)

                # 2. Применяем палитру (поиск ближайшего) в этом пространстве
                quantized_img_in_space = apply_fixed_palette(img_in_space, palette_in_space)

                # 3. Конвертируем результат обратно в RGB
                result_image_chw = from_quantize_space(quantized_img_in_space, color_space)

            except Exception as e:
                print(f"[ApplyPaletteNode] Error during palette application in {color_space}: {e}")
                traceback.print_exc()
                result_image_chw = image_chw # Fallback

        # --- Выход ---
        # Конвертируем обратно в формат ComfyUI (B, H, W, C)
        final_image_hwc = result_image_chw.permute(0, 2, 3, 1).clamp(0, 1)

        return (final_image_hwc,)


# --- Узел Замены Цветов Палитры ---
class ReplacePaletteColorsNode:
    CATEGORY = "😎 SnJake/PixelArt/Utils" # Уточнили категорию
    FUNCTION = "replace_colors"
    RETURN_TYPES = ("IMAGE",)

    SORT_METHODS = ["None", "Luminance", "Hue", "Saturation", "Value"]

    @classmethod
    def INPUT_TYPES(cls):
        if not LOGIC_IMPORTED:
             return {"required": {"error": ("STRING", {"default": "PixelArt logic failed to load", "forceInput": True, "hidden":True})}}

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Квантованное изображение (желательно с небольшим кол-вом цветов)."}),
                "source_palette": (PALETTE, {"tooltip": "Исходная палитра изображения ([N, 3] RGB 0-1)."}),
                "replacement_palette": (PALETTE, {"tooltip": "Палитра донора ([M, 3] RGB 0-1)."}),
                "sort_method": (cls.SORT_METHODS, {"default": "Luminance", "tooltip": "Метод сортировки палитр перед сопоставлением цветов 1-к-1."}),
                "mismatch_handling": (["Error", "Trim Replacement", "Repeat Replacement"], {"default": "Trim Replacement", "tooltip": "Как обрабатывать несовпадение кол-ва цветов после сортировки."}),
                "tolerance": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.5, "step": 0.001, "round":0.0001, "tooltip": "Допуск при поиске цветов исходной палитры на изображении."})
            }
        }

    def _sort_palette(self, palette_rgb, method, device):
        """Сортирует палитру [N, 3] RGB 0-1 по выбранному методу."""
        if method == "None" or palette_rgb is None or palette_rgb.shape[0] <= 1:
            return palette_rgb, torch.arange(palette_rgb.shape[0], device=device) # Возвращаем оригинал и исходные индексы

        palette_rgb = palette_rgb.float().to(device) # Работаем с float на нужном устройстве

        try:
            if method == "Luminance":
                # Используем Y из YCbCr Kornia для простоты и скорости
                # Kornia ожидает (B, C, H, W), добавляем измерения
                palette_ycbcr = kc.rgb_to_ycbcr(palette_rgb.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)).squeeze()
                if palette_ycbcr.ndim == 1: palette_ycbcr = palette_ycbcr.unsqueeze(0)
                values_to_sort = palette_ycbcr[:, 0] # Y канал
            elif method in ["Hue", "Saturation", "Value"]:
                 palette_hsv = kc.rgb_to_hsv(palette_rgb.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)).squeeze()
                 if palette_hsv.ndim == 1: palette_hsv = palette_hsv.unsqueeze(0)
                 idx = {"Hue": 0, "Saturation": 1, "Value": 2}[method]
                 values_to_sort = palette_hsv[:, idx]
                 # Для Hue (цикличный), сортировка может быть неидеальной, но это стандартный подход
            else: # Неизвестный метод
                print(f"Warning: Unknown sort method '{method}'. Sorting skipped.")
                return palette_rgb, torch.arange(palette_rgb.shape[0], device=device)

            # Получаем индексы сортировки
            sorted_indices = torch.argsort(values_to_sort)
            sorted_palette = palette_rgb[sorted_indices]
            return sorted_palette, sorted_indices

        except Exception as e:
            print(f"Error during palette sorting by {method}: {e}")
            traceback.print_exc()
            print("Sorting skipped due to error.")
            return palette_rgb, torch.arange(palette_rgb.shape[0], device=device)


    def replace_colors(self, image, source_palette, replacement_palette, sort_method, mismatch_handling, tolerance):
        device = image.device if isinstance(image, torch.Tensor) else get_torch_device()
        print(f"[ReplacePalette] Input image device: {image.device}, Target device: {device}")

        # --- Проверки входов ---
        if not LOGIC_IMPORTED: return (image,) # Если логика не загружена
        if image is None: return (None,)
        if source_palette is None or not isinstance(source_palette, torch.Tensor) or source_palette.ndim != 2 or source_palette.shape[1] != 3:
            print("[ReplacePalette] Error: Invalid source_palette.")
            return (image,)
        if replacement_palette is None or not isinstance(replacement_palette, torch.Tensor) or replacement_palette.ndim != 2 or replacement_palette.shape[1] != 3:
            print("[ReplacePalette] Error: Invalid replacement_palette.")
            return (image,)
        if tolerance < 0: tolerance = 0.0

        # --- Подготовка ---
        image = image.to(device)
        source_palette = source_palette.float().clamp(0, 1).to(device)
        replacement_palette = replacement_palette.float().clamp(0, 1).to(device)

        n_source_orig = source_palette.shape[0]
        n_replace_orig = replacement_palette.shape[0]

        print(f"[ReplacePalette] Original counts: Source={n_source_orig}, Replacement={n_replace_orig}. Sort: {sort_method}")

        # --- Сортировка палитр ---
        sorted_source_palette, _ = self._sort_palette(source_palette, sort_method, device)
        sorted_replacement_palette, _ = self._sort_palette(replacement_palette, sort_method, device) # Индексы донора не нужны

        n_source = sorted_source_palette.shape[0]
        n_replace = sorted_replacement_palette.shape[0]

        # --- Обработка несовпадения количества цветов ПОСЛЕ сортировки ---
        final_replacement_palette = sorted_replacement_palette
        if n_source != n_replace:
            print(f"Warning: Sorted palette size mismatch ({n_source} vs {n_replace}). Handling: {mismatch_handling}")
            if mismatch_handling == "Error":
                print("ERROR: Palette sizes must match.")
                return (image,) # Возвращаем оригинал при ошибке
            elif mismatch_handling == "Trim Replacement":
                final_replacement_palette = sorted_replacement_palette[:n_source]
            elif mismatch_handling == "Repeat Replacement":
                if n_replace == 0:
                    print("ERROR: Replacement palette is empty.")
                    return (image,)
                repeat_times = math.ceil(n_source / n_replace) # Сколько раз повторить
                final_replacement_palette = sorted_replacement_palette.repeat(repeat_times, 1)[:n_source]

            if final_replacement_palette.shape[0] != n_source:
                 print(f"ERROR: Failed to handle palette mismatch correctly. Sizes: {n_source} vs {final_replacement_palette.shape[0]}")
                 return (image,) # Ошибка

        if n_source == 0 or final_replacement_palette.shape[0] == 0:
             print("Error: One or both palettes are empty after handling mismatch.")
             return (image,)

        # --- Применение карты замен к изображению ---
        image_chw = image.permute(0, 3, 1, 2).float() # B, C, H, W
        B, C, H, W = image_chw.shape
        output_image_chw = image_chw.clone() # Работаем с копией

        print(f"Applying replacement map ({n_source} colors) with tolerance {tolerance}...")
        replaced_count = 0
        # Проходим по *отсортированной* исходной палитре
        for i, src_color_tensor in enumerate(sorted_source_palette):
            replacement_color_tensor = final_replacement_palette[i]

            # Находим пиксели, соответствующие этому исходному цвету с допуском
            # Используем евклидово расстояние или L1 для скорости? L1 (sum(abs(diff))) проще.
            diff = torch.abs(image_chw - src_color_tensor.view(1, C, 1, 1)) # B, C, H, W
            # mask = torch.sum(diff, dim=1) < tolerance # Сумма абсолютных разностей < tolerance?
            # Более надежно - L2 расстояние (корень не нужен, сравним квадрат)
            dist_sq = torch.sum(diff**2, dim=1) # B, H, W
            mask = dist_sq < (tolerance**2) # Маска [B, H, W]

            # Применяем замену там, где маска True
            # .unsqueeze(1) добавляет обратно измерение канала для маски
            output_image_chw = torch.where(mask.unsqueeze(1).expand_as(output_image_chw),
                                           replacement_color_tensor.view(1, C, 1, 1).expand_as(output_image_chw),
                                           output_image_chw)
            replaced_count += mask.sum().item() # Считаем замененные пиксели

        print(f"Replacement finished. Approximately {replaced_count} pixels potentially replaced.")


        # --- Выход ---
        # Конвертируем обратно в формат ComfyUI (B, H, W, C)
        final_image_hwc = output_image_chw.permute(0, 2, 3, 1).clamp(0, 1)

        return (final_image_hwc,)
