# /ComfyUI/custom_nodes/LoraLoaderWithPreview/lora_loader_preview.py
import os
import server
import folder_paths
import comfy.sd
import comfy.utils
import torch
import nodes # Import nodes module
import urllib.parse # Import urllib
from pathlib import Path # Import Path for robust path handling
import hashlib # For IS_CHANGED
import logging # For logging
# --- НОВЫЕ ИМПОРТЫ ---
from aiohttp import web
import mimetypes # Для определения MIME типа
# --- КОНЕЦ НОВЫХ ИМПОРТОВ ---
# Set up logger for this node
logger = logging.getLogger("ComfyUI.LoraLoaderWithPreview") # Use ComfyUI logging convention

# Helper class to allow any type matching for connections
class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True
    def __ne__(self, _):
        return False

any_type = AlwaysEqualProxy("*")

# Helper function to scan LoRA directories including subdirs recursively
def get_lora_subdirectories_recursive(lora_paths):
    subdirs = set(["/"]) # Start with root (representing the base directories themselves)
    for lora_path in lora_paths:
        abs_lora_path = os.path.abspath(lora_path) # Ensure absolute path
        if os.path.isdir(abs_lora_path):
            try:
                logger.debug(f"[LoraLoaderWithPreview] Scanning subdir base: {abs_lora_path}")
                for root, dirs, _ in os.walk(abs_lora_path, followlinks=True):
                    # Calculate relative path from the base lora directory
                    try:
                        relative_path_obj = Path(root).relative_to(abs_lora_path)
                        # Convert to string with forward slashes, add leading slash
                        relative_path = "/" + relative_path_obj.as_posix()
                        if relative_path == "/.": # Handle root case
                            relative_path = "/"
                        subdirs.add(relative_path)
                        #logger.debug(f"Added subdir: {relative_path}")

                        # Add parent directories as well - ensure root ('/') is handled
                        parent = relative_path_obj.parent
                        while str(parent) != '.': # Stop when we reach the root relative path
                            parent_path = "/" + parent.as_posix()
                            subdirs.add(parent_path)
                            #logger.debug(f"Added parent subdir: {parent_path}")
                            parent = parent.parent
                        subdirs.add("/") # Ensure root is always present

                    except ValueError as e:
                        logger.warning(f"[LoraLoaderWithPreview] Directory '{root}' relative_to error for base '{abs_lora_path}': {e}. Skipping for subdir list.")
                        continue
            except OSError as e:
                 logger.warning(f"[LoraLoaderWithPreview] Error walking directory {abs_lora_path} for subdirs: {e}")

    # Ensure "/" is always first if it exists
    sorted_subdirs = sorted(list(subdirs))
    if "/" in sorted_subdirs:
        sorted_subdirs.remove("/")
        sorted_subdirs.insert(0, "/")

    logger.debug(f"[LoraLoaderWithPreview] Found subdirectories: {sorted_subdirs}")
    return sorted_subdirs

# The main node class
class LoraLoaderWithPreview:
    CATEGORY = "😎 SnJake/Lora Loader"
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora_with_preview"

    def __init__(self):
        self.loaded_lora_path = None
        self.cached_lora_sd = None
        self.last_lora_mtime = None # Store mtime for change detection

    @classmethod
    def INPUT_TYPES(cls):
        try:
            lora_paths = folder_paths.get_folder_paths("loras")
            lora_subdirs = get_lora_subdirectories_recursive(lora_paths)
        except Exception as e:
            logger.error(f"[LoraLoaderWithPreview] Error getting LoRA paths/subdirs: {e}", exc_info=True)
            lora_subdirs = ["/"] # Fallback

        return {
            "required": {
                "model": ("MODEL", ),
                "clip": ("CLIP", ),
                "directory_filter": (lora_subdirs, {"default": "/"}),
                "selected_lora": ("STRING", {"default": "None", "widget": "HIDDEN"}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "lora_preview_widget": ("LORA_PREVIEW", {"default": "None"}), # Placeholder for JS widget
            }
        }

    @classmethod
    def IS_CHANGED(cls, model, clip, directory_filter, selected_lora, strength_model, strength_clip, **kwargs):
        lora_path = folder_paths.get_full_path("loras", selected_lora) if selected_lora not in [None, "None", ""] else None
        lora_details = f"{selected_lora}:{strength_model}:{strength_clip}"

        mtime = "0"
        filesize = "0"
        if lora_path and os.path.exists(lora_path):
            try:
                stats = os.stat(lora_path)
                mtime = str(stats.st_mtime)
                filesize = str(stats.st_size)
            except OSError as e:
                logger.warning(f"[LoraLoaderWithPreview] IS_CHANGED: Could not stat LoRA file {lora_path}: {e}")
                mtime = str(hash(selected_lora)) # Fallback to hashing name if stat fails

        input_hash = hashlib.sha256()
        input_hash.update(lora_details.encode('utf-8'))
        input_hash.update(mtime.encode('utf-8'))
        input_hash.update(filesize.encode('utf-8')) # Add filesize for extra robustness

        return input_hash.hexdigest()

    def load_lora_with_preview(self, model, clip, directory_filter, selected_lora, strength_model, strength_clip, **kwargs):
        if strength_model == 0 and strength_clip == 0 or selected_lora in [None, "None", ""]:
            if self.cached_lora_sd is not None:
                logger.info(f"[LoraLoaderWithPreview] Strengths zero or no LoRA selected. Unloading.")
                self.loaded_lora_path = None
                self.cached_lora_sd = None
                self.last_lora_mtime = None
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", selected_lora)

        if not lora_path:
             logger.warning(f"[LoraLoaderWithPreview] Path not found for '{selected_lora}'. Returning original models.")
             self.loaded_lora_path = None
             self.cached_lora_sd = None
             self.last_lora_mtime = None
             return (model, clip)

        load_new_lora = False
        current_mtime = None
        try:
             current_mtime = os.path.getmtime(lora_path)
        except OSError as e:
             logger.warning(f"[LoraLoaderWithPreview] Could not get mtime for {lora_path}: {e}. Will attempt to load.")
             load_new_lora = True # Reload if we can't check mtime

        if self.cached_lora_sd is None or self.loaded_lora_path != lora_path:
            load_new_lora = True
            logger.info(f"[LoraLoaderWithPreview] Cache miss or different LoRA selected ('{self.loaded_lora_path}' vs '{lora_path}').")
        elif current_mtime is not None and self.last_lora_mtime != current_mtime:
            load_new_lora = True
            logger.info(f"[LoraLoaderWithPreview] LoRA file changed ({selected_lora}), reloading.")


        if load_new_lora:
            try:
                logger.info(f"[LoraLoaderWithPreview] Loading LoRA '{selected_lora}' from {lora_path}")
                # Force device to CPU before loading to prevent potential VRAM issues with many LoRAs
                self.cached_lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True, device="cpu")
                self.loaded_lora_path = lora_path
                self.last_lora_mtime = current_mtime
            except Exception as e:
                logger.error(f"[LoraLoaderWithPreview] Error loading LoRA {lora_path}: {e}", exc_info=True)
                self.loaded_lora_path = None
                self.cached_lora_sd = None
                self.last_lora_mtime = None
                return (model, clip) # Return original models on error

        if self.cached_lora_sd is None:
             logger.error(f"[LoraLoaderWithPreview] LoRA data is None after loading attempt for {selected_lora}. Returning original models.")
             return (model, clip)

        logger.debug(f"[LoraLoaderWithPreview] Applying LoRA '{selected_lora}' with strengths ({strength_model}, {strength_clip})")
        try:
            # Model and CLIP are already on their correct devices, LoRA is on CPU
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, self.cached_lora_sd, strength_model, strength_clip)
            return (model_lora, clip_lora)
        except Exception as e:
            logger.error(f"[LoraLoaderWithPreview] Error *applying* LoRA {selected_lora}: {e}", exc_info=True)
            return (model, clip) # Return original models if application fails

# --- API Endpoint ---
PREVIEW_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp', '.gif']

# --- НОВЫЙ ЭНДПОИНТ ДЛЯ ПРЕВЬЮ ---
@server.PromptServer.instance.routes.get("/lora_loader_preview/get_preview")
async def get_lora_preview(request):
    lora_relative_path = request.query.get('lora_path')
    if not lora_relative_path:
        return web.Response(status=400, text="Missing lora_path parameter")

    try:
        # 1. Получить абсолютный путь к LoRA
        # Важно: get_full_path сам ищет по всем базовым путям
        lora_abs_path = folder_paths.get_full_path("loras", lora_relative_path)

        if not lora_abs_path or not os.path.isfile(lora_abs_path):
            logger.warning(f"[GetPreview] LoRA file not found for relative path: {lora_relative_path}")
            return web.Response(status=404, text="LoRA file not found")

        # 2. Проверка безопасности: Убедиться, что путь внутри разрешенных директорий LoRA
        lora_base_paths = folder_paths.get_folder_paths("loras")
        is_safe = False
        # Нормализуем путь к файлу для надежного сравнения
        normalized_lora_abs_path = os.path.normpath(lora_abs_path)
        for base_path in lora_base_paths:
            abs_base_path = os.path.abspath(base_path)
            normalized_base_path = os.path.normpath(abs_base_path)
            # Проверяем, что абсолютный путь LoRA начинается с одного из базовых путей
            if normalized_lora_abs_path.startswith(normalized_base_path + os.sep): # Используем os.sep для кроссплатформенности
                is_safe = True
                break

        if not is_safe:
             # Логируем подозрительный путь перед возвратом ошибки
             logger.warning(f"[GetPreview] Security check failed. Path: {normalized_lora_abs_path}. Allowed bases: {lora_base_paths}")
             return web.Response(status=403, text="Forbidden: Access denied to the specified file path.")


        # 3. Найти файл превью
        lora_dir = os.path.dirname(lora_abs_path)
        base_name, _ = os.path.splitext(os.path.basename(lora_abs_path))
        preview_filepath_abs = None
        for preview_ext in PREVIEW_IMAGE_EXTENSIONS:
            potential_preview_path = os.path.join(lora_dir, base_name + preview_ext)
            # Дополнительная проверка, что превью находится в той же папке и является файлом
            if os.path.exists(potential_preview_path) and os.path.isfile(potential_preview_path):
                 # Дополнительная проверка безопасности для превью (хотя должна быть покрыта проверкой LoRA)
                 normalized_preview_path = os.path.normpath(potential_preview_path)
                 preview_is_safe = False
                 for base_path in lora_base_paths:
                     normalized_base_path = os.path.normpath(os.path.abspath(base_path))
                     if normalized_preview_path.startswith(normalized_base_path + os.sep):
                         preview_is_safe = True
                         break
                 if preview_is_safe:
                    preview_filepath_abs = potential_preview_path
                    break
                 else:
                    logger.warning(f"[GetPreview] Preview file path check failed for {potential_preview_path}")


        if not preview_filepath_abs:
            logger.warning(f"[GetPreview] Preview image not found for LoRA: {lora_relative_path} (Searched in {lora_dir})")
            return web.Response(status=404, text="Preview not found")

        # 4. Вернуть файл
        mime_type, _ = mimetypes.guess_type(preview_filepath_abs)
        if not mime_type:
            mime_type = 'application/octet-stream' # Fallback

        logger.debug(f"[GetPreview] Sending preview file: {preview_filepath_abs} with MIME type: {mime_type}")
        # Используем FileResponse для эффективной отдачи файла
        # aiohttp обработает заголовки Content-Type, ETag и т.д.
        return web.FileResponse(preview_filepath_abs)

    except Exception as e:
        logger.error(f"[GetPreview] Error getting preview for {lora_relative_path}: {e}", exc_info=True)
        return web.Response(status=500, text="Internal server error")

@server.PromptServer.instance.routes.get("/lora_loader_preview/list_loras")
async def list_loras_with_previews(request):
    directory_filter = request.query.get('directory_filter', '/')
    if directory_filter in [None, "None", ""]: directory_filter = "/"

    norm_filter = directory_filter.strip('/\\').replace("\\", "/")
    if directory_filter == "/": norm_filter = "" # Root directory matches everything

    logger.debug(f"[ListLoras] API Call - Filter: '{directory_filter}' (Normalized: '{norm_filter}')")

    lora_items = []
    lora_base_paths = folder_paths.get_folder_paths("loras")
    # --- ИЗМЕНЕНИЕ: Ключ - относительный путь, Значение - абсолютный путь (для дедупликации) ---
    # Мы храним абсолютный путь, чтобы гарантировать, что найдем превью там же, где и LoRA
    seen_loras = {}

    logger.debug(f"[ListLoras] Using LoRA base paths: {lora_base_paths}")

    for base_path in lora_base_paths:
        abs_base_path = os.path.abspath(base_path)
        if not os.path.isdir(abs_base_path):
            logger.warning(f"[ListLoras] Skipping invalid base path (not a directory): {abs_base_path}")
            continue

        # logger.debug(f"[ListLoras] Scanning base path: {abs_base_path}")
        try:
            for root, _, files in os.walk(abs_base_path, followlinks=True):
                try:
                    # Относительная директория для фильтрации (относительно ТЕКУЩЕГО base_path)
                    relative_dir_obj = Path(root).relative_to(abs_base_path)
                    relative_dir_for_filter = relative_dir_obj.as_posix()
                    if relative_dir_for_filter == ".": relative_dir_for_filter = ""
                except ValueError:
                    # logger.warning(f"[ListLoras] Could not get relative path for filter in {root} from {abs_base_path}, skipping dir.")
                    continue # Пропускаем директории не относительно базового пути

                # Применяем фильтр
                should_include_dir = False
                if norm_filter == "": # Корневой фильтр включает все
                    should_include_dir = True
                elif relative_dir_for_filter == norm_filter or relative_dir_for_filter.startswith(norm_filter + '/'):
                    should_include_dir = True

                if should_include_dir:
                    for filename in files:
                        filepath_abs = os.path.join(root, filename) # Абсолютный путь к файлу LoRA
                        if os.path.isfile(filepath_abs):
                            base_name, ext = os.path.splitext(filename)
                            if ext.lower() in folder_paths.supported_pt_extensions:
                                try:
                                    # --- ПРАВИЛЬНЫЙ ОТНОСИТЕЛЬНЫЙ ПУТЬ ДЛЯ ВЫБОРА/ID ---
                                    # Этот путь будет 'subdir/lora.safetensors' относительно ТЕКУЩЕГО base_path
                                    # Это тот путь, который ожидает твой JS и Python backend для загрузки
                                    lora_relative_path_for_selection = Path(filepath_abs).relative_to(abs_base_path).as_posix()

                                    # --- Проверка на дубликаты по этому относительному пути ---
                                    # Если LoRA с таким относительным путем уже найдена в *другой* базовой папке,
                                    # пропускаем её, т.к. get_full_path все равно найдет первую.
                                    if lora_relative_path_for_selection in seen_loras:
                                        # logger.debug(f"[ListLoras] Skipping duplicate relative path: {lora_relative_path_for_selection}")
                                        continue
                                    # Сохраняем абсолютный путь, чтобы точно знать, где искать превью
                                    seen_loras[lora_relative_path_for_selection] = filepath_abs
                                    # --- Конец изменения ---

                                    preview_url = None # URL для нашего нового эндпоинта
                                    has_preview = False
                                    # Ищем превью рядом с файлом LoRA (используем сохраненный абсолютный путь)
                                    lora_dir = os.path.dirname(filepath_abs) # Директория, где лежит LoRA
                                    preview_base_name, _ = os.path.splitext(os.path.basename(filepath_abs)) # Имя без расширения

                                    for preview_ext in PREVIEW_IMAGE_EXTENSIONS:
                                        preview_filename_check = preview_base_name + preview_ext
                                        preview_filepath_abs_check = os.path.join(lora_dir, preview_filename_check)
                                        if os.path.exists(preview_filepath_abs_check) and os.path.isfile(preview_filepath_abs_check):
                                            has_preview = True
                                            break # Нашли превью, выходим

                                    if has_preview:
                                        try:
                                            # Кодируем относительный путь LoRA для URL
                                            encoded_lora_path = urllib.parse.quote(lora_relative_path_for_selection)
                                            preview_url = f"/lora_loader_preview/get_preview?lora_path={encoded_lora_path}"
                                            # logger.debug(f"[ListLoras DEBUG] Lora: {lora_relative_path_for_selection}, Found preview, Generated URL: {preview_url}")
                                        except Exception as e:
                                             logger.error(f"[ListLoras] Error generating URL for custom endpoint for lora {lora_relative_path_for_selection}: {e}")
                                             preview_url = None

                                    # Добавляем элемент в список
                                    lora_items.append({
                                        "name": filename, # Имя файла для отображения
                                        "path": lora_relative_path_for_selection, # Относительный путь для загрузки
                                        "preview_url": preview_url # URL для нашего нового эндпоинта
                                    })
                                except ValueError:
                                     # Эта ошибка возникает, если filepath_abs не находится внутри abs_base_path
                                     logger.warning(f"[ListLoras] Could not get relative path for selection for {filepath_abs} from {abs_base_path}, skipping.")
                                     continue
        except OSError as e:
            logger.warning(f"[ListLoras] Error walking directory {abs_base_path}: {e}")
        except Exception as e:
            logger.error(f"[ListLoras] Unexpected error scanning {abs_base_path}: {e}", exc_info=True)

    # Сортировка и возврат
    lora_items.sort(key=lambda x: x['name'].lower())
    logger.debug(f"[ListLoras] API Response Count: {len(lora_items)}")
    return server.web.json_response(lora_items)