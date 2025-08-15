import os, json, hashlib, logging, mimetypes
from pathlib import Path
from aiohttp import web

import server
import folder_paths
import comfy.sd
import comfy.utils

logger = logging.getLogger("ComfyUI.LoRAManagerWithPreview")

# --- Настройки превью ---
PREVIEW_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp', '.gif']

# ---------- ХЕЛПЕРЫ: поиск подпапок и безопасные пути ----------
def get_lora_subdirectories_recursive(lora_paths):
    subdirs = set(["/"])
    for lora_path in lora_paths:
        abs_lora_path = os.path.abspath(lora_path)
        if os.path.isdir(abs_lora_path):
            try:
                for root, dirs, _ in os.walk(abs_lora_path, followlinks=True):
                    try:
                        relative_path_obj = Path(root).relative_to(abs_lora_path)
                        relative_path = "/" + relative_path_obj.as_posix()
                        if relative_path == "/.": relative_path = "/"
                        subdirs.add(relative_path)
                        parent = relative_path_obj.parent
                        while str(parent) != '.':
                            parent_path = "/" + parent.as_posix()
                            subdirs.add(parent_path)
                            parent = parent.parent
                        subdirs.add("/")
                    except ValueError:
                        continue
            except OSError as e:
                logger.warning(f"[LoRAManager] walk error {abs_lora_path}: {e}")

    sorted_subdirs = sorted(list(subdirs))
    if "/" in sorted_subdirs:
        sorted_subdirs.remove("/")
        sorted_subdirs.insert(0, "/")
    return sorted_subdirs

def _is_under_any(base_paths, abs_path):
    norm = os.path.normpath(abs_path)
    for b in base_paths:
        nb = os.path.normpath(os.path.abspath(b))
        if norm.startswith(nb + os.sep):
            return True
    return False

# ---------- HTTP: превью и список LoRA (оставляем, как в твоей версии) ----------
@server.PromptServer.instance.routes.get("/lora_loader_preview/get_preview")
async def get_lora_preview(request):
    lora_relative_path = request.query.get('lora_path')
    if not lora_relative_path:
        return web.Response(status=400, text="Missing lora_path")

    try:
        lora_abs_path = folder_paths.get_full_path("loras", lora_relative_path)
        if not lora_abs_path or not os.path.isfile(lora_abs_path):
            return web.Response(status=404, text="LoRA file not found")

        lora_base_paths = folder_paths.get_folder_paths("loras")
        if not _is_under_any(lora_base_paths, lora_abs_path):
            return web.Response(status=403, text="Forbidden")

        lora_dir = os.path.dirname(lora_abs_path)
        base_name, _ = os.path.splitext(os.path.basename(lora_abs_path))

        preview_filepath_abs = None
        for ext in PREVIEW_IMAGE_EXTENSIONS:
            cand = os.path.join(lora_dir, base_name + ext)
            if os.path.exists(cand) and os.path.isfile(cand) and _is_under_any(lora_base_paths, cand):
                preview_filepath_abs = cand
                break

        if not preview_filepath_abs:
            return web.Response(status=404, text="Preview not found")

        mime_type, _ = mimetypes.guess_type(preview_filepath_abs)
        if not mime_type: mime_type = 'application/octet-stream'
        return web.FileResponse(preview_filepath_abs, headers={"Content-Type": mime_type})
    except Exception as e:
        logger.error(f"[get_preview] {e}", exc_info=True)
        return web.Response(status=500, text="Internal server error")

@server.PromptServer.instance.routes.get("/lora_loader_preview/list_loras")
async def list_loras_with_previews(request):
    import urllib.parse
    directory_filter = request.query.get('directory_filter', '/')
    if directory_filter in [None, "None", ""]: directory_filter = "/"

    norm_filter = directory_filter.strip('/\\').replace("\\", "/")
    if directory_filter == "/": norm_filter = ""

    lora_items = []
    lora_base_paths = folder_paths.get_folder_paths("loras")
    seen = {}

    for base_path in lora_base_paths:
        abs_base = os.path.abspath(base_path)
        if not os.path.isdir(abs_base): continue
        try:
            for root, _, files in os.walk(abs_base, followlinks=True):
                try:
                    rel_dir_obj = Path(root).relative_to(abs_base)
                    rel_dir_for_filter = rel_dir_obj.as_posix()
                    if rel_dir_for_filter == ".": rel_dir_for_filter = ""
                except ValueError:
                    continue

                include = (norm_filter == "") or (rel_dir_for_filter == norm_filter) or rel_dir_for_filter.startswith(norm_filter + "/")
                if not include: continue

                for filename in files:
                    filepath_abs = os.path.join(root, filename)
                    if not os.path.isfile(filepath_abs): continue
                    base_name, ext = os.path.splitext(filename)
                    if ext.lower() in folder_paths.supported_pt_extensions:
                        try:
                            rel_path_for_selection = Path(filepath_abs).relative_to(abs_base).as_posix()
                            if rel_path_for_selection in seen: continue
                            seen[rel_path_for_selection] = filepath_abs

                            # preview?
                            has_preview = False
                            lora_dir = os.path.dirname(filepath_abs)
                            for pext in PREVIEW_IMAGE_EXTENSIONS:
                                if os.path.isfile(os.path.join(lora_dir, base_name + pext)):
                                    has_preview = True
                                    break

                            preview_url = None
                            if has_preview:
                                encoded = urllib.parse.quote(rel_path_for_selection)
                                preview_url = f"/lora_loader_preview/get_preview?lora_path={encoded}"

                            lora_items.append({
                                "name": filename,
                                "path": rel_path_for_selection,
                                "preview_url": preview_url
                            })
                        except ValueError:
                            continue
        except OSError as e:
            logger.warning(f"[list_loras] {e}")
        except Exception as e:
            logger.error(f"[list_loras] {e}", exc_info=True)

    lora_items.sort(key=lambda x: x["name"].lower())
    return server.web.json_response(lora_items)

# ---------- ОСНОВНАЯ НОДА: мульти-стек LoRA ----------
class LoRAManagerWithPreview:
    CATEGORY = "😎 SnJake/LoRA"
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply_loras"

    _cache = {}  # path -> {"sd": state_dict, "mtime": float, "size": int}

    @classmethod
    def INPUT_TYPES(cls):
        try:
            lora_paths = folder_paths.get_folder_paths("loras")
            lora_subdirs = get_lora_subdirectories_recursive(lora_paths)
        except Exception:
            lora_subdirs = ["/"]

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "directory_filter": (lora_subdirs, {"default": "/"}),
                # Храним стек в JSON, UI рисуем на фронте
                "lora_stack_json": ("STRING", {"default": "[]", "widget": "HIDDEN"}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, model, clip, directory_filter, lora_stack_json, **kwargs):
        # Хешируем содержимое + mtime/size каждого файла для корректного инвалида
        try:
            items = json.loads(lora_stack_json or "[]")
        except Exception:
            items = []

        h = hashlib.sha256()
        h.update((directory_filter or "/").encode("utf-8"))
        h.update(repr(items).encode("utf-8"))

        for it in items:
            p = it.get("path")
            if not p: continue
            full = folder_paths.get_full_path("loras", p)
            mtime = "0"; size = "0"
            if full and os.path.exists(full):
                try:
                    st = os.stat(full)
                    mtime = str(st.st_mtime); size = str(st.st_size)
                except OSError:
                    pass
            h.update(p.encode("utf-8"))
            h.update(mtime.encode("utf-8"))
            h.update(size.encode("utf-8"))
        return h.hexdigest()

    def _load_lora_sd(self, lora_path_abs):
        try:
            st = os.stat(lora_path_abs)
            key = os.path.normpath(lora_path_abs)
            cached = self._cache.get(key)
            if cached and cached.get("mtime") == st.st_mtime and cached.get("size") == st.st_size and cached.get("sd") is not None:
                return cached["sd"]
    
            # БЕЗ device=...
            sd = comfy.utils.load_torch_file(lora_path_abs, safe_load=True)
            self._cache[key] = {"sd": sd, "mtime": st.st_mtime, "size": st.st_size}
            return sd
        except Exception as e:
            logger.error(f"[LoRAManager] load failed {lora_path_abs}: {e}", exc_info=True)
            return None

    def apply_loras(self, model, clip, directory_filter, lora_stack_json, **kwargs):
        # Пусто — ничего не делаем
        try:
            items = json.loads(lora_stack_json or "[]")
            # формат: [{"path": "...", "name": "...", "strength_model": 1.0, "strength_clip": 1.0}, ...]
        except Exception as e:
            logger.error(f"[LoRAManager] bad JSON: {e}")
            return (model, clip)

        if not isinstance(items, list) or len(items) == 0:
            return (model, clip)

        m, c = model, clip
        for it in items:
            p = it.get("path")
            if not p: continue
            sm = float(it.get("strength_model", 1.0))
            sc = float(it.get("strength_clip", 1.0))

            # Если обе силы = 0 — пропускаем
            if sm == 0.0 and sc == 0.0:
                continue

            full = folder_paths.get_full_path("loras", p)
            if not full or not os.path.isfile(full):
                logger.warning(f"[LoRAManager] not found: {p}")
                continue

            sd = self._load_lora_sd(full)
            if sd is None:
                continue

            try:
                m, c = comfy.sd.load_lora_for_models(m, c, sd, sm, sc)
            except Exception as e:
                logger.error(f"[LoRAManager] apply failed {p}: {e}", exc_info=True)
                continue

        return (m, c)

