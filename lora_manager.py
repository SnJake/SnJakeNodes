import os, json, hashlib, logging, mimetypes
from pathlib import Path
from aiohttp import web

import server
import folder_paths
import comfy.sd
import comfy.utils
try:
    from safetensors import safe_open as _safetensors_safe_open
except Exception:
    _safetensors_safe_open = None

logger = logging.getLogger("ComfyUI.LoRAManagerWithPreview")

# --- Настройки превью ---
PREVIEW_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp', '.gif']

# Sidecar JSON file name for metadata/description compatibility (Automatic1111-style)
def _sidecar_json_path(lora_abs_path: str) -> str:
    base, _ = os.path.splitext(lora_abs_path)
    return base + ".json"

def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"[LoRAManager] failed reading JSON {path}: {e}")
        return {}

def _write_json(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _load_safetensors_metadata(file_abs_path: str):
    if _safetensors_safe_open is None:
        return None
    try:
        if not file_abs_path.lower().endswith('.safetensors'):
            return None
        with _safetensors_safe_open(file_abs_path, framework="pt", device="cpu") as f:
            return f.metadata() or {}
    except Exception as e:
        logger.debug(f"[LoRAManager] safetensors metadata read failed {file_abs_path}: {e}")
        return None

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

# ---------- HTTP: Get/Save LoRA sidecar info (description + metadata compatibility) ----------
@server.PromptServer.instance.routes.get("/lora_loader_preview/get_lora_info")
async def get_lora_info(request):
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

        base_name = os.path.splitext(os.path.basename(lora_abs_path))[0]
        lora_dir = os.path.dirname(lora_abs_path)

        # Preview existence
        preview_exists = False
        preview_ext = None
        for ext in PREVIEW_IMAGE_EXTENSIONS:
            cand = os.path.join(lora_dir, base_name + ext)
            if os.path.isfile(cand) and _is_under_any(lora_base_paths, cand):
                preview_exists = True
                preview_ext = ext
                break

        import urllib.parse
        preview_url = None
        if preview_exists:
            encoded = urllib.parse.quote(lora_relative_path)
            preview_url = f"/lora_loader_preview/get_preview?lora_path={encoded}"

        # Sidecar JSON
        sidecar_path = _sidecar_json_path(lora_abs_path)
        sidecar = _read_json(sidecar_path)

        # Description resolution order: our namespaced, common keys, safetensors comment
        description = (
            (sidecar.get('snjake', {}) or {}).get('description')
            or sidecar.get('description')
            or sidecar.get('notes')
            or sidecar.get('comment')
        )

        # Metadata from safetensors, if any
        st_meta = _load_safetensors_metadata(lora_abs_path) or {}
        # Also expose any sidecar metadata keys for compatibility; don't override safetensors keys
        merged_meta = dict(sidecar)
        if 'snjake' in merged_meta:
            merged_meta.pop('snjake', None)
        # Merge: safetensors metadata takes precedence
        merged_meta.update({k: v for k, v in st_meta.items() if k not in merged_meta})

        payload = {
            "name": os.path.basename(lora_abs_path),
            "path": lora_relative_path,
            "preview_exists": preview_exists,
            "preview_url": preview_url,
            "description": description or "",
            "metadata": merged_meta,
        }
        return server.web.json_response(payload)
    except Exception as e:
        logger.error(f"[get_lora_info] {e}", exc_info=True)
        return web.Response(status=500, text="Internal server error")

@server.PromptServer.instance.routes.post("/lora_loader_preview/save_lora_info")
async def save_lora_info(request):
    try:
        body = await request.json()
    except Exception:
        return web.Response(status=400, text="Invalid JSON")

    lora_relative_path = (body or {}).get('lora_path')
    description = (body or {}).get('description')
    if not lora_relative_path:
        return web.Response(status=400, text="Missing lora_path")

    try:
        lora_abs_path = folder_paths.get_full_path("loras", lora_relative_path)
        if not lora_abs_path or not os.path.isfile(lora_abs_path):
            return web.Response(status=404, text="LoRA file not found")

        lora_base_paths = folder_paths.get_folder_paths("loras")
        if not _is_under_any(lora_base_paths, lora_abs_path):
            return web.Response(status=403, text="Forbidden")

        sidecar_path = _sidecar_json_path(lora_abs_path)
        sidecar = _read_json(sidecar_path)

        # Update common field and our namespaced field for compatibility
        if isinstance(description, str):
            sidecar['description'] = description
            sn = sidecar.get('snjake') or {}
            sn['description'] = description
            sidecar['snjake'] = sn

        # Mark source/version
        sn = sidecar.get('snjake') or {}
        sn['source'] = 'SnJake.LoRAManager'
        sidecar['snjake'] = sn

        _write_json(sidecar_path, sidecar)
        return server.web.json_response({"ok": True})
    except Exception as e:
        logger.error(f"[save_lora_info] {e}", exc_info=True)
        return web.Response(status=500, text="Internal server error")

@server.PromptServer.instance.routes.post("/lora_loader_preview/upload_lora_preview")
async def upload_lora_preview(request):
    import tempfile, time, urllib.parse

    reader = await request.multipart()
    lora_relative_path = None
    tmp_file_path = None
    uploaded_filename = None

    try:
        field = await reader.next()
        while field is not None:
            if field.name == 'lora_path':
                lora_relative_path = (await field.text()).strip()
            elif field.name == 'file':
                uploaded_filename = getattr(field, 'filename', None) or 'preview.png'
                # Stream to a temp file immediately to avoid draining when moving to next part
                with tempfile.NamedTemporaryFile(delete=False) as tmpf:
                    tmp_file_path = tmpf.name
                    while True:
                        chunk = await field.read_chunk()
                        if not chunk:
                            break
                        tmpf.write(chunk)
            field = await reader.next()

        if not lora_relative_path or not tmp_file_path:
            # Cleanup temp if created
            if tmp_file_path and os.path.exists(tmp_file_path):
                try: os.remove(tmp_file_path)
                except Exception: pass
            return web.Response(status=400, text="Missing lora_path or file")

        # Validate LoRA path and target directory
        lora_abs_path = folder_paths.get_full_path("loras", lora_relative_path)
        if not lora_abs_path or not os.path.isfile(lora_abs_path):
            try: os.remove(tmp_file_path)
            except Exception: pass
            return web.Response(status=404, text="LoRA file not found")

        lora_base_paths = folder_paths.get_folder_paths("loras")
        if not _is_under_any(lora_base_paths, lora_abs_path):
            try: os.remove(tmp_file_path)
            except Exception: pass
            return web.Response(status=403, text="Forbidden")

        base, _ = os.path.splitext(lora_abs_path)

        # Determine extension from uploaded filename
        _, ext = os.path.splitext(uploaded_filename)
        ext = (ext or '').lower()
        if ext not in PREVIEW_IMAGE_EXTENSIONS:
            ext = '.png'

        # Remove old previews
        for pext in PREVIEW_IMAGE_EXTENSIONS:
            old = base + pext
            if os.path.isfile(old):
                try: os.remove(old)
                except Exception: pass

        # Move temp file into place with correct extension
        target = base + ext
        try:
            os.replace(tmp_file_path, target)
        except Exception:
            # Fallback: copy+remove
            with open(tmp_file_path, 'rb') as src, open(target, 'wb') as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            try: os.remove(tmp_file_path)
            except Exception: pass

        encoded = urllib.parse.quote(lora_relative_path)
        preview_url = f"/lora_loader_preview/get_preview?lora_path={encoded}&t={int(time.time())}"
        return server.web.json_response({"ok": True, "preview_url": preview_url})
    except Exception as e:
        logger.error(f"[upload_lora_preview] {e}", exc_info=True)
        # Best-effort cleanup
        if tmp_file_path and os.path.exists(tmp_file_path):
            try: os.remove(tmp_file_path)
            except Exception: pass
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

# ---------- HTTP: листинг одной папки (shallow) ----------
@server.PromptServer.instance.routes.get("/lora_loader_preview/list_dir")
async def list_dir(request):
    import urllib.parse
    directory = request.query.get('directory', '/')
    q = (request.query.get('q') or '').strip().lower()

    # нормализуем путь "/*"
    norm_dir = directory.strip('/\\').replace("\\", "/")
    rel_prefix = "" if norm_dir == "" else norm_dir

    lora_base_paths = folder_paths.get_folder_paths("loras")
    dirs, files = [], []
    seen_dirs, seen_files = set(), set()

    for base_path in lora_base_paths:
        abs_base = os.path.abspath(base_path)
        target_abs = abs_base if rel_prefix == "" else os.path.join(abs_base, rel_prefix)
        if not os.path.isdir(target_abs):
            continue
        try:
            with os.scandir(target_abs) as it:
                for entry in it:
                    try:
                        rel_path = Path(entry.path).relative_to(abs_base).as_posix()
                    except ValueError:
                        continue

                    if entry.is_dir():
                        name = os.path.basename(entry.path)
                        if rel_path not in seen_dirs:
                            seen_dirs.add(rel_path)
                            dirs.append({"name": name, "path": rel_path})
                    else:
                        base_name, ext = os.path.splitext(entry.name)
                        if ext.lower() in folder_paths.supported_pt_extensions:
                            if q and q not in entry.name.lower():
                                continue
                            if rel_path in seen_files:
                                continue
                            seen_files.add(rel_path)

                            # превью (если есть файл-сосед по имени)
                            preview_url = None
                            for pext in PREVIEW_IMAGE_EXTENSIONS:
                                if os.path.isfile(os.path.join(os.path.dirname(entry.path), base_name + pext)):
                                    encoded = urllib.parse.quote(rel_path)
                                    preview_url = f"/lora_loader_preview/get_preview?lora_path={encoded}"
                                    break
                            files.append({"name": entry.name, "path": rel_path, "preview_url": preview_url})
        except OSError as e:
            logger.warning(f"[list_dir] {e}")
        except Exception as e:
            logger.error(f"[list_dir] {e}", exc_info=True)

    dirs.sort(key=lambda x: x["name"].lower())
    files.sort(key=lambda x: x["name"].lower())
    cwd = "/" + norm_dir if norm_dir else "/"
    return server.web.json_response({"cwd": cwd, "dirs": dirs, "files": files})


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

