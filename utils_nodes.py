import os
import glob
import random
import re
from io import BytesIO
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import numpy as np
import torch
import json

from pathlib import Path

from comfy_execution.graph import ExecutionBlocker
from comfy_execution.graph_utils import GraphBuilder

try:
    import av
except ImportError:
    av = None


AUDIO_FILE_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".aif",
    ".aifc",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}

AUDIO_OUTPUT_FORMATS = {
    ".flac": ("flac", "flac"),
    ".mp3": ("mp3", "libmp3lame"),
    ".ogg": ("ogg", "libopus"),
    ".opus": ("opus", "libopus"),
    ".wav": ("wav", "pcm_s16le"),
}

OPUS_SAMPLE_RATES = {8000, 12000, 16000, 24000, 48000}


def _natural_sort_key(file_path):
    filename = os.path.basename(file_path).casefold()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", filename)]


def _make_json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.hex()
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(v) for v in value]
    return str(value)


def _metadata_to_json_string(metadata):
    if not isinstance(metadata, dict) or not metadata:
        return ""

    safe_metadata = {str(k): _make_json_safe(v) for k, v in metadata.items()}
    try:
        return json.dumps(safe_metadata, ensure_ascii=False)
    except Exception:
        return str(safe_metadata)


def _try_parse_json_object(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def _extract_prompt_from_comfy_prompt(prompt_obj):
    if not isinstance(prompt_obj, dict):
        return ""

    preferred = []
    fallback = []
    for node_data in prompt_obj.values():
        if not isinstance(node_data, dict):
            continue

        class_type = str(node_data.get("class_type", ""))
        inputs = node_data.get("inputs", {})
        if not isinstance(inputs, dict):
            continue

        text_val = inputs.get("text", None)
        if isinstance(text_val, str) and text_val.strip():
            if "CLIPTextEncode" in class_type:
                preferred.append(text_val.strip())
            else:
                fallback.append(text_val.strip())

    if preferred:
        return preferred[0]
    if fallback:
        return fallback[0]
    return ""


def _extract_prompt_from_workflow(workflow_obj):
    nodes = workflow_obj.get("nodes", None)
    if not isinstance(nodes, list):
        return ""

    preferred = []
    fallback = []
    for node in nodes:
        if not isinstance(node, dict):
            continue

        node_type = str(node.get("type", ""))
        widgets_values = node.get("widgets_values", None)
        if not isinstance(widgets_values, list):
            continue

        string_values = [v.strip() for v in widgets_values if isinstance(v, str) and v.strip()]
        if not string_values:
            continue

        if "CLIPTextEncode" in node_type:
            preferred.extend(string_values)
        else:
            fallback.extend(string_values)

    if preferred:
        return preferred[0]
    if fallback:
        return fallback[0]
    return ""


def _extract_prompt_text_from_metadata(metadata):
    if not isinstance(metadata, dict):
        return ""

    prompt_obj = _try_parse_json_object(metadata.get("prompt", None))
    if isinstance(prompt_obj, dict):
        extracted = _extract_prompt_from_comfy_prompt(prompt_obj)
        if extracted:
            return extracted

    workflow_obj = _try_parse_json_object(metadata.get("workflow", None))
    if isinstance(workflow_obj, dict):
        extracted = _extract_prompt_from_workflow(workflow_obj)
        if extracted:
            return extracted

    for key in ("prompt", "parameters", "Description", "description", "Comment", "comment"):
        raw_value = metadata.get(key, None)
        if raw_value is None:
            continue

        text_value = str(raw_value).strip()
        if not text_value:
            continue

        if key == "parameters":
            return text_value.split("Negative prompt:", 1)[0].strip()
        return text_value

    return ""


def _audio_to_f32_pcm(waveform):
    if waveform.dtype.is_floating_point:
        return waveform.float()
    if waveform.dtype == torch.int16:
        return waveform.float() / (2 ** 15)
    if waveform.dtype == torch.int32:
        return waveform.float() / (2 ** 31)
    if waveform.dtype == torch.uint8:
        return (waveform.float() - 128.0) / 128.0
    return waveform.float()


def _load_audio_file(file_path):
    if av is None:
        raise RuntimeError("PyAV is not available. Install ComfyUI audio dependencies first.")

    with av.open(file_path) as container:
        if not container.streams.audio:
            raise ValueError("No audio stream found in the file.")

        stream = container.streams.audio[0]
        sample_rate = getattr(stream.codec_context, "sample_rate", None) or getattr(stream, "sample_rate", None)
        n_channels = getattr(stream, "channels", None) or 1
        metadata = dict(container.metadata or {})

        frames = []
        for frame in container.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())

            if buf.ndim == 1:
                buf = buf.unsqueeze(0)
            elif buf.shape[0] != n_channels:
                buf = buf.view(-1, n_channels).t()

            frames.append(buf)

        if not frames:
            raise ValueError("No audio frames decoded.")

        waveform = torch.cat(frames, dim=1)
        waveform = _audio_to_f32_pcm(waveform)
        return {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate)}, metadata


def _save_audio_file(audio, output_path, metadata=None):
    if av is None:
        raise RuntimeError("PyAV is not available. Install ComfyUI audio dependencies first.")
    if not isinstance(audio, dict):
        raise ValueError("Audio input must be a ComfyUI AUDIO dict.")

    waveform_batch = audio.get("waveform", None)
    sample_rate = int(audio.get("sample_rate", 0) or 0)
    if waveform_batch is None:
        raise ValueError("Audio dict does not contain 'waveform'.")
    if sample_rate <= 0:
        raise ValueError("Audio dict does not contain a valid 'sample_rate'.")
    if waveform_batch.ndim != 3 or waveform_batch.shape[0] < 1:
        raise ValueError("Audio waveform must have shape [B, C, T].")

    waveform = waveform_batch[0].detach().cpu().contiguous()
    if waveform.ndim != 2:
        raise ValueError("Single audio item must have shape [C, T].")
    if waveform.shape[0] not in (1, 2):
        raise ValueError("Only mono and stereo audio are supported for saving.")

    suffix = output_path.suffix.lower()
    if suffix not in AUDIO_OUTPUT_FORMATS:
        supported = ", ".join(sorted(AUDIO_OUTPUT_FORMATS.keys()))
        raise ValueError(f"Unsupported audio extension '{suffix}'. Supported: {supported}")

    container_format, codec = AUDIO_OUTPUT_FORMATS[suffix]
    if codec == "libopus" and sample_rate not in OPUS_SAMPLE_RATES:
        supported_rates = ", ".join(str(x) for x in sorted(OPUS_SAMPLE_RATES))
        raise ValueError(
            f"Opus/Ogg export requires sample_rate in [{supported_rates}]. Current: {sample_rate}"
        )

    layout = "mono" if waveform.shape[0] == 1 else "stereo"

    output_buffer = BytesIO()
    output_container = av.open(output_buffer, mode="w", format=container_format)
    try:
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                output_container.metadata[str(key)] = str(value)

        out_stream = output_container.add_stream(codec, rate=sample_rate, layout=layout)
        if codec == "libmp3lame":
            out_stream.bit_rate = 320000
        elif codec == "libopus":
            out_stream.bit_rate = 192000

        frame = av.AudioFrame.from_ndarray(
            waveform.movedim(0, 1).reshape(1, -1).float().numpy(),
            format="flt",
            layout=layout,
        )
        frame.sample_rate = sample_rate
        frame.pts = 0

        for packet in out_stream.encode(frame):
            output_container.mux(packet)
        for packet in out_stream.encode(None):
            output_container.mux(packet)
    finally:
        output_container.close()

    output_buffer.seek(0)
    with open(output_path, "wb") as f:
        f.write(output_buffer.getbuffer())

class BatchLoadImages:
    """
    ???? ??? ???????? ???????? ??????????? ?? ????? ??? ???????????.
    ????????????? ????? ???????? RAW-?????????? ? ??????????? prompt.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["single_image", "incremental_image", "random"], {}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32-1}),
                "index": ("INT", {"default": 0, "min": 0, "max": 150000}),
                "label": ("STRING", {"default": "Batch 001"}),
                "path": ("STRING", {"default": ""}),
                "pattern": ("STRING", {"default": "*"}),
                "allow_RGBA_output": (["false", "true"], {"default": "false"}),

                # ????????????? "?? ?????"
                "allow_cycle": (["true", "false"], {"default":"true", "label_on":"Cycle On", "label_off":"Cycle Off"}),
            },
            "optional": {
                "filename_text_extension": (["true", "false"], {"default":"true"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "filename_text", "raw_metadata", "prompt_text")
    FUNCTION = "load_batch_images"
    CATEGORY = "😎 SnJake/Utils"

    # Internal state for incremental mode.
    # label -> next index to use
    incremental_counters = {}
    # label -> last seen seed
    incremental_last_seed = {}

    def load_batch_images(
        self,
        path,
        pattern="*",
        index=0,
        mode="single_image",
        seed=0,
        label="Batch 001",
        allow_RGBA_output="false",
        filename_text_extension="true",
        allow_cycle="true",
        unique_id=None,
        extra_pnginfo=None
    ):
        # 1) ???????? ?????? ???????? ??? ???????????
        all_files = self._scan_directory(path, pattern)
        if not all_files:
            print(f"[BatchLoadImages] ????? '{path}' ????? ??? ??? ?????????? ?????? ?? ???????? '{pattern}'")
            return (None, None, "", "")

        # 2) ?????? ?????? ???????
        if mode == "single_image":
            if index < 0 or index >= len(all_files):
                print(f"[BatchLoadImages] ???????? index={index}, ?? ? ????? ?????? {len(all_files)} ??????.")
                return (None, None, "", "")
            chosen_index = index

        elif mode == "incremental_image":
            if label not in self.incremental_counters:
                self.incremental_counters[label] = 0

            last_seed = self.incremental_last_seed.get(label, None)
            # If seed was changed manually (jumped or moved back), sync sequence to seed.
            # This makes "seed incremental" deterministic and allows reset to 0 -> first image.
            if last_seed is None:
                if seed > 0:
                    self.incremental_counters[label] = seed
            elif seed < last_seed or seed > (last_seed + 1):
                self.incremental_counters[label] = seed

            chosen_index = self.incremental_counters[label]

            # ???? ???????? ?????
            if chosen_index >= len(all_files):
                if allow_cycle == "true":
                    # ?? ?????: ?????????? ? 0
                    print(f"[BatchLoadImages] ??? label='{label}' ?????? ?????? ????? ({chosen_index}). ?????????? ? 0 (cycling).")
                    chosen_index = 0
                    self.incremental_counters[label] = 0
                else:
                    # ??????? ? ???????
                    print(f"[BatchLoadImages] ??????????? ? ????? ??????????? ??? label='{label}'. ???????????????.")
                    return (None, None, "", "")

            # ??????? ?????? ?? ????????? ??????
            self.incremental_counters[label] += 1
            self.incremental_last_seed[label] = seed

        else:  # mode == 'random'
            random.seed(seed)
            chosen_index = random.randint(0, len(all_files) - 1)

        # 3) ????????? ????????
        img_path = all_files[chosen_index]
        image_tensor = self._load_as_tensor(img_path, allow_RGBA_output == "true")

        # 4) ???? ????? ?????? ?????????? ? filename
        filename = os.path.basename(img_path)
        if filename_text_extension == "false":
            filename = os.path.splitext(filename)[0]

        # 5) ?????? ?????????? ?????? ???? ???? ?? ???? ?? ????? ??????? ?????????.
        #    ???? ?????????? ??????????? ?? ???????, ?????? (fallback ? ??????? ????????????).
        raw_metadata_text = ""
        prompt_text = ""

        raw_out_connected = self._is_output_connected(extra_pnginfo, unique_id, 2)
        prompt_out_connected = self._is_output_connected(extra_pnginfo, unique_id, 3)
        should_read_metadata = not (raw_out_connected is False and prompt_out_connected is False)

        if should_read_metadata:
            raw_metadata_text, prompt_text = self._read_metadata_and_prompt(img_path)

        return (image_tensor, filename, raw_metadata_text, prompt_text)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        ???? mode != single_image, ?????????? NaN, ????? ComfyUI ?????? ??????????????
        (????? "random"/"incremental" ????? ?? ??????????).
        """
        if kwargs["mode"] != "single_image":
            return float("NaN")
        else:
            path = kwargs["path"]
            index = kwargs["index"]
            pattern = kwargs["pattern"]
            mode = kwargs["mode"]
            # ??? single_image ?????????? ?????????? ????????, ????? ???-?? ????????
            return (path, pattern, mode, index)

    # --------------------------------------------------------------------
    # ????????? ??????
    # --------------------------------------------------------------------
    def _scan_directory(self, directory_path, pattern):
        exts = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tiff"]
        files = []
        for file_name in glob.glob(os.path.join(directory_path, pattern), recursive=True):
            if os.path.splitext(file_name)[1].lower() in exts:
                files.append(os.path.abspath(file_name))
        files.sort(key=_natural_sort_key)
        return files

    def _load_as_tensor(self, file_path, allow_rgba=False):
        pil_img = Image.open(file_path)
        pil_img = ImageOps.exif_transpose(pil_img)

        # ???????? ? RGB, ???? ?? ???????? RGBA
        if not allow_rgba and pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        elif allow_rgba and pil_img.mode == "RGBA":
            # ????????? RGBA, ???? ???????????? ????????
            pass
        else:
            # ???? ????? ?????? P, LA ? ?.?., ???????? ? RGB
            if pil_img.mode not in ["RGB", "RGBA"]:
                pil_img = pil_img.convert("RGB")

        np_img = np.array(pil_img).astype(np.float32) / 255.0
        # batch dimension
        tensor = torch.from_numpy(np_img)[None, ]
        return tensor

    def _is_output_connected(self, extra_pnginfo, unique_id, output_slot):
        """
        ??????????:
        - True/False, ???? ??????? ?????????? ??????????? ?????? ?? workflow;
        - None, ???? ?????????? ?? ???????.
        """
        workflow = self._extract_workflow_from_extra(extra_pnginfo)
        if workflow is None or unique_id is None:
            return None

        nodes = workflow.get("nodes", None)
        if not isinstance(nodes, list):
            return None

        node_info = None
        for n in nodes:
            if str(n.get("id")) == str(unique_id):
                node_info = n
                break

        if node_info is None:
            return None

        outputs = node_info.get("outputs", None)
        if not isinstance(outputs, list):
            return None

        if output_slot < 0 or output_slot >= len(outputs):
            return False

        out = outputs[output_slot]
        if not isinstance(out, dict):
            return False

        links = out.get("links", None)
        return isinstance(links, list) and len(links) > 0

    def _extract_workflow_from_extra(self, extra_pnginfo):
        if extra_pnginfo is None:
            return None

        workflow = None
        if isinstance(extra_pnginfo, dict):
            workflow = extra_pnginfo.get("workflow", None)
            if workflow is None and "nodes" in extra_pnginfo:
                workflow = extra_pnginfo
        elif isinstance(extra_pnginfo, str):
            try:
                parsed = json.loads(extra_pnginfo)
                if isinstance(parsed, dict):
                    workflow = parsed.get("workflow", parsed)
            except Exception:
                workflow = None

        if isinstance(workflow, str):
            try:
                workflow = json.loads(workflow)
            except Exception:
                return None

        return workflow if isinstance(workflow, dict) else None

    def _read_metadata_and_prompt(self, file_path):
        metadata = {}
        try:
            with Image.open(file_path) as pil_img:
                metadata.update(dict(getattr(pil_img, "info", {}) or {}))
                exif_dict = self._extract_exif_dict(pil_img)
                if exif_dict:
                    metadata["exif"] = exif_dict
        except Exception as e:
            print(f"[BatchLoadImages] ?? ??????? ????????? ?????????? ?? '{file_path}': {e}")
            return ("", "")

        raw_metadata_text = self._metadata_to_json_string(metadata)
        prompt_text = self._extract_prompt_text(metadata)
        return (raw_metadata_text, prompt_text)

    def _extract_exif_dict(self, pil_img):
        exif_result = {}
        try:
            exif = pil_img.getexif()
            if exif:
                for tag_id, value in exif.items():
                    exif_result[str(tag_id)] = self._make_json_safe(value)
        except Exception:
            pass
        return exif_result

    def _metadata_to_json_string(self, metadata):
        if not isinstance(metadata, dict) or not metadata:
            return ""
        safe_metadata = {str(k): self._make_json_safe(v) for k, v in metadata.items()}
        try:
            return json.dumps(safe_metadata, ensure_ascii=False)
        except Exception:
            return str(safe_metadata)

    def _make_json_safe(self, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except Exception:
                return value.hex()
        if isinstance(value, dict):
            return {str(k): self._make_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._make_json_safe(v) for v in value]
        return str(value)

    def _extract_prompt_text(self, metadata):
        if not isinstance(metadata, dict):
            return ""

        # 1) Comfy prompt object
        prompt_obj = self._try_parse_json_object(metadata.get("prompt", None))
        if isinstance(prompt_obj, dict):
            extracted = self._extract_prompt_from_comfy_prompt(prompt_obj)
            if extracted:
                return extracted

        # 2) workflow.widgets_values fallback
        workflow_obj = self._try_parse_json_object(metadata.get("workflow", None))
        if isinstance(workflow_obj, dict):
            extracted = self._extract_prompt_from_workflow(workflow_obj)
            if extracted:
                return extracted

        # 3) A1111/generic fields fallback
        for key in ("prompt", "parameters", "Description", "description", "Comment", "comment"):
            raw_value = metadata.get(key, None)
            if raw_value is None:
                continue
            text_value = str(raw_value).strip()
            if not text_value:
                continue
            if key == "parameters":
                return text_value.split("Negative prompt:", 1)[0].strip()
            return text_value

        return ""

    def _extract_prompt_from_comfy_prompt(self, prompt_obj):
        if not isinstance(prompt_obj, dict):
            return ""

        preferred = []
        fallback = []
        for node_data in prompt_obj.values():
            if not isinstance(node_data, dict):
                continue

            class_type = str(node_data.get("class_type", ""))
            inputs = node_data.get("inputs", {})
            if not isinstance(inputs, dict):
                continue

            text_val = inputs.get("text", None)
            if isinstance(text_val, str) and text_val.strip():
                if "CLIPTextEncode" in class_type:
                    preferred.append(text_val.strip())
                else:
                    fallback.append(text_val.strip())

        if preferred:
            return preferred[0]
        if fallback:
            return fallback[0]
        return ""

    def _extract_prompt_from_workflow(self, workflow_obj):
        nodes = workflow_obj.get("nodes", None)
        if not isinstance(nodes, list):
            return ""

        preferred = []
        fallback = []
        for node in nodes:
            if not isinstance(node, dict):
                continue

            node_type = str(node.get("type", ""))
            widgets_values = node.get("widgets_values", None)
            if not isinstance(widgets_values, list):
                continue

            string_values = [v.strip() for v in widgets_values if isinstance(v, str) and v.strip()]
            if not string_values:
                continue

            if "CLIPTextEncode" in node_type:
                preferred.extend(string_values)
            else:
                fallback.extend(string_values)

        if preferred:
            return preferred[0]
        if fallback:
            return fallback[0]
        return ""

    def _try_parse_json_object(self, value):
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return None
        return None


class BatchLoadTextFiles:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["single_file", "incremental_file", "random"], {}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "index": ("INT", {"default": 0, "min": 0, "max": 150000}),
                "label": ("STRING", {"default": "Batch TXT 001"}),
                "path": ("STRING", {"default": ""}),
                "pattern": ("STRING", {"default": "*"}),
                "filename_filter_enabled": ("BOOLEAN", {"default": False, "tooltip": "Load only .txt files whose filename contains filter text."}),
                "filename_filter_text": ("STRING", {"default": "", "placeholder": "e.g. cat, prompt, scene_01"}),
                "allow_cycle": (["true", "false"], {"default": "true", "label_on": "Cycle On", "label_off": "Cycle Off"}),
            },
            "optional": {
                "filename_text_extension": (["true", "false"], {"default": "true"}),
                "match_filename_text": ("STRING", {"default": "", "forceInput": True, "tooltip": "Optional image filename from Batch Load Images. Limits TXT candidates to the same base name, e.g. result_1 -> result_1_face.txt."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "filename_text")
    FUNCTION = "load_batch_texts"
    CATEGORY = "😎 SnJake/Utils"

    incremental_counters = {}
    incremental_last_seed = {}

    def load_batch_texts(
        self,
        path,
        pattern="*",
        index=0,
        mode="single_file",
        seed=0,
        label="Batch TXT 001",
        filename_filter_enabled=False,
        filename_filter_text="",
        filename_text_extension="true",
        allow_cycle="true",
        match_filename_text="",
    ):
        # 1) Collect matching files for the current execution.
        all_files = self._scan_directory(path, pattern)

        if match_filename_text.strip():
            all_files = [
                f for f in all_files
                if self._filename_matches_base(f, match_filename_text)
            ]

        if self._filename_filter_is_enabled(filename_filter_enabled) and filename_filter_text.strip():
            filter_text = filename_filter_text.strip().casefold()
            all_files = [
                f for f in all_files
                if self._filename_matches_filter(f, filter_text)
            ]

        if not all_files:
            if self._filename_filter_is_enabled(filename_filter_enabled) and filename_filter_text.strip():
                print(
                    f"[BatchLoadTextFiles] No .txt files found in '{path}' for pattern '{pattern}' "
                    f"with filename filter '{filename_filter_text}'."
                )
            else:
                print(f"[BatchLoadTextFiles] No .txt files found in '{path}' for pattern '{pattern}'")
            return ("", "")

        # 2) Resolve which file should be used this run.
        if mode == "single_file":
            if index < 0 or index >= len(all_files):
                print(f"[BatchLoadTextFiles] Invalid index={index}, available files: {len(all_files)}")
                return ("", "")
            chosen_index = index

        elif mode == "incremental_file":
            if label not in self.incremental_counters:
                self.incremental_counters[label] = 0

            last_seed = self.incremental_last_seed.get(label, None)
            # Match BatchLoadImages behavior:
            # if seed was changed manually, synchronize the next incremental index to it.
            if last_seed is None:
                if seed > 0:
                    self.incremental_counters[label] = seed
            elif seed < last_seed or seed > (last_seed + 1):
                self.incremental_counters[label] = seed

            chosen_index = self.incremental_counters[label]

            if chosen_index >= len(all_files):
                if allow_cycle == "true":
                    print(f"[BatchLoadTextFiles] End of list for label='{label}' ({chosen_index}). Cycling to 0.")
                    chosen_index = 0
                    self.incremental_counters[label] = 0
                else:
                    print(f"[BatchLoadTextFiles] End of list for label='{label}'. Blocking output.")
                    return ("", "")

            self.incremental_counters[label] += 1
            self.incremental_last_seed[label] = seed

        else:  # mode == "random"
            random.seed(seed)
            chosen_index = random.randint(0, len(all_files) - 1)

        # 3) Read and return the selected file.
        txt_path = all_files[chosen_index]
        text_value = self._read_text(txt_path)
        filename = os.path.basename(txt_path)
        if filename_text_extension == "false":
            filename = os.path.splitext(filename)[0]

        return (text_value, filename)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Match BatchLoadImages behavior:
        # non-single modes always execute again so incremental/random can advance.
        if kwargs["mode"] != "single_file":
            return float("NaN")

        path = kwargs["path"]
        index = kwargs["index"]
        pattern = kwargs["pattern"]
        mode = kwargs["mode"]
        filename_filter_enabled = kwargs.get("filename_filter_enabled", False)
        filename_filter_text = kwargs.get("filename_filter_text", "")
        match_filename_text = kwargs.get("match_filename_text", "")
        return (path, pattern, mode, index, filename_filter_enabled, filename_filter_text, match_filename_text)

    def _scan_directory(self, directory_path, pattern):
        files = []
        for file_name in glob.glob(os.path.join(directory_path, pattern), recursive=True):
            if os.path.splitext(file_name)[1].lower() == ".txt":
                files.append(os.path.abspath(file_name))
        files.sort(key=_natural_sort_key)
        return files

    def _filename_filter_is_enabled(self, value):
        if isinstance(value, str):
            return value.strip().casefold() == "true"
        return bool(value)

    def _filename_matches_filter(self, file_path, filter_text):
        filename_without_extension = os.path.splitext(os.path.basename(file_path))[0]
        return filter_text in filename_without_extension.casefold()

    def _filename_matches_base(self, file_path, match_filename_text):
        wanted_base = os.path.splitext(os.path.basename(match_filename_text.strip()))[0].casefold()
        candidate_base = os.path.splitext(os.path.basename(file_path))[0].casefold()
        if not wanted_base:
            return True
        if candidate_base == wanted_base:
            return True
        return any(candidate_base.startswith(wanted_base + separator) for separator in ("_", "-", ".", " "))

    def _read_text(self, file_path):
        encodings = ["utf-8", "utf-8-sig", "cp1251", "latin-1"]
        last_error = None
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except Exception as e:
                last_error = e

        print(f"[BatchLoadTextFiles] Failed to read '{file_path}': {last_error}")
        return ""


class BatchLoadAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["single_audio", "incremental_audio", "random"], {}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "index": ("INT", {"default": 0, "min": 0, "max": 150000}),
                "label": ("STRING", {"default": "Batch Audio 001"}),
                "path": ("STRING", {"default": ""}),
                "pattern": ("STRING", {"default": "*"}),
                "allow_cycle": (["true", "false"], {"default": "true", "label_on": "Cycle On", "label_off": "Cycle Off"}),
            },
            "optional": {
                "filename_text_extension": (["true", "false"], {"default": "true"}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "filename_text", "raw_metadata", "prompt_text")
    FUNCTION = "load_batch_audio"
    CATEGORY = "\U0001F60E SnJake/Utils"

    incremental_counters = {}
    incremental_last_seed = {}

    def load_batch_audio(
        self,
        path,
        pattern="*",
        index=0,
        mode="single_audio",
        seed=0,
        label="Batch Audio 001",
        filename_text_extension="true",
        allow_cycle="true",
    ):
        if av is None:
            print("[BatchLoadAudio] PyAV is not available. Audio nodes are disabled.")
            return (None, None, "", "")

        all_files = self._scan_directory(path, pattern)
        if not all_files:
            print(f"[BatchLoadAudio] No audio files found in '{path}' for pattern '{pattern}'")
            return (None, None, "", "")

        if mode == "single_audio":
            if index < 0 or index >= len(all_files):
                print(f"[BatchLoadAudio] Invalid index={index}, available files: {len(all_files)}")
                return (None, None, "", "")
            chosen_index = index

        elif mode == "incremental_audio":
            if label not in self.incremental_counters:
                self.incremental_counters[label] = 0

            last_seed = self.incremental_last_seed.get(label, None)
            if last_seed is None:
                if seed > 0:
                    self.incremental_counters[label] = seed
            elif seed < last_seed or seed > (last_seed + 1):
                self.incremental_counters[label] = seed

            chosen_index = self.incremental_counters[label]

            if chosen_index >= len(all_files):
                if allow_cycle == "true":
                    print(f"[BatchLoadAudio] End of list for label='{label}' ({chosen_index}). Cycling to 0.")
                    chosen_index = 0
                    self.incremental_counters[label] = 0
                else:
                    print(f"[BatchLoadAudio] End of list for label='{label}'. Blocking output.")
                    return (None, None, "", "")

            self.incremental_counters[label] += 1
            self.incremental_last_seed[label] = seed

        else:
            random.seed(seed)
            chosen_index = random.randint(0, len(all_files) - 1)

        audio_path = all_files[chosen_index]
        try:
            audio_value, metadata = _load_audio_file(audio_path)
        except Exception as e:
            print(f"[BatchLoadAudio] Failed to load '{audio_path}': {e}")
            return (None, None, "", "")

        filename = os.path.basename(audio_path)
        if filename_text_extension == "false":
            filename = os.path.splitext(filename)[0]

        raw_metadata_text = _metadata_to_json_string(metadata)
        prompt_text = _extract_prompt_text_from_metadata(metadata)
        return (audio_value, filename, raw_metadata_text, prompt_text)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs["mode"] != "single_audio":
            return float("NaN")

        path = kwargs["path"]
        index = kwargs["index"]
        pattern = kwargs["pattern"]
        mode = kwargs["mode"]
        return (path, pattern, mode, index)

    def _scan_directory(self, directory_path, pattern):
        files = []
        for file_name in glob.glob(os.path.join(directory_path, pattern), recursive=True):
            if os.path.splitext(file_name)[1].lower() in AUDIO_FILE_EXTENSIONS:
                files.append(os.path.abspath(file_name))
        files.sort(key=_natural_sort_key)
        return files


class SaveTextToPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "save_path": ("STRING", {"default": "D:\\Stable diffusion\\result_7.txt"}),
                "append_mode": ("BOOLEAN", {"default": False, "tooltip": "If enabled, appends text instead of overwriting file."}),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_text"
    CATEGORY = "😎 SnJake/Utils"
    OUTPUT_NODE = True

    def save_text(self, text, save_path, append_mode=False):
        if text is None:
            text = ""

        try:
            path_obj = Path(save_path.strip().strip('"').strip("'"))
            full_path = path_obj.resolve()
            print(f"[SaveTextToPath] Full path: {full_path}")
        except Exception as e:
            print(f"[SaveTextToPath] Path error: {e}")
            return ()

        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[SaveTextToPath] Failed to create directory {full_path.parent}: {e}")
            return ()

        try:
            mode = "a" if append_mode else "w"
            with open(full_path, mode, encoding="utf-8") as f:
                f.write(str(text))
            print(f"[SaveTextToPath] Text saved: {full_path}")
        except Exception as e:
            print(f"[SaveTextToPath] Save error: {e}")

        return ()


class LoadSingleImageFromPath:
    """
    Узел для загрузки ОДНОГО изображения по ПОЛНОМУ пути, включая имя и формат.
    Пример входа:  /home/user/images/test.png
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"
    CATEGORY = "😎 SnJake/Utils"

    def load_image(self, image_path):
        if not os.path.exists(image_path):
            print(f"[LoadSingleImageFromPath] Файл '{image_path}' не найден!")
            return (None,)

        pil_img = Image.open(image_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        # Переводим всё к RGB на всякий случай
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        np_img = np.array(pil_img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_img)[None,]
        return (tensor,)




class SaveImageToPath:
    """
    Узел для сохранения полученного изображения в указанный путь.
    Пример полного пути: D:\Stable diffusion\result_7.png
    Если директории не существует, она будет создана автоматически.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "save_path": ("STRING", {"default": "D:\\Stable diffusion\\result_7.png"}),
                "save_workflow": ("BOOLEAN", {"default": True, "tooltip": "Сохранять ли workflow (метаданные) внутри изображения."}), # Новый параметр
            },
            "hidden": { # Скрытые входы для получения информации о workflow от ComfyUI
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_image"
    CATEGORY = "😎 SnJake/Utils"
    OUTPUT_NODE = True

    def save_image(self, image, save_path, save_workflow, prompt=None, extra_pnginfo=None):
        if image is None:
            print("[SaveImageToPath] Нет входного изображения!")
            return ()

        # Используем pathlib для нормализации пути
        try:
            path_obj = Path(save_path.strip().strip('"').strip("'"))
            # Если требуется, можно преобразовать путь к абсолютному
            full_path = path_obj.resolve()
            print(f"[SaveImageToPath] Полный путь: {full_path}")
        except Exception as e:
            print(f"[SaveImageToPath] Ошибка при обработке пути: {e}")
            return ()

        # Создаем директорию, если не существует
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[SaveImageToPath] Ошибка при создании директории {full_path.parent}: {e}")
            return ()

        # Преобразуем torch.Tensor -> numpy -> PIL Image
        try:
            # Берем первое изображение из батча, если их несколько
            # ComfyUI обычно передает батч изображений, даже если он состоит из одного элемента.
            # Индексация [0] предполагает, что мы сохраняем только первое изображение из батча,
            # или что узел предназначен для работы с одним изображением за раз.
            # Если нужно сохранять все изображения из батча, логику нужно будет изменить (например, цикл и модификация имени файла).
            # Для данного примера, предполагаем сохранение одного изображения.
            img_tensor = image[0] 
            np_img = img_tensor.cpu().numpy()
            
            # Транспонирование, если формат [C,H,W]
            if len(np_img.shape) == 3 and np_img.shape[0] in [1, 3, 4]: # (C, H, W)
                np_img = np.transpose(np_img, (1, 2, 0)) # (H, W, C)
            
            np_img = (np_img * 255.0).clip(0, 255).astype(np.uint8)
            
            # Определяем режим в зависимости от числа каналов
            if np_img.ndim == 2: # Grayscale
                mode = "L"
            elif np_img.shape[2] == 3: # RGB
                mode = "RGB"
            elif np_img.shape[2] == 4: # RGBA
                mode = "RGBA"
            elif np_img.shape[2] == 1: # Одноканальное, но не L (может быть маска)
                np_img = np_img.squeeze(axis=2) # Удаляем последнюю размерность
                mode = "L" # Сохраняем как Grayscale
            else:
                print(f"[SaveImageToPath] Неподдерживаемое количество каналов: {np_img.shape[2]}")
                return ()
                
            pil_img = Image.fromarray(np_img, mode=mode)
        except Exception as e:
            print(f"[SaveImageToPath] Ошибка при конвертации изображения: {e}")
            return ()

        # Подготовка метаданных для сохранения workflow
        metadata_to_save = None
        if save_workflow:
            metadata_to_save = PngInfo()
            if prompt is not None:
                metadata_to_save.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None and isinstance(extra_pnginfo, dict):
                for k, v in extra_pnginfo.items():
                    metadata_to_save.add_text(k, json.dumps(v))
        
        # Сохраняем изображение
        try:
            pil_img.save(str(full_path), pnginfo=metadata_to_save)
            print(f"[SaveImageToPath] Изображение успешно сохранено: {full_path}")
            if save_workflow:
                if prompt or extra_pnginfo:
                     print(f"[SaveImageToPath] Workflow data has been included in the image.")
                else:
                     print(f"[SaveImageToPath] Workflow saving was enabled, but no workflow data (prompt/extra_pnginfo) was available to save.")
            else:
                print(f"[SaveImageToPath] Workflow data was not saved to the image (option disabled).")
        except Exception as e:
            print(f"[SaveImageToPath] Ошибка при сохранении изображения: {e}")

        return ()


class SaveAudioToPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "save_path": ("STRING", {"default": "D:\\Stable diffusion\\result_7.flac"}),
                "save_workflow": ("BOOLEAN", {"default": True, "tooltip": "Save workflow metadata inside the audio container when supported."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_audio"
    CATEGORY = "\U0001F60E SnJake/Utils"
    OUTPUT_NODE = True

    def save_audio(self, audio, save_path, save_workflow, prompt=None, extra_pnginfo=None):
        if av is None:
            print("[SaveAudioToPath] PyAV is not available. Audio nodes are disabled.")
            return ()

        if audio is None:
            print("[SaveAudioToPath] No input audio provided.")
            return ()

        try:
            path_obj = Path(save_path.strip().strip('"').strip("'"))
            full_path = path_obj.resolve()
            print(f"[SaveAudioToPath] Full path: {full_path}")
        except Exception as e:
            print(f"[SaveAudioToPath] Path error: {e}")
            return ()

        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[SaveAudioToPath] Failed to create directory {full_path.parent}: {e}")
            return ()

        waveform = audio.get("waveform", None) if isinstance(audio, dict) else None
        if waveform is not None and waveform.ndim == 3 and waveform.shape[0] > 1:
            print("[SaveAudioToPath] Batch audio detected. Only the first audio item will be saved.")

        metadata_to_save = None
        if save_workflow:
            metadata_to_save = {}
            if prompt is not None:
                metadata_to_save["prompt"] = json.dumps(prompt, ensure_ascii=False)
            if extra_pnginfo is not None and isinstance(extra_pnginfo, dict):
                for k, v in extra_pnginfo.items():
                    metadata_to_save[str(k)] = json.dumps(v, ensure_ascii=False)

        try:
            _save_audio_file(audio, full_path, metadata_to_save)
            print(f"[SaveAudioToPath] Audio saved: {full_path}")
            if save_workflow:
                if prompt or extra_pnginfo:
                    print("[SaveAudioToPath] Workflow metadata has been included in the audio file.")
                else:
                    print("[SaveAudioToPath] Workflow saving was enabled, but no workflow metadata was available.")
            else:
                print("[SaveAudioToPath] Workflow metadata was not saved (option disabled).")
        except Exception as e:
            print(f"[SaveAudioToPath] Save error: {e}")

        return ()


class ImageRouter:
    CATEGORY = "😎 SnJake/Utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,  # Максимальное количество выходов
                    "step": 1,
                    "tooltip": "Выберите номер выхода для перенаправления изображения."
                }),
                "image_in": ("IMAGE", {
                    "tooltip": "Входящее изображение для перенаправления."
                }),
            },
            "optional": {
                "max_outputs": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 10,  # Максимальное количество выходов не должно превышать RETURN_TYPES
                    "step": 1,
                    "tooltip": "Максимальное количество выходов. Не должно превышать 10."
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = tuple(["IMAGE"] * 10)  # Определяем 10 выходов типа IMAGE
    RETURN_NAMES = tuple([f"output{i+1}" for i in range(10)])  # Имена выходов: output1, output2, ..., output10
    FUNCTION = "switch_image"

    def switch_image(self, select, image_in, max_outputs=10, unique_id=None, prompt=None, **kwargs):
        """
        Перенаправляет входящее изображение в выбранный выход.
        Остальные выходы блокируются.
        """
        # Ограничиваем значение max_outputs до 10
        max_outputs = max(1, min(max_outputs, 10))

        # Проверяем, что значение select находится в допустимом диапазоне
        if not 1 <= select <= max_outputs:
            raise ValueError(f"Значение 'select' ({select}) должно быть в диапазоне от 1 до {max_outputs}.")

        outputs = []
        for i in range(1, max_outputs + 1):
            if i == select:
                outputs.append(image_in)
            else:
                # Блокируем остальные выходы
                outputs.append(ExecutionBlocker(None))

        # Заполняем оставшиеся выходы значением None, если max_outputs меньше 10
        while len(outputs) < 10:
            outputs.append(None)

        return tuple(outputs)





class StringToNumber:
    """
    Узел, который берёт строку (STRING) и пробует сконвертировать в int и float.
    На выходе два значения: int_value, float_value
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {"default": "123"})
            }
        }

    RETURN_TYPES = ("INT","FLOAT")
    RETURN_NAMES = ("as_int","as_float")
    FUNCTION = "convert"
    CATEGORY = "😎 SnJake/Utils"

    def convert(self, input_string):
        try:
            i_val = int(input_string)
        except:
            i_val = 0
            print(f"[StringToNumber] Не удалось преобразовать '{input_string}' к int. Ставим 0.")

        try:
            f_val = float(input_string)
        except:
            f_val = 0.0
            print(f"[StringToNumber] Не удалось преобразовать '{input_string}' к float. Ставим 0.0.")

        return (i_val, f_val)




class StringReplace:
    """
    Узел, который заменяет в исходном тексте (source_string) подстроку (old_string)
    на (new_string). Пример:
      source_string = "Hello"
      old_string    = "ell"
      new_string    = "bob"
      => "Hbobo"
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_string": ("STRING", {"default": "Hello", "multiline": True}),
                "old_string":    ("STRING", {"default": "ell", "multiline": True}),
                "new_string":    ("STRING", {"default": "bob", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("replaced_string",)
    FUNCTION = "string_replace"
    CATEGORY = "😎 SnJake/Utils"

    def string_replace(self, source_string, old_string, new_string):
        if source_string is None:
            return ("",)
        result = source_string.replace(old_string, new_string)
        return (result,)





class RandomIntNode:
    CATEGORY = "😎 SnJake/Utils"
    FUNCTION = "generate"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("random_int",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_value": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1, "tooltip": "Минимальное значение"}),
                "max_value": ("INT", {"default": 10, "min": -10000, "max": 10000, "step": 1, "tooltip": "Максимальное значение"})
            }
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Возвращает NaN, чтобы узел всегда пересчитывался и не использовался кэш
        return float("NaN")

    def generate(self, min_value, max_value):
        result = random.randint(min_value, max_value)
        return (result,)


class RandomFloatNode:
    CATEGORY = "😎 SnJake/Utils"
    FUNCTION = "generate"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("random_float",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_value": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01, "tooltip": "Минимальное значение"}),
                "max_value": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01, "tooltip": "Максимальное значение"})
            }
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Отключаем кэширование, возвращая NaN
        return float("NaN")

    def generate(self, min_value, max_value):
        value = random.uniform(min_value, max_value)
        # Округляем до 2 знаков после запятой, чтобы, например, 0.53228 превратилось в 0.53
        result = round(value, 2)
        return (result,)
