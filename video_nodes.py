import glob
import os
import random
import re
from fractions import Fraction

import torch
import torch.nn.functional as F

import comfy.utils
import folder_paths
from comfy_api.latest import InputImpl, Types


VIDEO_EXTENSIONS = {
    ".avi",
    ".flv",
    ".m2ts",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mts",
    ".ogv",
    ".ts",
    ".webm",
    ".wmv",
}


def _natural_sort_key(file_path):
    filename = os.path.basename(file_path).casefold()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", filename)]


def _input_videos():
    input_dir = folder_paths.get_input_directory()
    files = [name for name in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, name))]
    return folder_paths.filter_files_content_types(files, ["video"])


def _validate_video_path(video_path):
    if os.path.splitext(video_path)[1].lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video file: {os.path.basename(video_path)}")


def _scan_video_files(directory_path, pattern):
    if not os.path.isdir(directory_path):
        return []

    files = [
        os.path.abspath(file_path)
        for file_path in glob.glob(os.path.join(directory_path, pattern), recursive=True)
        if os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in VIDEO_EXTENSIONS
    ]
    files.sort(key=_natural_sort_key)
    return files


def _target_size(width, height, custom_width, custom_height):
    if custom_width == 0 and custom_height == 0:
        return width, height
    if custom_width == 0:
        custom_width = max(1, round(width * custom_height / height))
    if custom_height == 0:
        custom_height = max(1, round(height * custom_width / width))
    return custom_width, custom_height


def _resample_frames(images, source_fps, target_fps):
    if target_fps <= 0 or abs(target_fps - source_fps) < 1e-6:
        return images

    target_count = max(1, round(len(images) * target_fps / source_fps))
    indices = [min(int(index * source_fps / target_fps), len(images) - 1) for index in range(target_count)]
    return images[indices]


def _trim_audio(audio, duration):
    if audio is None:
        return None

    sample_rate = int(audio["sample_rate"])
    waveform = audio["waveform"]
    sample_count = min(waveform.shape[-1], round(duration * sample_rate))
    if sample_count == waveform.shape[-1]:
        return audio
    return {"waveform": waveform[..., :sample_count].clone(), "sample_rate": sample_rate}


def _load_video_file(video_path, force_rate, custom_width, custom_height, frame_load_cap, skip_first_frames, select_every_nth):
    source_video = InputImpl.VideoFromFile(video_path)
    source_fps = float(source_video.get_frame_rate())
    source_frame_count = source_video.get_frame_count()
    source_duration = source_video.get_duration()
    source_width, source_height = source_video.get_dimensions()
    source_bit_depth = source_video.get_bit_depth()
    container_format = source_video.get_container_format()

    start_time = skip_first_frames / source_fps
    base_fps = force_rate if force_rate > 0 else source_fps
    duration = frame_load_cap * select_every_nth / base_fps if frame_load_cap > 0 else 0
    video = source_video.as_trimmed(start_time, duration, strict_duration=False)
    if video is None:
        raise ValueError("The selected video range is empty.")

    components = video.get_components()
    images = components.images
    if len(images) == 0:
        raise ValueError("No video frames were decoded from the selected range.")

    decoded_fps = float(components.frame_rate)
    images = _resample_frames(images, decoded_fps, force_rate)
    loaded_fps = (force_rate if force_rate > 0 else decoded_fps) / select_every_nth

    selected_images = images[::select_every_nth]
    if frame_load_cap > 0:
        selected_images = selected_images[:frame_load_cap]
    if select_every_nth > 1 or frame_load_cap > 0:
        selected_images = selected_images.clone()
    images = selected_images

    loaded_height, loaded_width = images.shape[1:3]
    target_width, target_height = _target_size(loaded_width, loaded_height, custom_width, custom_height)
    if (target_width, target_height) != (loaded_width, loaded_height):
        images = comfy.utils.common_upscale(images.movedim(-1, 1), target_width, target_height, "lanczos", "disabled").movedim(1, -1)
        loaded_width, loaded_height = target_width, target_height

    loaded_frame_count = len(images)
    loaded_duration = loaded_frame_count / loaded_fps
    audio = _trim_audio(components.audio, loaded_duration)
    video_info = {
        "source_fps": source_fps,
        "source_frame_count": source_frame_count,
        "source_duration": source_duration,
        "source_width": source_width,
        "source_height": source_height,
        "loaded_fps": loaded_fps,
        "loaded_frame_count": loaded_frame_count,
        "loaded_duration": loaded_duration,
        "loaded_width": loaded_width,
        "loaded_height": loaded_height,
        "source_bit_depth": source_bit_depth,
        "container_format": container_format,
        "filename": os.path.basename(video_path),
    }
    processed_video = InputImpl.VideoFromComponents(
        Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(loaded_fps).limit_denominator(100000)),
        bit_depth=source_bit_depth,
    )
    return images, loaded_frame_count, audio, video_info, processed_video


def _loader_inputs(video_input):
    return {
        "required": {
            "video": video_input,
            "force_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01}),
            "custom_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 2}),
            "custom_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 2}),
            "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 1000000}),
            "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 1000000}),
            "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 1000000}),
        }
    }


class SnJakeVideoFrameLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return _loader_inputs((sorted(_input_videos()), {"video_upload": True}))

    RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "SNJAKE_VIDEO_INFO", "VIDEO")
    RETURN_NAMES = ("images", "frame_count", "audio", "video_info", "video")
    FUNCTION = "load_video"
    CATEGORY = "😎 SnJake/Video"

    def load_video(self, video, force_rate, custom_width, custom_height, frame_load_cap, skip_first_frames, select_every_nth):
        video_path = folder_paths.get_annotated_filepath(video)
        _validate_video_path(video_path)
        return _load_video_file(video_path, force_rate, custom_width, custom_height, frame_load_cap, skip_first_frames, select_every_nth)

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        video_path = folder_paths.get_annotated_filepath(video)
        _validate_video_path(video_path)
        return os.path.getmtime(video_path)

    @classmethod
    def VALIDATE_INPUTS(cls, video, **kwargs):
        if not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        video_path = folder_paths.get_annotated_filepath(video)
        if os.path.splitext(video_path)[1].lower() not in VIDEO_EXTENSIONS:
            return f"Unsupported video file: {video}"
        return True


class SnJakeVideoDetails:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video_info": ("SNJAKE_VIDEO_INFO",)}}

    RETURN_TYPES = ("FLOAT", "INT", "FLOAT", "INT", "INT", "FLOAT", "INT", "FLOAT", "INT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = (
        "source_fps",
        "source_frame_count",
        "source_duration",
        "source_width",
        "source_height",
        "loaded_fps",
        "loaded_frame_count",
        "loaded_duration",
        "loaded_width",
        "loaded_height",
        "source_bit_depth",
        "container_format",
        "filename",
    )
    FUNCTION = "get_video_details"
    CATEGORY = "😎 SnJake/Video"

    def get_video_details(self, video_info):
        return tuple(video_info[name] for name in self.RETURN_NAMES)


class SnJakeBatchLoadVideos:
    incremental_counters = {}
    incremental_last_seed = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["single_video", "incremental_video", "random"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "index": ("INT", {"default": 0, "min": 0, "max": 150000}),
                "label": ("STRING", {"default": "Batch Video 001"}),
                "path": ("STRING", {"default": ""}),
                "pattern": ("STRING", {"default": "*"}),
                "force_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 2}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 2}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 1000000}),
                "allow_cycle": (["true", "false"], {"default": "true"}),
            },
            "optional": {
                "filename_text_extension": (["true", "false"], {"default": "true"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "SNJAKE_VIDEO_INFO", "VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("images", "frame_count", "audio", "video_info", "video", "filename_text", "full_path")
    FUNCTION = "load_batch_video"
    CATEGORY = "😎 SnJake/Video"

    def load_batch_video(
        self,
        mode,
        seed,
        index,
        label,
        path,
        pattern,
        force_rate,
        custom_width,
        custom_height,
        frame_load_cap,
        skip_first_frames,
        select_every_nth,
        allow_cycle,
        filename_text_extension="true",
    ):
        files = _scan_video_files(path, pattern)
        if not files:
            raise FileNotFoundError(f"No video files found in '{path}' for pattern '{pattern}'.")

        if mode == "single_video":
            if index >= len(files):
                raise IndexError(f"Video index {index} is out of range; found {len(files)} files.")
            chosen_index = index
        elif mode == "incremental_video":
            counter = self.incremental_counters.get(label, seed)
            last_seed = self.incremental_last_seed.get(label)
            if last_seed is not None and (seed < last_seed or seed > last_seed + 1):
                counter = seed
            if counter >= len(files):
                if allow_cycle == "false":
                    raise IndexError(f"End of video list for label '{label}'.")
                counter = 0
            chosen_index = counter
            self.incremental_counters[label] = counter + 1
            self.incremental_last_seed[label] = seed
        else:
            chosen_index = random.Random(seed).randrange(len(files))

        video_path = files[chosen_index]
        loaded = _load_video_file(video_path, force_rate, custom_width, custom_height, frame_load_cap, skip_first_frames, select_every_nth)
        filename = os.path.basename(video_path)
        if filename_text_extension == "false":
            filename = os.path.splitext(filename)[0]
        return (*loaded, filename, video_path)

    @classmethod
    def IS_CHANGED(cls, mode, path, pattern, index, **kwargs):
        if mode != "single_video":
            return float("NaN")
        files = _scan_video_files(path, pattern)
        if index >= len(files):
            return (path, pattern, index)
        return (files[index], os.path.getmtime(files[index]))


class SnJakeVideoComposer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 240.0, "step": 0.01}),
                "filename_prefix": ("STRING", {"default": "video/SnJake"}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51}),
                "bit_depth": (["8", "10"], {"default": "8"}),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "trim_to_audio": ("BOOLEAN", {"default": False}),
                "save_metadata": ("BOOLEAN", {"default": True}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "filename")
    FUNCTION = "compose_video"
    CATEGORY = "😎 SnJake/Video"
    OUTPUT_NODE = True

    def compose_video(
        self,
        images,
        frame_rate,
        filename_prefix,
        crf,
        bit_depth,
        loop_count,
        pingpong,
        trim_to_audio,
        save_metadata,
        save_output,
        audio=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        if len(images) == 0:
            raise ValueError("Cannot create a video from an empty image batch.")

        if images.shape[-1] > 3:
            images = images[..., :3].clone()
        if pingpong and len(images) > 2:
            images = torch.cat((images, images[1:-1].flip(0)), dim=0)
        if loop_count > 0:
            images = torch.cat([images] * (loop_count + 1), dim=0)
            if audio is not None:
                audio = {
                    "waveform": audio["waveform"].repeat(1, 1, loop_count + 1),
                    "sample_rate": audio["sample_rate"],
                }

        if trim_to_audio and audio is not None:
            audio_duration = audio["waveform"].shape[-1] / audio["sample_rate"]
            frame_count = min(len(images), max(1, int(audio_duration * frame_rate)))
            if frame_count < len(images):
                images = images[:frame_count].clone()

        pad_width = images.shape[2] % 2
        pad_height = images.shape[1] % 2
        if pad_width or pad_height:
            images = F.pad(images.movedim(-1, 1), (0, pad_width, 0, pad_height), mode="replicate").movedim(1, -1)

        output_bit_depth = {"8": 8, "10": 10}[bit_depth]
        video = InputImpl.VideoFromComponents(
            Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(frame_rate).limit_denominator(100000)),
            bit_depth=output_bit_depth,
        )
        if not save_output:
            return (video, "")

        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            images.shape[2],
            images.shape[1],
        )
        output_name = f"{filename}_{counter:05}.mp4"
        output_path = os.path.join(full_output_folder, output_name)
        metadata = None
        if save_metadata:
            metadata = dict(extra_pnginfo or {})
            if prompt is not None:
                metadata["prompt"] = prompt
            if not metadata:
                metadata = None

        video.save_to(
            output_path,
            format=Types.VideoContainer.MP4,
            codec=Types.VideoCodec.H264,
            metadata=metadata,
            bit_depth=output_bit_depth,
            crf=crf,
        )
        preview = {
            "filename": output_name,
            "subfolder": subfolder,
            "type": "output",
            "format": "video/mp4",
            "frame_rate": frame_rate,
            "fullpath": output_path,
        }
        return {"ui": {"gifs": [preview]}, "result": (video, output_path)}
