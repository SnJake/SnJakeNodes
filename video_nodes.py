import glob
import os
import random
import re
from fractions import Fraction

import av
import numpy as np
import torch
import torch.nn.functional as F

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


def _audio_to_float32(audio_data):
    if np.issubdtype(audio_data.dtype, np.floating):
        return audio_data.astype(np.float32, copy=False)
    if np.issubdtype(audio_data.dtype, np.signedinteger):
        return audio_data.astype(np.float32) / float(2 ** (audio_data.dtype.itemsize * 8 - 1))
    if np.issubdtype(audio_data.dtype, np.unsignedinteger):
        midpoint = float(2 ** (audio_data.dtype.itemsize * 8 - 1))
        return (audio_data.astype(np.float32) - midpoint) / midpoint
    return audio_data.astype(np.float32)


def _load_audio_range(video_path, start_time, duration):
    with av.open(video_path) as container:
        if not container.streams.audio:
            return None

        stream = container.streams.audio[0]
        sample_rate = stream.codec_context.sample_rate or stream.sample_rate
        channel_count = stream.codec_context.channels or 1
        end_time = start_time + duration
        audio_frames = []
        fallback_time = 0.0

        for frame in container.decode(stream):
            frame_rate = frame.sample_rate or sample_rate
            frame_start = float(frame.time) if frame.time is not None else fallback_time
            frame_end = frame_start + frame.samples / frame_rate
            fallback_time = frame_end
            if frame_end <= start_time:
                continue
            if frame_start >= end_time:
                break

            audio_data = frame.to_ndarray()
            if audio_data.ndim == 1:
                audio_data = audio_data[None, :]
            elif audio_data.shape[0] != channel_count:
                audio_data = audio_data.reshape(-1, channel_count).T

            first_sample = max(0, round((start_time - frame_start) * frame_rate))
            last_sample = min(frame.samples, round((end_time - frame_start) * frame_rate))
            if last_sample > first_sample:
                audio_frames.append(_audio_to_float32(audio_data[:, first_sample:last_sample]))

        if not audio_frames:
            return None

        waveform = np.ascontiguousarray(np.concatenate(audio_frames, axis=1))
        return {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": int(sample_rate)}


def _decode_video_frames(video_path, source_fps, width, height, force_rate, frame_load_cap, skip_first_frames, select_every_nth):
    target_fps = force_rate if force_rate > 0 else source_fps

    def frame_generator():
        target_index = 0
        resampled_index = 0
        frames_added = 0

        with av.open(video_path) as container:
            stream = container.streams.video[0]
            for source_index, frame in enumerate(container.decode(stream)):
                if source_index < skip_first_frames:
                    continue

                relative_index = source_index - skip_first_frames
                repeat_count = 1
                if force_rate > 0:
                    repeat_count = 0
                    while int(target_index * source_fps / target_fps) <= relative_index:
                        if int(target_index * source_fps / target_fps) == relative_index:
                            repeat_count += 1
                        target_index += 1

                frame_array = None
                for _ in range(repeat_count):
                    if resampled_index % select_every_nth == 0:
                        if frame_array is None:
                            if (frame.width, frame.height) != (width, height):
                                frame = frame.reformat(
                                    width=width,
                                    height=height,
                                    format="rgb24",
                                    interpolation=av.video.reformatter.Interpolation.LANCZOS,
                                )
                            frame_array = frame.to_ndarray(format="rgb24")
                        yield frame_array
                        frames_added += 1
                        if frame_load_cap > 0 and frames_added >= frame_load_cap:
                            return
                    resampled_index += 1

    frame_dtype = np.dtype((np.float32, (height, width, 3)))
    images = torch.from_numpy(np.fromiter(frame_generator(), dtype=frame_dtype))
    if len(images) == 0:
        raise ValueError("No video frames were decoded from the selected range.")
    images.div_(255.0)
    return images, target_fps / select_every_nth


def _load_video_file(video_path, force_rate, custom_width, custom_height, frame_load_cap, skip_first_frames, select_every_nth):
    source_video = InputImpl.VideoFromFile(video_path)
    source_fps = float(source_video.get_frame_rate())
    source_frame_count = source_video.get_frame_count()
    source_duration = source_video.get_duration()
    source_width, source_height = source_video.get_dimensions()
    source_bit_depth = source_video.get_bit_depth()
    container_format = source_video.get_container_format()

    loaded_width, loaded_height = _target_size(source_width, source_height, custom_width, custom_height)
    images, loaded_fps = _decode_video_frames(
        video_path,
        source_fps,
        loaded_width,
        loaded_height,
        force_rate,
        frame_load_cap,
        skip_first_frames,
        select_every_nth,
    )
    loaded_frame_count = len(images)
    loaded_duration = loaded_frame_count / loaded_fps
    audio = _load_audio_range(video_path, skip_first_frames / source_fps, loaded_duration)
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
