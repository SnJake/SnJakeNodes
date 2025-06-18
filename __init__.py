from .api import VLMApiNode
from .list_image import ScanImageFolder
from .list_image_not import ScanImageFolder2
from .crt_effect_node import CRTEffectNode
from .liminal_effects_node import LiminalEffectsNode
from .image_resize_node import ImageResizeNode
from .color_adjusment import ImageAdjustmentNode
from .color_balance import ColorBalance
from .utils_nodes import (
    BatchLoadImages,
    LoadSingleImageFromPath,
    SaveImageToPath,
    ImageRouter,
    StringToNumber,
    StringReplace,
    RandomIntNode,
    RandomFloatNode
)

from .custom_mask_node import CustomMaskNode
from .mask_selector_node import ImageMaskSelector

from .anynode_snjake import OpenAICompatibleNode, LocalOpenAICompatibleNode

from .image_concatenate import ConcatenateImagesByDirectory

from .expand_image_right import ExpandImageRight



from .pixel_art_node import PixelArtNode
from .color_palette_image_node import ColorPaletteImageNode
from .color_palette_compression_node import ColorPaletteCompressionNode
from .region_merging_node import RegionMergingNode
from .pixelart_postprocess_node import PixelArtPostProcessNode

from .segmentation_pixel_art_node import SegmentationPixelArtNode



from .prompt_selector import MultilinePromptSelector




from .pixel_art_utils import ExtractPaletteNode, ApplyPaletteNode, ReplacePaletteColorsNode



from .xy_plot_node import XYPlotAdvanced

from .lora_loader_preview import LoraLoaderWithPreview

from .datetime_node import DateTimeToStringNode

from .detailer_node import DetailerForEachMask

from .lora-switch import LoraSwitchDynamic, LoraBlocker

# Регистарция путей
import os
import folder_paths
import traceback
import sys

# --- Регистрация папки моделей PixelArt и подпапок ---
# Получаем абсолютный путь к папке models внутри ComfyUI
comfy_models_dir = folder_paths.models_dir
# Определяем базовый путь для pixelart
pixelart_model_dir = os.path.join(comfy_models_dir, "pixelart")
# Создаем папку, если её нет
os.makedirs(pixelart_model_dir, exist_ok=True)
# Регистрируем базовую папку (может быть полезно для других целей)
folder_paths.add_model_folder_path("pixelart", pixelart_model_dir)
print(f"-> Registered PixelArt base folder: {pixelart_model_dir}")

# Определяем и регистрируем подпапки с составными ключами
subfolders = ["i2pnet", "aliasnet", "csenc"]
for subfolder in subfolders:
    subfolder_path = os.path.join(pixelart_model_dir, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)
    # Создаем составной ключ типа "pixelart/i2pnet"
    folder_key = f"pixelart/{subfolder}"
    # Регистрируем путь с этим ключом
    folder_paths.add_model_folder_path(folder_key, subfolder_path)
    print(f"-> Registered PixelArt subfolder '{folder_key}': {subfolder_path}")
# --- Конец регистрации ---





# Add the parent directory of 'custom_nodes' to sys.path
# This is necessary to import folder_paths from the main ComfyUI directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import folder_paths
except ImportError:
     print("Warning: folder_paths module not found in __init__. Model loading might fail.")
     # Define a dummy class if import fails, so the rest of the script doesn't crash
     class MockFolderPaths:
        def add_model_folder_path(self, *args, **kwargs): pass
     folder_paths = MockFolderPaths()

# --- Register the custom model folder ---
# Define the name and path relative to the ComfyUI base directory
watermark_models_dir_name = "watermark_detection"
watermark_models_path = os.path.join(folder_paths.models_dir, watermark_models_dir_name)

# Create the directory if it doesn't exist
if not os.path.exists(watermark_models_path):
    try:
        os.makedirs(watermark_models_path)
        print(f"Created directory: {watermark_models_path}")
    except OSError as e:
        print(f"Warning: Could not create directory {watermark_models_path}: {e}")

# Add the path to ComfyUI's model paths
# The second argument is the set of supported extensions (same as checkpoints)
# Use getattr to safely access supported_pt_extensions in case folder_paths is mocked
supported_exts = getattr(folder_paths, 'supported_pt_extensions', {'.pth', '.safetensors'})
folder_paths.add_model_folder_path(watermark_models_dir_name, watermark_models_path)
print(f"[*] Registered watermark detection models folder: {watermark_models_dir_name} -> {watermark_models_path}")
# --- End of registration ---



NODE_CLASS_MAPPINGS = {
    "VLMApiNode": VLMApiNode,
    "ScanImageFolder": ScanImageFolder,
    "ScanImageFolder2": ScanImageFolder2,
    "CRTEffectNode": CRTEffectNode,
    "LiminalEffectsNode": LiminalEffectsNode,
    "ImageResizeNode": ImageResizeNode,
    "ImageAdjustmentNode": ImageAdjustmentNode,
    "ColorBalance": ColorBalance,

    "CustomMaskNode": CustomMaskNode,
    "ExpandImageRight": ExpandImageRight,
    "ImageMaskSelector": ImageMaskSelector,
    
    "OpenAICompatibleNode": OpenAICompatibleNode,
    "LocalOpenAICompatibleNode": LocalOpenAICompatibleNode,
   

    "MultilinePromptSelector": MultilinePromptSelector,
    
    "ConcatenateImagesByDirectory": ConcatenateImagesByDirectory,
    
    "RandomFloatNode": RandomFloatNode,
    "RandomIntNode": RandomIntNode,

    
    "BatchLoadImages": BatchLoadImages,
    "LoadSingleImageFromPath": LoadSingleImageFromPath,
    "SaveImageToPath": SaveImageToPath,
    "ImageRouter": ImageRouter,
    "StringToNumber": StringToNumber,
    "StringReplace": StringReplace,
    
    
    
    "PixelArtNode": PixelArtNode,
    "ColorPaletteImageNode": ColorPaletteImageNode,
    "ColorPaletteCompressionNode": ColorPaletteCompressionNode,
    "RegionMergingNode": RegionMergingNode,
    "PixelArtPostProcessNode": PixelArtPostProcessNode,
    "SegmentationPixelArtNode": SegmentationPixelArtNode,


    "ExtractPaletteNode": ExtractPaletteNode,
    "ApplyPaletteNode": ApplyPaletteNode,
    "ReplacePaletteColorsNode": ReplacePaletteColorsNode,
    
    


    "XYPlotAdvanced": XYPlotAdvanced,
    "LoraLoaderWithPreview": LoraLoaderWithPreview,
    
    "DateTimeToStringNode": DateTimeToStringNode,

    "DetailerForEachMask": DetailerForEachMask,

    "LoraSwitchDynamic": LoraSwitchDynamic,
    "LoraBlocker": LoraBlocker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VLMApiNode": "😎 VLM Api Node",
    "ScanImageFolder": "😎 List Image Files In Folder",
    "ScanImageFolder2": "😎 Simple List Image Files In Folder",    
    "CRTEffectNode": "😎 CRT Effect",
    "LiminalEffectsNode": "😎 Liminal Effects Node",
    "ImageResizeNode": "😎 Image Resize Node",
    "ColorBalance": "😎 Color Balance",   
    "ImageAdjustmentNode": "😎 Image Adjustment Node",
    
    "OpenAICompatibleNode": "😎 AnyNode OpenAI",   
    
    "MultilinePromptSelector": "😎 Multiline Prompt Selector",    
    
    "LocalOpenAICompatibleNode": "😎 AnyNode Local LLM", 
    
    "ConcatenateImagesByDirectory": "😎 Concatenate Images By Directory", 
    

    "RandomFloatNode": "😎 Random Float Node",
    "RandomIntNode": "😎  Random Int Node",

    "BatchLoadImages": "😎 Batch Load Images",
    "LoadSingleImageFromPath": "😎 Load Image By Full Path",
    "SaveImageToPath": "😎 Save Image To Path",
    "ImageRouter": "😎 Route Image By Int",
    "StringToNumber": "😎 String->Int/Float",
    "StringReplace": "😎 String Replace",
    
    "CustomMaskNode": "😎 Custom Mask Node",
    "ExpandImageRight": "😎 Expand Image Right",
    "ImageMaskSelector": "😎 Image Mask Selector",
    
    
    "PixelArtNode": "😎 Pixel Art Effect",
    "ColorPaletteImageNode": "😎 Color Palette",
    "ColorPaletteCompressionNode": "😎 Color Comprassion",
    "RegionMergingNode": "😎 Color Merge",
    "PixelArtPostProcessNode": "😎 Pixel Art Post Process Node",
    "SegmentationPixelArtNode": "😎 Segmentation Pixel Art Node",
    
    

    "ExtractPaletteNode": "😎 Extract Palette",
    "ApplyPaletteNode": "😎 Apply Palette (and Dither)",
    "ReplacePaletteColorsNode": "😎 Replace Palette Colors",

    
    "XYPlotAdvanced": "😎 XY Plot Advanced",
    
    "LoraLoaderWithPreview": "😎 Lora Loader With Preview",
    
    "DateTimeToStringNode": "😎 Date Time To String",

    "DetailerForEachMask": "😎 Sequential Mask Detailer",
    
    "LoraSwitchDynamic": "😎 Lora Switcher",
    "LoraBlocker": "😎 Lora Blocker",


}


WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "WEB_DIRECTORY"]


print("### SnJake Nodes Initialized ###")
