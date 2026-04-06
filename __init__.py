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
    BatchLoadAudio,
    BatchLoadTextFiles,
    LoadSingleImageFromPath,
    SaveImageToPath,
    SaveAudioToPath,
    SaveTextToPath,
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
from .lora_manager import LoRAManagerWithPreview
from .datetime_node import DateTimeToStringNode
from .detailer_node import DetailerForEachMask
from .lora_switch import LoraSwitchDynamic
from .ultralytics import YoloModelLoader, YoloInference
from .mask_utils import ResizeAllMasks, BlurImageByMasks, OverlayImageByMasks, MergeMasksToOne
from .lora_metadata_parser import LoraMetadataParser
from .switch_nodes import SnJakeAnySwitch, SnJakeImageSwitch, SnJakeMaskSwitch, SnJakeStringSwitch, SnJakeLatentSwitch, SnJakeConditioningSwitch
from .random_node import SnJakeNumberNode
from .text_utils_nodes import SnJakeTextConcatenate, SnJakeMultilineText
from .image_resize_nodes import SnJakeResizeIfLarger
from .qwenrescal import QwenImageResolutionCalc
from .hunyuanrescal import HunyuanImageResolutionCalc
from .image_crop_nodes import SnJakeInteractiveCropLoader, SnJakeImagePatchNode
from .random_prompt_window_selector import SnJakeRandomPromptWindowSelector
from .execution_counter_node import SnJakeExecutionCounter

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
    "BatchLoadAudio": BatchLoadAudio,
    "BatchLoadTextFiles": BatchLoadTextFiles,
    "LoadSingleImageFromPath": LoadSingleImageFromPath,
    "SaveImageToPath": SaveImageToPath,
    "SaveAudioToPath": SaveAudioToPath,
    "SaveTextToPath": SaveTextToPath,
    "ImageRouter": ImageRouter,
    "StringToNumber": StringToNumber,
    "SnJakeStringReplace": StringReplace,
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
    "LoRAManagerWithPreview": LoRAManagerWithPreview,
    "DateTimeToStringNode": DateTimeToStringNode,
    "DetailerForEachMask": DetailerForEachMask,
    "LoraSwitchDynamic": LoraSwitchDynamic,
    "YoloModelLoader": YoloModelLoader,
    "YoloInference": YoloInference,
    "ResizeAllMasks": ResizeAllMasks,
    "BlurImageByMasks": BlurImageByMasks,
    "OverlayImageByMasks": OverlayImageByMasks,
    "MergeMasksToOne": MergeMasksToOne,
    "LoraMetadataParser": LoraMetadataParser,
    "SnJakeAnySwitch": SnJakeAnySwitch,
    "SnJakeImageSwitch": SnJakeImageSwitch,
    "SnJakeMaskSwitch": SnJakeMaskSwitch,
    "SnJakeStringSwitch": SnJakeStringSwitch,
    "SnJakeLatentSwitch": SnJakeLatentSwitch,
    "SnJakeConditioningSwitch": SnJakeConditioningSwitch,
    "SnJakeNumberNode": SnJakeNumberNode,
    "SnJakeTextConcatenate": SnJakeTextConcatenate,
    "SnJakeMultilineText": SnJakeMultilineText,
    "SnJakeResizeIfLarger": SnJakeResizeIfLarger,
    "QwenImageResolutionCalc": QwenImageResolutionCalc,
    "HunyuanImageResolutionCalc": HunyuanImageResolutionCalc,
    "SnJakeInteractiveCropLoader": SnJakeInteractiveCropLoader,
    "SnJakeImagePatchNode": SnJakeImagePatchNode,
    "SnJakeRandomPromptWindowSelector": SnJakeRandomPromptWindowSelector,
    "SnJakeExecutionCounter": SnJakeExecutionCounter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanImageResolutionCalc": "HunyuanImage-2.1 Resolution Calc",
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
    "BatchLoadAudio": "😎 Batch Load Audio",
    "BatchLoadTextFiles": "😎 Batch Load TXT",
    "LoadSingleImageFromPath": "😎 Load Image By Full Path",
    "SaveImageToPath": "😎 Save Image To Path",
    "SaveAudioToPath": "😎 Save Audio To Path",
    "SaveTextToPath": "😎 Save TXT To Path",
    "ImageRouter": "😎 Route Image By Int",
    "StringToNumber": "😎 String->Int/Float",
    "SnJakeStringReplace": "😎 String Replace",
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
    "LoRAManagerWithPreview": "😎 LoRA Manager With Preview",
    "DateTimeToStringNode": "😎 Date Time To String",
    "DetailerForEachMask": "😎 Sequential Mask Detailer",
    "LoraSwitchDynamic": "😎 Lora Switcher",
    "YoloModelLoader": "😎 YOLO Loader",
    "YoloInference": "😎 YOLO Inference",
    "ResizeAllMasks": "😎 Masks Resize",
    "BlurImageByMasks": "😎 Image Blur By Mask",
    "OverlayImageByMasks": "😎 Image Overlay By Mask",
    "MergeMasksToOne": "😎 Merge Masks To One",
    "LoraMetadataParser": "😎 LoRA Metadata Parser",
    "SnJakeAnySwitch": "😎 Switch (Any)",
    "SnJakeImageSwitch": "😎 Switch (Image)",
    "SnJakeMaskSwitch": "😎 Switch (Mask)",
    "SnJakeStringSwitch": "😎 Switch (String)",
    "SnJakeLatentSwitch": "😎 Switch (Latent)",
    "SnJakeConditioningSwitch": "😎 Switch (Conditioning)",
    "SnJakeNumberNode": "😎 Number Node",
    "SnJakeTextConcatenate": "😎 Text Concatenate",
    "SnJakeMultilineText": "😎 Multiline Text",
    "SnJakeResizeIfLarger": "😎 Resize If Larger/Smaller",
    "QwenImageResolutionCalc": "😎 Qwen-Image Resolution Calc",
    "HunyuanImageResolutionCalc": "😎 Hunyuan Image Resolution Calc",
    "SnJakeInteractiveCropLoader": "😎 Load & Crop Image",
    "SnJakeImagePatchNode": "😎 Patch Image Fragment",
    "SnJakeRandomPromptWindowSelector": "😎 Random Prompt Window Selector",
    "SnJakeExecutionCounter": "😎 Execution Counter",
}


WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "WEB_DIRECTORY"]


print("### SnJake Nodes Initialized ###")
