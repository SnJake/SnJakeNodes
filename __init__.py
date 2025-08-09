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

from .lora_switch import LoraSwitchDynamic

from .ultralytics import YoloModelLoader, YoloInference
from .mask_utils import ResizeAllMasks, BlurImageByMasks, OverlayImageByMasks


from .lora_metadata_parser import LoraMetadataParser

from .teleport_nodes import SnJake_TeleportSet, SnJake_TeleportGet

from .switch_nodes import SnJakeAnySwitch, SnJakeImageSwitch, SnJakeMaskSwitch, SnJakeStringSwitch, SnJakeLatentSwitch, SnJakeConditioningSwitch

from .random_node import SnJakeNumberNode

from .text_utils_nodes import SnJakeTextConcatenate, SnJakeMultilineText

from .image_resize_nodes import SnJakeResizeIfLarger

from .sam2_nodes import Sam2Loader, Sam2ImageInference

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
    
    "YoloModelLoader": YoloModelLoader,
    "YoloInference": YoloInference,

    "ResizeAllMasks": ResizeAllMasks,
    "BlurImageByMasks": BlurImageByMasks,
    "OverlayImageByMasks": OverlayImageByMasks,

    "LoraMetadataParser": LoraMetadataParser,


    "SnJake_TeleportSet": SnJake_TeleportSet,
    "SnJake_TeleportGet": SnJake_TeleportGet,

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

    "Sam2Loader": Sam2Loader,
    "Sam2ImageInference": Sam2ImageInference,
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

    "YoloModelLoader": "😎 YOLO Loader",
    "YoloInference": "😎 YOLO Inference",

    "ResizeAllMasks": "😎 Masks Resize",
    "BlurImageByMasks": "😎 Image Blur By Mask",
    "OverlayImageByMasks": "😎 Image Overlay By Mask",

    "LoraMetadataParser": "😎 LoRA Metadata Parser",

    "SnJake_TeleportSet": "😎 Teleport Set (Sender)",
    "SnJake_TeleportGet": "😎 Teleport Get (Receiver)",

    "SnJakeAnySwitch": "😎 Switch (Any)",
    "SnJakeImageSwitch": "😎 Switch (Image)",
    "SnJakeMaskSwitch": "😎 Switch (Mask)",
    "SnJakeStringSwitch": "😎 Switch (String)",
    "SnJakeLatentSwitch": "😎 Switch (Latent)",
    "SnJakeConditioningSwitch": "😎 Switch (Conditioning)",

    "SnJakeNumberNode": "😎 Number Node",

    "SnJakeTextConcatenate": "😎 Text Concatenate",
    "SnJakeMultilineText": "😎 Multiline Text",

    "SnJakeResizeIfLarger": "😎 Resize If Larger",

    "Sam2Loader": "😎 Sam2 Loader",
    "Sam2ImageInference": "😎 Sam2 Image Inference",
}


WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "WEB_DIRECTORY"]


print("### SnJake Nodes Initialized ###")







