# SnJake's Custom Nodes for ComfyUI

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![Made for ComfyUI](https://img.shields.io/badge/Made%20for-ComfyUI-blueviolet)

This repository contains a versatile collection of custom nodes for ComfyUI, designed to extend its capabilities in image processing, workflow management, and creative effects. From advanced pixel art tools to powerful utility nodes, this pack aims to enhance your creative pipeline.

---

## 🚀 Installation

You can install this node pack by following this method:

### Method 1: Using Git

1.  Open a terminal or command prompt.
2.  Navigate to your ComfyUI `custom_nodes` directory.
    ```bash
    # Example for Windows
    cd D:\ComfyUI\custom_nodes\
    
    # Example for Linux/macOS
    cd ~/ComfyUI/custom_nodes/
    ```

3.  Clone this repository into the `custom_nodes` folder:
    ```bash
    git clone https://github.com/SnJake/SnJakeNodes.git
    ```

4.  **Install Dependencies**: Now, you need to install the required Python packages. The command depends on which version of ComfyUI you are using.

    *   **For standard ComfyUI installations (with venv):**
        1.  Make sure your ComfyUI virtual environment (`venv`) is activated.
        2.  Navigate into the new node directory and install the requirements:
            ```bash
            cd SnJakeNodes
            pip install -r requirements.txt
            ```

    *   **For Portable ComfyUI installations:**
        1.  Navigate back to the **root** of your portable ComfyUI directory (e.g., `D:\ComfyUI_windows_portable`).
        2.  Run the following command to use the embedded Python to install the requirements. *Do not activate any venv.*
            ```bash
            python_embeded\python.exe -m pip install -r custom_nodes\SnJakeNodes\requirements.txt
            ```

5.  **Restart ComfyUI**: Close the terminal and restart ComfyUI completely. The new `😎 SnJake` nodes will be available in the "Add Node" menu.

## ✨ Nodes Included

All nodes can be found in the "Add Node" menu under the **`😎 SnJake/...`** category.

### 🎨 Pixel Art & Color
- **Pixel Art Effect** (`pixel_art_node.py`): A comprehensive node for creating pixel art with advanced controls for color quantization, dithering, and downscaling.
- **Pixel Art Post Process** (`pixelart_postprocess_node.py`): Applies post-processing effects like palette matching and contour expansion to pixel art images.
- **Segmentation Pixel Art** (`segmentation_pixel_art_node.py`): Creates a stylized pixel art effect using SLIC segmentation to average colors within superpixels.
- **Color Palette** (`color_palette_image_node.py`): Generates a visual image of a color palette from a list of HEX codes.
- **Color Compression** (`color_palette_compression_node.py`): Reduces a given color palette to a smaller, representative set of colors using KMeans clustering.
- **Extract Palette** (`pixel_art_utils.py`): Extracts a color palette from a source image using various quantization methods.
- **Apply Palette (and Dither)** (`pixel_art_utils.py`): Applies a given color palette to an image, with optional dithering.
- **Replace Palette Colors** (`pixel_art_utils.py`): Replaces the colors of an image based on a source and a replacement palette, with sorting options.
- **Color Merge** (`region_merging_node.py`): Merges small, single-color regions in an image into larger neighboring regions to clean up pixel art.

### 🔧 Utilities & Workflow
- **Batch Load Images** (`utils_nodes.py`): Loads images from a directory with various modes (incremental, random, single).
- **Load Image By Full Path** (`utils_nodes.py`): Loads a single image from an absolute file path.
- **Save Image To Path** (`utils_nodes.py`): Saves an image to a specified absolute path, with an option to embed the workflow.
- **List Image Files In Folder** (`list_image.py`): Scans a folder and outputs a list of full image paths.
- **Simple List Image Files** (`list_image_not.py`): Scans a folder and outputs filenames and full paths as separate multiline strings.
- **Concatenate Images By Directory** (`image_concatenate.py`): Groups images by directory and concatenates them based on a specified base image.
- **Image Mask Selector** (`mask_selector_node.py`): Routes an image to one of two outputs based on whether the input mask is empty or not.
- **Multiline Prompt Selector** (`prompt_selector.py`): Selects a specific line from a multiline text block.
- **Date Time To String** (`datetime_node.py`): Outputs the current date and time as a formatted string, perfect for unique filenames.
- **Route Image By Int** (`utils_nodes.py`): A simple router that directs an image to one of 10 outputs based on an integer input.
- **String->Int/Float** (`utils_nodes.py`): Converts a string to integer and float values.
- **String Replace** (`utils_nodes.py`): Performs a simple find-and-replace operation on a string.
- **Random Int/Float** (`utils_nodes.py`): Generates a random integer or float within a specified range.

### 🖼️ Image Adjustments & Effects
- **Image Adjustment Node** (`color_adjusment.py`): Adjusts temperature, hue, brightness, contrast, saturation, gamma, and midtones.
- **Color Balance** (`color_balance.py`): Provides Lift, Gamma, and Gain controls similar to professional color grading tools.
- **Image Resize Node** (`image_resize_node.py`): A powerful image resizing node with multiple resampling methods, including `bislerp`.
- **CRT Effect** (`crt_effect_node.py`): Simulates a retro Cathode-Ray Tube (CRT) monitor effect with scanlines and pixel structure.
- **Liminal Effects Node** (`liminal_effects_node.py`): A multi-effect node for creating eerie, liminal space aesthetics with noise, chromatic aberration, fog, VHS glitches, and more.
- **Expand Image Right** (`expand_image_right.py`): Doubles the width of an image, filling the new right half with white and providing a corresponding mask for outpainting.

### 🎭 Masking & Detailing
- **Custom Mask Node** (`custom_mask_node.py`): Creates a rectangular mask with a specified size, position, and blur on a larger canvas.
- **Sequential Mask Detailer** (`detailer_node.py`): An advanced detailer that processes multiple masks sequentially, applying a KSampler pass to each region for targeted improvements.

### 🤖 AI & API Integration
- **VLM Api Node** (`api.py`): A Vision-Language Model node that sends an image and a text prompt to an OpenAI-compatible API (like GPT-4o) and returns the text description.
- **AnyNode OpenAI** (`anynode_snjake.py`): A powerful node that takes a user task and input data, sends it to GPT-4o to generate Python code, and executes it.
- **AnyNode Local LLM** (`anynode_snjake.py`): Same as AnyNode, but configured to work with a local OpenAI-compatible API endpoint.

### 🧪 Experimental & Advanced
- **Lora Loader With Preview** (`lora_loader_preview.py`): A LoRA loader that includes a visual preview of the LoRA's associated image in the ComfyUI interface.
- **XY Plot Advanced** (`xy_plot_node.py`): An advanced XY plot node for generating a grid of images by varying two parameters (e.g., Seed, CFG, Steps, Prompt S/R). Includes built-in Hires Fix.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
