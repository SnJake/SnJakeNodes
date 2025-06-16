# Файл: pixelart_postprocess_node.py (или как вы его назовете)
import torch
import numpy as np
from PIL import Image, ImageOps
import os
import folder_paths # ComfyUI specific import
import traceback
import re # Для парсинга цвета контура
# Убедитесь, что sklearn установлен для KMeans (если вы его используете)
try:
    from sklearn.cluster import KMeans
    from scipy.spatial import KDTree
    SKLEARN_AVAILABLE = True
except ImportError:
    print("[PixelArtPostProcessNode] Warning: scikit-learn or scipy not installed. Palette matching features will be disabled.")
    SKLEARN_AVAILABLE = False
    # Заглушки, чтобы код не падал
    class KMeans: pass
    class KDTree: pass


# --- Вспомогательные функции (из предыдущего узла) ---
def comfy_tensor_to_pil(tensor):
    # ... (код тот же) ...
    if not isinstance(tensor, torch.Tensor): raise TypeError("Input must be a torch.Tensor")
    if tensor.ndim != 4: raise ValueError("Input tensor must have 4 dimensions [B, H, W, C]")
    tensor = tensor.cpu()
    images = []
    for i in range(tensor.shape[0]):
        img_np = tensor[i].numpy()
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        if img_np.shape[-1] == 1: images.append(Image.fromarray(img_np.squeeze(-1), 'L').convert('RGB'))
        else: images.append(Image.fromarray(img_np, 'RGB'))
    return images

def pil_to_comfy_tensor(images):
    # ... (код тот же) ...
    if not isinstance(images, list): images = [images]
    tensors = []
    for img in images:
        if not isinstance(img, Image.Image): raise TypeError("Input must be a PIL Image or a list of PIL Images")
        if img.mode != 'RGB': img = img.convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(img_np))
    if not tensors: return torch.zeros((0, 1, 1, 3))
    return torch.stack(tensors, dim=0)


# --- Функции для постобработки ---

def extract_palette_kmeans(image_pil, n_colors=16):
    """Extracts dominant colors using KMeans."""
    if not SKLEARN_AVAILABLE: return None
    if not isinstance(image_pil, Image.Image): return None
    try:
        img = image_pil.convert('RGB')
        img.thumbnail((128, 128)) # Resize for speed
        img_array = np.array(img, dtype=np.float64) / 255.0
        w, h, d = img_array.shape
        if w*h == 0: return None # Handle empty image
        img_flat = img_array.reshape((w * h, d))

        unique_colors = np.unique(img_flat, axis=0)
        actual_n_colors = min(n_colors, len(unique_colors))

        if actual_n_colors < 2: # If only 1 color or empty
            return (unique_colors * 255.0).astype(np.uint8) if len(unique_colors) > 0 else np.array([[0,0,0]], dtype=np.uint8) # Return the single color or black

        kmeans = KMeans(n_clusters=actual_n_colors, random_state=0, n_init=10).fit(img_flat)
        palette = kmeans.cluster_centers_ * 255.0
        return palette.astype(np.uint8)
    except Exception as e:
        print(f"[PixelArtPostProcessNode] Error extracting palette: {e}")
        traceback.print_exc()
        return None

def apply_palette_kdtree(image_pil, palette):
    """Applies a fixed palette to an image using KDTree for nearest color."""
    if not SKLEARN_AVAILABLE: return image_pil # Cannot apply without scipy
    if not isinstance(image_pil, Image.Image) or palette is None or len(palette) == 0:
        return image_pil
    try:
        img = image_pil.convert('RGB')
        img_array = np.array(img)
        w, h, d = img_array.shape
        if w*h == 0: return image_pil # Handle empty image
        pixels = img_array.reshape(-1, d)

        if len(palette) == 1: # Special case for single color palette
             new_pixels = np.tile(palette[0], (pixels.shape[0], 1))
        else:
             tree = KDTree(palette)
             distances, indices = tree.query(pixels)
             new_pixels = palette[indices]

        new_img_array = new_pixels.reshape(w, h, d).astype(np.uint8)
        return Image.fromarray(new_img_array, 'RGB')
    except Exception as e:
        print(f"[PixelArtPostProcessNode] Error applying palette: {e}")
        traceback.print_exc()
        return image_pil

# <<<--- НОВАЯ ФУНКЦИЯ ---<<<
def apply_contour_expansion(image_pil, outline_color=(0, 0, 0), connectivity=4):
    """Adds outlines by changing boundary pixels to outline_color."""
    if not isinstance(image_pil, Image.Image): return image_pil
    try:
        img_array = np.array(image_pil.convert('RGB'))
        output_array = img_array.copy() # Work on a copy
        h, w = img_array.shape[:2]
        outline_color_np = np.array(outline_color, dtype=np.uint8)

        for y in range(h):
            for x in range(w):
                current_color = img_array[y, x]

                # Skip if already outline color (optional: could be handled differently)
                if np.array_equal(current_color, outline_color_np):
                    continue

                is_boundary = False
                # Define neighbors based on connectivity
                if connectivity == 4:
                    neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
                elif connectivity == 8:
                    neighbors = [
                        (y - 1, x - 1), (y - 1, x), (y - 1, x + 1),
                        (y, x - 1),                 (y, x + 1),
                        (y + 1, x - 1), (y + 1, x), (y + 1, x + 1)
                    ]
                else: # Default to 4 if invalid connectivity
                    neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]

                for ny, nx in neighbors:
                    # Check boundary conditions
                    if 0 <= ny < h and 0 <= nx < w:
                        neighbor_color = img_array[ny, nx]
                        # Check if neighbor is different and NOT the outline color itself
                        if not np.array_equal(current_color, neighbor_color) and \
                           not np.array_equal(neighbor_color, outline_color_np):
                            is_boundary = True
                            break # Found a boundary neighbor

                if is_boundary:
                    output_array[y, x] = outline_color_np # Change pixel to outline color

        return Image.fromarray(output_array, 'RGB')

    except Exception as e:
        print(f"[PixelArtPostProcessNode] Error applying contour expansion: {e}")
        traceback.print_exc()
        return image_pil
# >>>----------------------<<<


# --- Узел Постобработки ---

class PixelArtPostProcessNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Check if scikit-learn is available to enable/disable palette matching
        palette_options_enabled = {"default": False}
        palette_options_disabled = {"default": False, "disabled": "disabled"} # ComfyUI doesn't have a standard 'disabled' key, use tooltip instead?

        return {
            "required": {
                "image": ("IMAGE",), # Выход из AdvancedPixelArtNode
                # Palette Matching Options
                "palette_matching": ("BOOLEAN", palette_options_enabled if SKLEARN_AVAILABLE else palette_options_disabled),
                "palette_colors": ("INT", {"default": 16, "min": 2, "max": 256, "step": 1}),
                "palette_source": (["processed_image", "original_image"], {"default": "processed_image"}),
                # Contour Expansion Options
                "contour_expansion": ("BOOLEAN", {"default": False}),
                "outline_color": ("STRING", {"default": "0,0,0"}),
                "connectivity": (["4", "8"], {"default": "4"}),
            },
            "optional": {
                "original_image": ("IMAGE",), # Для извлечения палитры
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("postprocessed_image",)
    FUNCTION = "execute"
    CATEGORY = "😎 SnJake/PixelArt"

    def execute(self, image, palette_matching, palette_colors, palette_source,
                contour_expansion, outline_color, connectivity, original_image=None):

        pil_images = comfy_tensor_to_pil(image)
        original_pil_list = comfy_tensor_to_pil(original_image) if original_image is not None else []

        result_images = []
        batch_size = len(pil_images)

        # Parse outline color once
        try:
            r, g, b = map(int, re.findall(r'\d+', outline_color))
            parsed_outline_color = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
        except Exception:
            print(f"[PixelArtPostProcessNode] Warning: Invalid outline_color format '{outline_color}'. Using black (0,0,0).")
            parsed_outline_color = (0, 0, 0)

        # Parse connectivity
        try:
            parsed_connectivity = int(connectivity)
            if parsed_connectivity not in [4, 8]:
                 print(f"[PixelArtPostProcessNode] Warning: Invalid connectivity '{connectivity}'. Using 4.")
                 parsed_connectivity = 4
        except ValueError:
            print(f"[PixelArtPostProcessNode] Warning: Invalid connectivity format '{connectivity}'. Using 4.")
            parsed_connectivity = 4


        print(f"[PixelArtPostProcessNode] Processing batch of {batch_size} images...")
        for i, pil_img in enumerate(pil_images):
            print(f"  - Processing image {i+1}/{batch_size}...")
            processed_pil = pil_img
            current_original_pil = original_pil_list[i] if i < len(original_pil_list) else None

            # 1. Palette Matching
            if palette_matching and SKLEARN_AVAILABLE:
                print("    - Applying Palette Matching...")
                # Determine source image for palette extraction
                palette_src = None
                if palette_source == "original_image" and current_original_pil:
                    palette_src = current_original_pil
                    print(f"    - Extracting palette ({palette_colors} colors) from original image...")
                elif palette_source == "processed_image":
                    palette_src = processed_pil
                    print(f"    - Extracting palette ({palette_colors} colors) from processed image...")
                else:
                    print(f"    - Warning: Cannot extract palette. Source '{palette_source}' invalid or missing.")

                if palette_src:
                    palette = extract_palette_kmeans(palette_src, n_colors=palette_colors)
                    if palette is not None and len(palette) > 0:
                        print(f"    - Applying extracted palette (found {len(palette)} colors)...")
                        processed_pil = apply_palette_kdtree(processed_pil, palette)
                    else:
                        print("    - Warning: Failed to extract palette.")
            elif palette_matching and not SKLEARN_AVAILABLE:
                 print("    - Warning: Cannot apply Palette Matching, scikit-learn/scipy not available.")


            # 2. Contour Expansion
            if contour_expansion:
                print(f"    - Applying Contour Expansion (Color: {parsed_outline_color}, Conn: {parsed_connectivity})...")
                processed_pil = apply_contour_expansion(processed_pil, outline_color=parsed_outline_color, connectivity=parsed_connectivity)

            result_images.append(processed_pil)

        # Convert back to tensor
        final_batch_tensor = pil_to_comfy_tensor(result_images)
        print(f"[PixelArtPostProcessNode] Post-processing finished. Output shape: {final_batch_tensor.shape}")
        return (final_batch_tensor,)