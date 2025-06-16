import torch
import kornia
import kornia.color as kc
import numpy as np
import traceback
import math

def to_quantize_space(image, space):
    """Converts image (RGB [0,1]) to the target quantization space."""
    # (Скопируйте код _to_quantize_space из вашего узла)
    image = image.float().clamp(0, 1) # Вход всегда RGB [0,1]
    try:
        if space == "RGB": return image
        elif space == "HSV": return kc.rgb_to_hsv(image) # H:[0,2pi], S:[0,1], V:[0,1]
        elif space == "LAB": return kc.rgb_to_lab(image) # L:[0,100], a,b:[-100,100] approx
        elif space == "YCbCr": return kc.rgb_to_ycbcr(image) # Y,Cb,Cr:[0,1] approx
        else:
             print(f"Warning: Unknown target space '{space}'. Returning RGB.")
             return image
    except Exception as e:
        print(f"Warning: Error converting TO {space}: {e}. Returning RGB.")
        traceback.print_exc()
        return image # Fallback to RGB

def from_quantize_space(image, space):
    """Converts image from quantization space back to RGB [0,1]."""
    # (Скопируйте код _from_quantize_space из вашего узла)
    image = image.float() # Вход - данные в `space`
    try:
        if space == "RGB": return image.clamp(0, 1) # На всякий случай clamp
        # Kornia ожидает специфичные диапазоны на входе
        elif space == "HSV": return kc.hsv_to_rgb(image).clamp(0.0, 1.0)
        elif space == "LAB": return kc.lab_to_rgb(image).clamp(0.0, 1.0)
        elif space == "YCbCr": return kc.ycbcr_to_rgb(image).clamp(0.0, 1.0)
        else:
             print(f"Warning: Unknown source space '{space}'. Returning clamped input.")
             return image.clamp(0, 1) # Если неизвестно, предполагаем RGB
    except Exception as e:
        print(f"Warning: Error converting FROM {space}: {e}. Returning as is (clamped 0-1).")
        traceback.print_exc()
        # Попытка вернуть хоть что-то, зажав в [0,1], но это может быть некорректно
        return image.clamp(0.0, 1.0)

def convert_palette_to_string(color_palette):
    """Converts a palette tensor (RGB [0,1]) to a hex string."""
    # (Скопируйте код _convert_palette_to_string из вашего узла)
    if color_palette is None: return "N/A"
    # Ensure tensor is on CPU and NumPy format for processing
    if isinstance(color_palette, torch.Tensor):
        palette_cpu = color_palette.detach().cpu().numpy()
    elif isinstance(color_palette, np.ndarray):
        palette_cpu = color_palette
    else:
        print("Warning: Invalid palette type for string conversion.")
        return "Invalid Palette Type"

    hex_colors = []
    # Handle potential single color (1D array) vs multiple colors (2D array)
    if palette_cpu.ndim == 1:
        palette_cpu = palette_cpu[np.newaxis, :] # Make it 2D

    for color in palette_cpu:
        # Clamp color values to [0, 1] before scaling
        color = np.clip(color, 0, 1)
        # Scale to 0-255 and convert to int
        r, g, b = (color * 255).round().astype(int)
        # Format as hex string
        hex_colors.append('#{:02X}{:02X}{:02X}'.format(r, g, b))

    return ', '.join(hex_colors)

def calculate_dbi(pixels, labels, centroids):
    """Calculates the Davies-Bouldin Index."""
    # (Скопируйте код _davies_bouldin_index из вашего узла)
    n_clusters = centroids.shape[0]
    if n_clusters < 2:
        return float('inf') # Undefined for < 2 clusters

    device = pixels.device
    # Intra-cluster dispersions (average distance to centroid)
    intra_dispersions = torch.zeros(n_clusters, device=device, dtype=torch.float64) # Use float64
    cluster_sizes = torch.zeros(n_clusters, device=device, dtype=torch.long)

    for i in range(n_clusters):
        cluster_points = pixels[labels == i]
        cluster_sizes[i] = cluster_points.shape[0]
        if cluster_sizes[i] > 0:
            # Ensure centroids are float for distance calculation
            distances = torch.norm(cluster_points.float() - centroids[i].float(), p=2, dim=1) # Euclidean distance
            intra_dispersions[i] = distances.mean() # Mean preserves float64

    # Inter-cluster distances
    # Add epsilon to avoid division by zero if centroids are identical
    centroid_distances = torch.cdist(centroids.float(), centroids.float(), p=2) + 1e-9 # Use float, slightly larger epsilon

    db_index = 0.0
    valid_clusters_count = 0
    for i in range(n_clusters):
        if cluster_sizes[i] == 0: continue # Skip empty clusters for outer loop

        max_ratio = 0.0
        found_comparison = False
        for j in range(n_clusters):
            if i != j and cluster_sizes[j] > 0: # Compare only with non-empty clusters
                # Ensure division is done in float64 for precision
                ratio_numerator = intra_dispersions[i] + intra_dispersions[j]
                ratio_denominator = centroid_distances[i, j].double() # Cast denominator to double

                # Check for near-zero denominator
                if ratio_denominator < 1e-9:
                    ratio = torch.tensor(float('inf'), device=device, dtype=torch.float64) # Assign inf if centroids are too close
                else:
                    ratio = ratio_numerator / ratio_denominator

                if torch.isfinite(ratio): # Check if ratio is valid
                    if ratio > max_ratio:
                        max_ratio = ratio
                    found_comparison = True

        # Only add to db_index if we could compare with at least one other cluster
        if found_comparison:
            # max_ratio can be tensor, ensure it's added as float64
            db_index += max_ratio.item() if isinstance(max_ratio, torch.Tensor) else max_ratio
            valid_clusters_count += 1

    if valid_clusters_count == 0:
        return float('inf') # No valid comparisons possible

    final_dbi = db_index / valid_clusters_count

    if not math.isfinite(final_dbi):
        print("Warning: DBI calculation resulted in non-finite value. Returning Inf.")
        return float('inf')

    return final_dbi