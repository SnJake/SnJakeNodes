# filename: pixel_art_logic.py
import torch
import torch.nn.functional as F
import kornia
import kornia.color as kc
import kornia.filters
import numpy as np
import math
import traceback

# --- Константы для Дизеринга ---
BAYER_2 = torch.tensor([[0, 2], [3, 1]], dtype=torch.float32) / 4.0 - 0.5
BAYER_4 = torch.tensor([
    [ 0,  8,  2, 10], [12,  4, 14,  6], [ 3, 11,  1,  9], [15,  7, 13,  5]
], dtype=torch.float32) / 16.0 - 0.5
BAYER_8 = torch.tensor([
    [ 0, 32,  8, 40,  2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26], [12, 44,  4, 36, 14, 46,  6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22], [ 3, 35, 11, 43,  1, 33,  9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47,  7, 39, 13, 45,  5, 37], [63, 31, 55, 23, 61, 29, 53, 21]
], dtype=torch.float32) / 64.0 - 0.5
HALFTONE_6X6 = torch.tensor([
    [34, 25, 21, 17, 29, 33], [30, 13,  9,  5, 16, 24], [18,  6,  1,  0,  8, 20],
    [22, 10,  2,  3, 12, 19], [26, 14,  7,  4, 11, 23], [35, 31, 27, 15, 28, 32]
], dtype=torch.float32) / 36.0 - 0.5
HALFTONE_8X8 = torch.tensor([
    [ 0, 48, 12, 60,  3, 51, 15, 63], [32, 16, 44, 28, 35, 19, 47, 31],
    [ 8, 56,  4, 52, 11, 59,  7, 55], [40, 24, 36, 20, 43, 27, 39, 23],
    [ 2, 50, 14, 62,  1, 49, 13, 61], [34, 18, 46, 30, 33, 17, 45, 29],
    [10, 58,  6, 54,  9, 57,  5, 53], [42, 26, 38, 22, 41, 25, 37, 21]
], dtype=torch.float32) / 64.0 - 0.5

ORDERED_PATTERNS = {
    "OrderedBayer2": BAYER_2, "OrderedBayer4": BAYER_4, "OrderedBayer8": BAYER_8,
    "OrderedHalftone6x6": HALFTONE_6X6, "OrderedHalftone8x8": HALFTONE_8X8
}

DIFFUSION_PATTERNS = {
    # (dx, dy, factor)
    "Floyd-Steinberg":      [(1, 0, 7/16), (-1, 1, 3/16), (0, 1, 5/16), (1, 1, 1/16)],
    "Jarvis-Judice-Ninke":  [(1, 0, 7/48), (2, 0, 5/48), (-2, 1, 3/48), (-1, 1, 5/48), (0, 1, 7/48), (1, 1, 5/48), (2, 1, 3/48), (-2, 2, 1/48), (-1, 2, 3/48), (0, 2, 5/48), (1, 2, 3/48), (2, 2, 1/48)],
    "Stucki":               [(1, 0, 8/42), (2, 0, 4/42), (-2, 1, 2/42), (-1, 1, 4/42), (0, 1, 8/42), (1, 1, 4/42), (2, 1, 2/42), (-2, 2, 1/42), (-1, 2, 2/42), (0, 2, 4/42), (1, 2, 2/42), (2, 2, 1/42)],
    "Atkinson":             [(1, 0, 1/8), (2, 0, 1/8), (-1, 1, 1/8), (0, 1, 1/8), (1, 1, 1/8), (0, 2, 1/8)],
    "Burkes":               [(1, 0, 8/32), (2, 0, 4/32), (-2, 1, 2/32), (-1, 1, 4/32), (0, 1, 8/32), (1, 1, 4/32), (2, 1, 2/32)],
    "Sierra":               [(1, 0, 5/32), (2, 0, 3/32), (-2, 1, 2/32), (-1, 1, 4/32), (0, 1, 5/32), (1, 1, 4/32), (2, 1, 2/32), (-1, 2, 2/32), (0, 2, 3/32), (1, 2, 2/32)],
    "Two-Row-Sierra":       [(1, 0, 4/16), (2, 0, 3/16), (-2, 1, 1/16), (-1, 1, 2/16), (0, 1, 3/16), (1, 1, 2/16), (2, 1, 1/16)],
    "Sierra-Lite":          [(1, 0, 2/4), (-1, 1, 1/4), (0, 1, 1/4)]
    # WhiteNoise handled separately in apply_dithering
}

# --- Функции преобразования цветовых пространств ---
def to_quantize_space(image, space):
    """Конвертирует тензор RGB [0,1] в указанное цветовое пространство."""
    image = image.float().clamp(0, 1) # Вход всегда RGB [0,1]
    try:
        if space == "RGB": return image
        elif space == "HSV": return kc.rgb_to_hsv(image) # H:[0,2pi], S:[0,1], V:[0,1]
        elif space == "LAB": return kc.rgb_to_lab(image) # L:[0,100], a,b:[-100,100] approx
        elif space == "YCbCr": return kc.rgb_to_ycbcr(image) # Y,Cb,Cr:[0,1] approx
        else:
            print(f"Warning: Unknown target color space '{space}'. Returning RGB.")
            return image
    except Exception as e:
        print(f"Warning: Error converting TO {space}: {e}. Returning RGB.")
        traceback.print_exc()
        return image # Fallback to RGB

def from_quantize_space(image, space):
    """Конвертирует тензор из указанного пространства обратно в RGB [0,1]."""
    image = image.float() # Вход - данные в `space`
    try:
        if space == "RGB": return image.clamp(0, 1)
        elif space == "HSV": return kc.hsv_to_rgb(image).clamp(0.0, 1.0)
        elif space == "LAB": return kc.lab_to_rgb(image).clamp(0.0, 1.0)
        elif space == "YCbCr": return kc.ycbcr_to_rgb(image).clamp(0.0, 1.0)
        else:
            print(f"Warning: Unknown source color space '{space}'. Assuming RGB and clamping.")
            return image.clamp(0, 1) # Если неизвестно, предполагаем RGB
    except Exception as e:
        print(f"Warning: Error converting FROM {space}: {e}. Returning as is (clamped 0-1).")
        traceback.print_exc()
        # Попытка вернуть хоть что-то, зажав в [0,1]
        return image.clamp(0.0, 1.0)

# --- Функции K-Means ---
def kmeans_plus_plus_initialization(pixels, num_colors):
    """Инициализация центроидов K-Means++."""
    num_pixels, num_features = pixels.shape
    device = pixels.device
    centroids = torch.empty((num_colors, num_features), dtype=pixels.dtype, device=device)
    if num_pixels == 0: return centroids # Handle empty input

    # 1. Choose one center uniformly at random
    first_centroid_idx = torch.randint(0, num_pixels, (1,), device=device)
    centroids[0] = pixels[first_centroid_idx]

    # 2. For each subsequent center
    min_sq_distances = torch.full((num_pixels,), float('inf'), dtype=torch.float64, device=device)

    for c in range(1, num_colors):
        # Calculate squared distances from points to the *most recently added* center
        dist_sq = torch.sum((pixels.double() - centroids[c-1:c].double())**2, dim=1)
        min_sq_distances = torch.minimum(min_sq_distances, dist_sq)

        # 3. Choose new center with probability proportional to D(x)^2
        min_sq_distances_sum = torch.sum(min_sq_distances)

        if min_sq_distances_sum <= 1e-9:
             next_centroid_idx = torch.randint(0, num_pixels, (1,), device=device)
        else:
             probabilities = min_sq_distances / min_sq_distances_sum
             try:
                  next_centroid_idx = torch.multinomial(probabilities, 1)
             except RuntimeError as multi_err:
                  print(f"KMeans++ Multinomial Error ({multi_err}). Sum={min_sq_distances_sum}. Prob Sum: {probabilities.sum()}. Falling back to argmax.")
                  # Fallback: choose the point with max probability (max distance)
                  next_centroid_idx = torch.argmax(probabilities)

        centroids[c] = pixels[next_centroid_idx]
        # Optional optimization: set distance of chosen point to 0? Might not be worth it.
        # min_sq_distances[next_centroid_idx] = 0.0

    return centroids.to(pixels.dtype) # Return in original dtype

def kmeans_quantization(pixels, num_colors, max_iters):
    """Квантование цветов с использованием K-Means."""
    device = pixels.device
    if pixels.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=device), \
               torch.empty((0, pixels.shape[1]), dtype=pixels.dtype, device=device)

    # Handle cases where num_colors >= num_pixels or only 1 color needed
    if num_colors <= 1 or pixels.shape[0] <= num_colors:
         unique_colors, inverse_indices = torch.unique(pixels, dim=0, return_inverse=True)
         if num_colors <= 1:
              centroid = pixels.mean(dim=0, keepdim=True)
              labels = torch.zeros(pixels.shape[0], dtype=torch.long, device=device)
              return labels, centroid
         else: # num_pixels <= num_colors, return unique colors
              return inverse_indices, unique_colors

    # K-Means++ Initialization
    try:
         centroids = kmeans_plus_plus_initialization(pixels.float(), num_colors)
    except Exception as init_err:
         print(f"KMeans++ Init Error: {init_err}. Falling back to random init.")
         indices = torch.randperm(pixels.shape[0], device=device)[:num_colors]
         centroids = pixels[indices].float()

    pixels_float = pixels.float() # Ensure float for calculations
    for i in range(max_iters):
        # Assign labels
        distances = torch.cdist(pixels_float, centroids)
        labels = torch.argmin(distances, dim=1)

        # Update centroids using scatter_add_ for efficiency
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(num_colors, device=device, dtype=torch.long) # Use long for counts

        # Scatter add needs matching types usually, let's accumulate in float64?
        # Or use a loop? Loop might be slower but safer for type mismatches.
        # Let's try loop for safety.
        # for k in range(num_colors):
        #    mask = labels == k
        #    points_in_cluster = pixels_float[mask]
        #    if points_in_cluster.shape[0] > 0:
        #        new_centroids[k] = points_in_cluster.mean(dim=0)
        #        counts[k] = points_in_cluster.shape[0]

        # Trying scatter_add_ again, ensuring indices are long
        labels_expanded = labels.unsqueeze(1).expand_as(pixels_float)
        new_centroids.scatter_add_(0, labels_expanded, pixels_float)

        # Count points using bincount (more efficient than scatter_add_ for counts)
        counts = torch.bincount(labels, minlength=num_colors).long()

        # Avoid division by zero for empty clusters
        counts_safe = counts.clamp(min=1).float() # Use float for division
        new_centroids /= counts_safe.unsqueeze(1)

        # Handle empty clusters
        empty_cluster_mask = (counts == 0)
        num_empty = empty_cluster_mask.sum().item()

        if num_empty > 0:
            if num_empty < pixels_float.shape[0]:
                 non_empty_centroids = new_centroids[~empty_cluster_mask]
                 if non_empty_centroids.shape[0] > 0:
                      all_dists = torch.cdist(pixels_float, non_empty_centroids)
                      min_dists_to_non_empty, _ = torch.min(all_dists, dim=1)
                      furthest_points_indices = torch.topk(min_dists_to_non_empty, k=num_empty).indices
                      new_centroids[empty_cluster_mask] = pixels_float[furthest_points_indices]
                 else:
                      indices = torch.randperm(pixels.shape[0], device=device)[:num_colors]
                      new_centroids = pixels[indices].float()
            else:
                 indices = torch.randperm(pixels.shape[0], device=device)[:num_colors]
                 new_centroids = pixels[indices].float()

        # Check for convergence
        if torch.allclose(centroids, new_centroids, atol=1e-5, rtol=1e-4):
            centroids = new_centroids
            break
        centroids = new_centroids

    # Final label assignment
    distances = torch.cdist(pixels_float, centroids)
    labels = torch.argmin(distances, dim=1)

    return labels, centroids

# --- Функции Median Cut ---
class Box:
    """Вспомогательный класс для Median Cut."""
    def __init__(self, data):
        self.data = data # (num_pixels_in_box, C)
        if data.shape[0] > 0:
            self.min_vals = torch.min(data, dim=0)[0]
            self.max_vals = torch.max(data, dim=0)[0]
            self.ranges = torch.clamp(self.max_vals - self.min_vals, min=0.0)
            max_range_val = torch.max(self.ranges)
            if max_range_val > 1e-8: # Use tolerance
                self.longest_dim = torch.argmax(self.ranges).item()
            else:
                self.longest_dim = 0 # Fallback
            self.num_pixels = data.shape[0]
        else:
            # Initialize safely for empty box
            C = data.shape[1] if data.ndim == 2 else 3 # Default to 3 channels if unknown
            self.min_vals = torch.zeros(C, device=data.device, dtype=data.dtype)
            self.max_vals = torch.zeros(C, device=data.device, dtype=data.dtype)
            self.ranges = torch.zeros(C, device=data.device, dtype=data.dtype)
            self.longest_dim = 0
            self.num_pixels = 0

    def __lt__(self, other):
        # Prioritize splitting boxes with more pixels * largest range
        my_metric = self.ranges[self.longest_dim] * self.num_pixels if self.num_pixels > 0 else -1
        other_metric = other.ranges[other.longest_dim] * other.num_pixels if other.num_pixels > 0 else -1
        # Negative because heapq is a min-heap, but we sort lists directly (reverse=True)
        return my_metric < other_metric # For list sort reverse=True

def median_cut_quantization(pixels, num_colors):
    """Квантование цветов методом Median Cut."""
    device = pixels.device
    if pixels.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=device), \
               torch.empty((0, pixels.shape[1]), dtype=pixels.dtype, device=device)
    if num_colors <= 1:
        centroid = pixels.mean(dim=0, keepdim=True)
        labels = torch.zeros(pixels.shape[0], dtype=torch.long, device=device)
        return labels, centroid

    initial_box = Box(pixels)
    boxes = [initial_box]
    while len(boxes) < num_colors:
        boxes.sort(reverse=True) # Find box with largest metric
        box_to_split = boxes.pop(0)

        if box_to_split.num_pixels <= 1 or torch.max(box_to_split.ranges) <= 1e-8: # Cannot split further
            boxes.append(box_to_split)
            break

        dim_to_split = box_to_split.longest_dim
        try:
            # Use median for splitting point
            median_val = torch.median(box_to_split.data[:, dim_to_split])[0]
            # Split into two groups based on median
            mask_le = box_to_split.data[:, dim_to_split] <= median_val
            mask_gt = box_to_split.data[:, dim_to_split] > median_val

            # Handle cases where median splits perfectly or results in one empty box
            if mask_le.sum() == 0 or mask_gt.sum() == 0:
                 # Fallback to midpoint split if median fails (e.g., all values equal)
                 mid_point = (box_to_split.min_vals[dim_to_split] + box_to_split.max_vals[dim_to_split]) / 2.0
                 mask_le = box_to_split.data[:, dim_to_split] <= mid_point
                 mask_gt = box_to_split.data[:, dim_to_split] > mid_point
                 # If still fails, force split near middle index after sorting
                 if mask_le.sum() == 0 or mask_gt.sum() == 0:
                      sorted_indices = torch.argsort(box_to_split.data[:, dim_to_split])
                      median_index = (box_to_split.num_pixels + 1) // 2
                      mask_le = torch.zeros(box_to_split.num_pixels, dtype=torch.bool, device=device)
                      mask_le[sorted_indices[:median_index]] = True
                      mask_gt = ~mask_le

            box1_data = box_to_split.data[mask_le]
            box2_data = box_to_split.data[mask_gt]

        except Exception as sort_err:
            print(f"MedianCut: Error splitting box: {sort_err}. Skipping split.")
            boxes.append(box_to_split) # Put it back
            continue

        if box1_data.shape[0] > 0: boxes.append(Box(box1_data))
        if box2_data.shape[0] > 0: boxes.append(Box(box2_data))

    # Calculate centroids (average color) for each final box
    centroids = torch.stack([box.data.mean(dim=0) for box in boxes if box.num_pixels > 0])

    # Assign original pixels to the nearest centroid using cdist
    if centroids.shape[0] > 0:
        distances = torch.cdist(pixels.float(), centroids.float())
        labels = torch.argmin(distances, dim=1)
    else:
        print("MedianCut Warning: No centroids generated.")
        labels = torch.zeros(pixels.shape[0], dtype=torch.long, device=device)
        if pixels.shape[0] > 0:
            centroids = pixels.mean(dim=0, keepdim=True)
        else:
            centroids = torch.empty((0, pixels.shape[1]), dtype=pixels.dtype, device=device)

    return labels, centroids


# --- Функции Octree (Заглушка) ---
def octree_quantization_impl(pixels_rgb_0_1, num_colors):
    """Заглушка для Octree, вызывает K-Means."""
    print("Warning: Octree quantization not implemented, using K-Means as fallback.")
    # K-Means ожидает пиксели, кол-во цветов, макс итераций
    return kmeans_quantization(pixels_rgb_0_1, num_colors, max_iters=20)

# --- Фильтрация Маленьких Кластеров/Палитры ---
def filter_small_clusters(pixels, labels, centroids, min_area):
    """Фильтрует кластеры K-Means/MedianCut, размер которых меньше min_area."""
    device = pixels.device
    if centroids is None or centroids.shape[0] <= 1 or labels is None or min_area <= 0:
        return labels, centroids

    num_centroids = centroids.shape[0]
    unique_labels, counts = torch.unique(labels, return_counts=True)

    valid_centroids_list = []
    new_to_old_map = []
    old_to_new_map = -torch.ones(num_centroids, dtype=torch.long, device=device)

    new_idx_counter = 0
    # Find valid clusters first
    for i in range(num_centroids): # Iterate through all potential original indices
        mask = (unique_labels == i)
        if mask.any(): # Check if this label index actually exists in the output
            count = counts[mask].item()
            if count >= min_area:
                valid_centroids_list.append(centroids[i])
                new_to_old_map.append(i) # Store old index
                old_to_new_map[i] = new_idx_counter
                new_idx_counter += 1
        # Else: label index `i` was not present or filtered out

    num_valid_clusters = len(valid_centroids_list)

    # --- Handle edge cases ---
    if num_valid_clusters == num_centroids:
        return labels, centroids # No filtering needed
    if num_valid_clusters == 0:
        if counts.numel() > 0:
            print(f"Warning: All clusters are smaller than min_area ({min_area}). Keeping the largest original cluster.")
            largest_original_idx_in_counts = torch.argmax(counts)
            largest_original_label = unique_labels[largest_original_idx_in_counts].item() # Get python int index
            if largest_original_label < num_centroids:
                new_centroids = centroids[largest_original_label].unsqueeze(0)
                new_labels = torch.zeros_like(labels)
                return new_labels, new_centroids
            else:
                print("Error: Largest cluster label index out of bounds.")
                return labels, centroids
        else:
             print("Warning: No clusters found for filtering. Returning original.")
             return labels, centroids

    # --- Reassign pixels from invalid clusters ---
    new_centroids = torch.stack(valid_centroids_list)
    new_labels = torch.full_like(labels, -1) # Initialize with -1

    # Map pixels from valid clusters
    for new_idx, old_idx in enumerate(new_to_old_map):
        mask = (labels == old_idx)
        new_labels[mask] = new_idx

    # Reassign pixels from invalid clusters
    invalid_pixel_mask = (new_labels == -1)
    if invalid_pixel_mask.any():
        pixels_to_reassign = pixels[invalid_pixel_mask]
        if pixels_to_reassign.shape[0] > 0:
            distances = torch.cdist(pixels_to_reassign.float(), new_centroids.float())
            nearest_new_labels = torch.argmin(distances, dim=1)
            new_labels[invalid_pixel_mask] = nearest_new_labels

    # Sanity check
    if (new_labels == -1).any():
         print("Error: Some pixels were not reassigned during filtering! Assigning to cluster 0.")
         new_labels[new_labels == -1] = 0 # Fallback

    return new_labels, new_centroids

def filter_palette_by_usage(pixels_in_space, labels, palette_in_space, min_area):
    """Фильтрует цвета палитры на основе количества назначенных пикселей."""
    device = pixels_in_space.device
    if palette_in_space is None or palette_in_space.shape[0] <= 1 or labels is None or min_area <= 0:
        return labels, palette_in_space

    num_palette_colors = palette_in_space.shape[0]
    unique_labels, counts = torch.unique(labels, return_counts=True) # Labels correspond to palette indices

    valid_palette_list = []
    new_to_old_map = []
    old_to_new_map = -torch.ones(num_palette_colors, dtype=torch.long, device=device)

    new_idx_counter = 0
    # Find valid palette colors
    for i in range(num_palette_colors): # Iterate through all potential original indices
         mask = (unique_labels == i)
         if mask.any():
             count = counts[mask].item()
             if count >= min_area:
                  valid_palette_list.append(palette_in_space[i])
                  new_to_old_map.append(i)
                  old_to_new_map[i] = new_idx_counter
                  new_idx_counter += 1

    num_valid_colors = len(valid_palette_list)

    if num_valid_colors == num_palette_colors:
        return labels, palette_in_space # No filtering needed
    if num_valid_colors == 0:
        if counts.numel() > 0:
             print(f"Warning: All palette colors used less than min_area ({min_area}). Keeping the most used color.")
             largest_original_idx_in_counts = torch.argmax(counts)
             largest_original_label = unique_labels[largest_original_idx_in_counts].item()
             if largest_original_label < num_palette_colors:
                  new_palette = palette_in_space[largest_original_label].unsqueeze(0)
                  new_labels = torch.zeros_like(labels)
                  return new_labels, new_palette
             else:
                  print("Error: Most used palette color index out of bounds.")
                  return labels, palette_in_space
        else:
             print("Warning: No palette colors seem to be used. Returning original.")
             return labels, palette_in_space

    # Reassign pixels belonging to filtered-out palette colors
    new_palette = torch.stack(valid_palette_list)
    new_labels = torch.full_like(labels, -1)

    # Map valid pixels
    for new_idx, old_idx in enumerate(new_to_old_map):
        mask = (labels == old_idx)
        new_labels[mask] = new_idx

    # Reassign invalid pixels
    invalid_pixel_mask = (new_labels == -1)
    if invalid_pixel_mask.any():
        pixels_to_reassign = pixels_in_space[invalid_pixel_mask]
        if pixels_to_reassign.shape[0] > 0:
             distances = torch.cdist(pixels_to_reassign.float(), new_palette.float())
             nearest_new_labels = torch.argmin(distances, dim=1)
             new_labels[invalid_pixel_mask] = nearest_new_labels

    if (new_labels == -1).any():
         print("Error: Some pixels were not reassigned during palette filtering! Assigning to color 0.")
         new_labels[new_labels == -1] = 0 # Fallback

    print(f"Filtered palette colors by usage: {num_palette_colors} -> {new_palette.shape[0]}")
    return new_labels, new_palette

# --- Обертка для Квантования ---
def run_color_quantization(image_in_space, num_colors, method, min_pixel_area, max_iter, processing_space):
    """
    Обертка для вызова нужного метода квантования.
    Ожидает и возвращает данные в processing_space.
    """
    batch_size, channels, height, width = image_in_space.shape
    device = image_in_space.device
    image_flat = image_in_space.view(batch_size, channels, -1).permute(0, 2, 1) # (B, H*W, C)

    quantized_images_batch = []
    final_centroids_in_space_list = []

    for i in range(batch_size):
        pixels = image_flat[i].float() # (H*W, C)
        if pixels.shape[0] == 0:
             quantized_images_batch.append(image_in_space[i].unsqueeze(0))
             final_centroids_in_space_list.append(torch.empty((0, channels), device=device, dtype=image_in_space.dtype))
             continue

        labels, centroids_in_processing_space = None, None
        try:
            unique_colors = torch.unique(pixels, dim=0)
            effective_num_colors = min(num_colors, unique_colors.shape[0])

            if effective_num_colors < 2:
                 labels = torch.zeros(pixels.shape[0], dtype=torch.long, device=device)
                 centroids_in_processing_space = unique_colors
            elif effective_num_colors < num_colors:
                 print(f"Requested {num_colors}, using {effective_num_colors} unique colors.")

            # --- Вызов метода квантования ---
            if labels is None:
                # Wu/Octree требуют RGB, конвертируем туда и обратно
                if method == "wu" or method == "octree":
                    # Wu не реализован, используем Octree (который fallback на kmeans)
                     print(f"Quantizing with {method} (RGB internally)...")
                     pixels_rgb_0_1 = from_quantize_space(pixels.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), processing_space).squeeze().clamp(0,1)
                     if pixels_rgb_0_1.ndim == 1: pixels_rgb_0_1 = pixels_rgb_0_1.unsqueeze(0)

                     if method == "octree":
                         labels, centroids_rgb_0_1 = octree_quantization_impl(pixels_rgb_0_1, effective_num_colors)
                     else: # Wu - заглушка
                         print("Wu quantization not implemented, using Octree fallback.")
                         labels, centroids_rgb_0_1 = octree_quantization_impl(pixels_rgb_0_1, effective_num_colors)

                     # Конвертируем центроиды обратно в рабочее пространство
                     centroids_in_processing_space = to_quantize_space(centroids_rgb_0_1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), processing_space).squeeze()
                     if centroids_in_processing_space.ndim == 1: centroids_in_processing_space = centroids_in_processing_space.unsqueeze(0)

                elif method == "kmeans":
                     print(f"Quantizing with {method} in {processing_space} space...")
                     labels, centroids_in_processing_space = kmeans_quantization(pixels, effective_num_colors, max_iter)
                elif method == "median_cut":
                     print(f"Quantizing with {method} in {processing_space} space...")
                     labels, centroids_in_processing_space = median_cut_quantization(pixels, effective_num_colors)
                else:
                    raise ValueError(f"Unknown quantization method: {method}")

            # --- Фильтр маленьких кластеров ---
            if centroids_in_processing_space is not None and centroids_in_processing_space.shape[0] > 1 and min_pixel_area > 0 and labels is not None:
                 original_count = centroids_in_processing_space.shape[0]
                 labels, centroids_in_processing_space = filter_small_clusters(pixels, labels, centroids_in_processing_space, min_pixel_area)
                 if centroids_in_processing_space.shape[0] < original_count:
                      print(f"Filtered small clusters: {original_count} -> {centroids_in_processing_space.shape[0]}")

            # --- Применение квантованных цветов ---
            if centroids_in_processing_space is None or centroids_in_processing_space.shape[0] == 0:
                 quantized_image = image_in_space[i]
                 final_centroids_in_space_list.append(torch.empty((0, channels), device=device, dtype=image_in_space.dtype))
            else:
                 labels = labels.clamp(0, centroids_in_processing_space.shape[0] - 1)
                 quantized_pixels = centroids_in_processing_space[labels].to(image_in_space.dtype)
                 quantized_image = quantized_pixels.permute(1, 0).view(channels, height, width)
                 final_centroids_in_space_list.append(centroids_in_processing_space.float())

            quantized_images_batch.append(quantized_image.unsqueeze(0))

        except Exception as e:
            print(f"Error during quantization logic: {e}")
            traceback.print_exc()
            quantized_images_batch.append(image_in_space[i].unsqueeze(0)) # Fallback
            final_centroids_in_space_list.append(torch.empty((0, channels), device=device, dtype=image_in_space.dtype))

    quantized_images_final = torch.cat(quantized_images_batch, dim=0)
    # Берем центроиды первого изображения батча
    final_centroids = final_centroids_in_space_list[0] if final_centroids_in_space_list else torch.empty((0, channels), device=device, dtype=torch.float32)

    # Возвращает изображение и центроиды в processing_space
    return quantized_images_final, final_centroids

# --- Функции Применения Палитры и Дизеринга ---
def apply_fixed_palette(image, palette):
    """Применяет фиксированную палитру к изображению (поиск ближайшего)."""
    B, C, H, W = image.shape
    device = image.device
    if palette is None or palette.shape[0] == 0:
        print("Warning: Attempting to apply an empty palette. Returning original image.")
        return image
    if palette.ndim == 1: palette = palette.unsqueeze(0)
    if C != palette.shape[1]:
        raise ValueError(f"Internal Error: Image channels ({C}) != palette channels ({palette.shape[1]})")

    pixels = image.permute(0, 2, 3, 1).reshape(-1, C) # (B*H*W, C)
    distances = torch.cdist(pixels.float(), palette.float())
    labels = torch.argmin(distances, dim=1)
    quantized_pixels = palette[labels]
    quantized_image = quantized_pixels.view(B, H, W, C).permute(0, 3, 1, 2) # -> (B, C, H, W)
    return quantized_image

def apply_fixed_palette_get_labels(image, palette):
    """Применяет палитру и возвращает метки."""
    B, C, H, W = image.shape
    device = image.device
    if palette is None or palette.shape[0] == 0:
        return image, None
    if palette.ndim == 1: palette = palette.unsqueeze(0)
    if C != palette.shape[1]:
         raise ValueError(f"Internal Error: Image channels ({C}) != palette channels ({palette.shape[1]})")

    pixels = image.permute(0, 2, 3, 1).reshape(-1, C)
    distances = torch.cdist(pixels.float(), palette.float())
    labels = torch.argmin(distances, dim=1)
    quantized_pixels = palette[labels]
    quantized_image = quantized_pixels.view(B, H, W, C).permute(0, 3, 1, 2)

    labels_batch0 = labels.view(B, H*W)[0] if B > 0 else None
    return quantized_image, labels_batch0


def apply_dithering(image_rgb_source, palette_rgb, dither_pattern, dither_strength, color_distance_threshold):
    """Применяет дизеринг. Работает в RGB."""
    B, C, H, W = image_rgb_source.shape
    device = image_rgb_source.device

    if palette_rgb is None or palette_rgb.shape[0] < 1:
        print("Warning: Dithering requires a palette. Skipping.")
        return image_rgb_source # Return original
    if palette_rgb.shape[0] < 2 and dither_pattern != "No Dithering":
         print(f"Warning: Dithering pattern '{dither_pattern}' requires >= 2 palette colors. Mapping to the single color.")
         return apply_fixed_palette(image_rgb_source, palette_rgb)

    dither_image_buffer = image_rgb_source.clone().float()
    final_dithered_image = torch.zeros_like(image_rgb_source, dtype=torch.float32)

    # --- Ordered Dithering ---
    if dither_pattern in ORDERED_PATTERNS:
        bayer_matrix = ORDERED_PATTERNS[dither_pattern].to(device)
        matrix_size = bayer_matrix.shape[0]
        repeat_h = (H + matrix_size - 1) // matrix_size
        repeat_w = (W + matrix_size - 1) // matrix_size
        full_bayer = bayer_matrix.repeat(repeat_h, repeat_w)[:H, :W].unsqueeze(0).unsqueeze(0)

        num_palette_colors = palette_rgb.shape[0]
        dither_intensity_divisor = max(1, num_palette_colors - 1)
        dither_amount = (full_bayer * dither_strength) / dither_intensity_divisor

        image_plus_dither = (dither_image_buffer + dither_amount).clamp(0.0, 1.0)
        final_dithered_image = apply_fixed_palette(image_plus_dither, palette_rgb)

    # --- White Noise Dithering ---
    elif dither_pattern == "WhiteNoise":
        noise = (torch.rand_like(dither_image_buffer) - 0.5)
        num_palette_colors = palette_rgb.shape[0]
        noise_scale = dither_strength / max(1, num_palette_colors - 1)
        image_plus_noise = (dither_image_buffer + noise * noise_scale).clamp(0.0, 1.0)
        final_dithered_image = apply_fixed_palette(image_plus_noise, palette_rgb)

    # --- Error Diffusion Dithering ---
    elif dither_pattern in DIFFUSION_PATTERNS:
        pattern = DIFFUSION_PATTERNS[dither_pattern]
        threshold_sq = color_distance_threshold ** 2 if color_distance_threshold > 0 else -1.0

        output_batch = []
        for b in range(B):
             current_image_buffer = dither_image_buffer[b].clone() # C, H, W
             current_output_image = torch.zeros_like(current_image_buffer) # C, H, W

             for y in range(H):
                 x_range = range(W) if y % 2 == 0 else range(W - 1, -1, -1)
                 for x in x_range:
                     old_pixel = current_image_buffer[:, y, x].clone()
                     diff = palette_rgb - old_pixel
                     distances_sq = torch.sum(diff * diff, dim=1)
                     min_dist_sq, closest_index = torch.min(distances_sq, dim=0)
                     new_pixel = palette_rgb[closest_index]
                     current_output_image[:, y, x] = new_pixel
                     quant_error = old_pixel - new_pixel

                     should_diffuse = (threshold_sq < 0) or (min_dist_sq.item() <= threshold_sq)

                     if should_diffuse:
                         error_to_distribute = quant_error * dither_strength
                         for dx, dy, factor in pattern:
                             current_dx = dx if y % 2 == 0 else -dx
                             nx, ny = x + current_dx, y + dy
                             if 0 <= nx < W and 0 <= ny < H:
                                 current_image_buffer[:, ny, nx] += error_to_distribute * factor
                                 # Clamp buffer immediately to prevent extreme values
                                 current_image_buffer[:, ny, nx].clamp_(0.0, 1.0)

             output_batch.append(current_output_image.unsqueeze(0))

        if output_batch:
             final_dithered_image = torch.cat(output_batch, dim=0)
        else:
             final_dithered_image = image_rgb_source

    else: # Unknown pattern or No Dithering
        if dither_pattern != "No Dithering":
             print(f"Warning: Unknown dither pattern '{dither_pattern}'. Applying palette without dithering.")
        final_dithered_image = apply_fixed_palette(image_rgb_source, palette_rgb)

    return final_dithered_image.clamp(0.0, 1.0)


# --- Вспомогательные функции для палитры ---
def convert_palette_to_string(color_palette):
    """Конвертирует тензор палитры [N, 3] RGB [0,1] в HEX-строку."""
    if color_palette is None: return "N/A"
    palette_cpu = color_palette.detach().cpu().numpy()
    hex_colors = []
    for color in palette_cpu:
        color = np.clip(color, 0, 1)
        r, g, b = (color * 255).round().astype(int)
        hex_colors.append('#{:02X}{:02X}{:02X}'.format(r, g, b))
    return ', '.join(hex_colors)

# --- Функции Davies-Bouldin Index (DBI) ---
def davies_bouldin_index(pixels, labels, centroids):
    """Вычисляет индекс Davies-Bouldin."""
    n_clusters = centroids.shape[0]
    device = pixels.device
    if n_clusters < 2:
        return float('inf')

    # Intra-cluster dispersions
    intra_dispersions = torch.zeros(n_clusters, device=device, dtype=torch.float64)
    cluster_sizes = torch.bincount(labels, minlength=n_clusters).long()

    for i in range(n_clusters):
        cluster_points = pixels[labels == i]
        if cluster_sizes[i] > 0:
            distances = torch.norm(cluster_points.float() - centroids[i].float(), p=2, dim=1)
            intra_dispersions[i] = distances.mean()

    # Inter-cluster distances
    centroid_distances = torch.cdist(centroids.float(), centroids.float(), p=2) + 1e-8 # Add epsilon

    db_index = 0.0
    valid_clusters_count = 0
    for i in range(n_clusters):
        if cluster_sizes[i] == 0: continue

        max_ratio = 0.0
        found_comparison = False
        for j in range(n_clusters):
            if i != j and cluster_sizes[j] > 0:
                ratio = (intra_dispersions[i] + intra_dispersions[j]) / centroid_distances[i, j]
                if torch.isfinite(ratio):
                    if ratio > max_ratio:
                        max_ratio = ratio
                    found_comparison = True

        if found_comparison:
             db_index += max_ratio
             valid_clusters_count += 1

    if valid_clusters_count == 0:
         return float('inf')

    final_dbi = (db_index / valid_clusters_count).item()

    if not math.isfinite(final_dbi):
        print("Warning: DBI calculation resulted in non-finite value. Returning Inf.")
        return float('inf')

    return final_dbi


def determine_optimal_num_colors_dbi(pixels_rgb, method, max_k=16, sample_size=5000, min_k=2):
    """Определяет оптимальное число цветов с помощью DBI."""
    device = pixels_rgb.device
    try:
        unique_colors = torch.unique(pixels_rgb, dim=0)
        num_unique = unique_colors.shape[0]
        if num_unique <= min_k:
             print(f"DBI: Found <= {min_k} unique colors ({num_unique}). Returning this count.")
             return max(1, num_unique)
        max_k = min(max_k, num_unique)
        if min_k >= max_k: min_k = max(2, max_k) # Ensure min_k >= 2 and <= max_k
    except Exception as e:
        print(f"DBI: Warning during unique color check: {e}")
        pass

    if min_k < 2: min_k = 2
    if min_k > max_k:
        print(f"DBI: min_k ({min_k}) > max_k ({max_k}). Returning min_k.")
        return min_k

    K_range = range(min_k, max_k + 1)
    num_samples = min(sample_size, pixels_rgb.shape[0])
    if num_samples <= 1: return min_k

    pixels_sampled = pixels_rgb[torch.randperm(pixels_rgb.shape[0], device=device)[:num_samples]].float()

    best_k = min_k
    best_dbi = float('inf')
    print(f"Calculating DBI for K={min_k} to {max_k} (sample size: {num_samples})...")
    for k in K_range:
        try:
            # Используем K-Means (в RGB) для оценки DBI
            labels, centroids = kmeans_quantization(pixels_sampled, k, max_iters=15)
            if centroids.shape[0] < 2: continue

            dbi = davies_bouldin_index(pixels_sampled, labels, centroids)
            if math.isfinite(dbi) and dbi < best_dbi:
                best_dbi = dbi
                best_k = k
        except Exception as e:
            print(f"  Warning: Error calculating DBI for K={k}: {e}")
            continue

    if best_dbi == float('inf'):
         print(f"DBI calculation failed. Returning default k={min_k}")
         return min_k

    print(f"DBI calculation finished. Best K={best_k} with DBI={best_dbi:.4f}")
    return best_k
