import torch
import math
import traceback
from .color_utils import from_quantize_space, to_quantize_space, calculate_dbi # Импорт DBI

# === Вспомогательные функции Квантования ===



# --- Диапазоны для нормализации и проекции ---
# Определяем ожидаемые диапазоны для разных пространств
# Kornia LAB: L [0, 100], a [-~110, ~110], b [-~110, ~110]
# Kornia HSV: H [0, 2pi] or [0, 360], S [0, 1], V [0, 1] (Kornia uses 2pi for H)
# Kornia YCbCr: Y [0, 1], Cb [0, 1], Cr [0, 1] (приблизительно, зависит от стандарта)
# RGB: [0, 1]
COLOR_SPACE_RANGES = {
    "RGB": torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
    "HSV": torch.tensor([[0.0, 2.0 * math.pi], [0.0, 1.0], [0.0, 1.0]]),
    "LAB": torch.tensor([[0.0, 100.0], [-110.0, 110.0], [-110.0, 110.0]]), # Approximate a/b range
    "YCbCr": torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]), # Approx. after conversion
}

def apply_fixed_palette(image_in_space, palette_in_space):
    """Applies a fixed palette using nearest color search in the feature space."""
    B, C, H, W = image_in_space.shape
    if palette_in_space is None or palette_in_space.shape[0] == 0:
        print("Warning: Attempting to apply an empty palette. Returning original image.")
        return image_in_space
    if palette_in_space.ndim == 1: palette_in_space = palette_in_space.unsqueeze(0)
    # Compatibility check
    if C != palette_in_space.shape[1]:
         # If channels mismatch (e.g. applying RGB palette to LAB image)
         # This indicates an error earlier in the workflow.
         print(f"ERROR: Channel mismatch between image ({C}) and palette ({palette_in_space.shape[1]}) in apply_fixed_palette. Returning original.")
         # traceback.print_stack() # Optional: Print stack trace to find the source
         return image_in_space # Return original to avoid crashing

    pixels = image_in_space.permute(0, 2, 3, 1).reshape(-1, C) # (B*H*W, C)

    # Find the closest palette color index for each pixel in the current feature space
    # Ensure both are float32 for cdist
    distances = torch.cdist(pixels.float(), palette_in_space.float())
    labels = torch.argmin(distances, dim=1)  # Shape: (B*H*W)

    # Map pixels to palette colors using the indices
    quantized_pixels = palette_in_space[labels]
    quantized_image = quantized_pixels.view(B, H, W, C).permute(0, 3, 1, 2) # -> (B, C, H, W)
    return quantized_image # Returns image with palette colors, in the same space

def apply_fixed_palette_get_labels(image_in_space, palette_in_space):
    """Applies palette and returns labels for the first batch item."""
    B, C, H, W = image_in_space.shape
    if palette_in_space is None or palette_in_space.shape[0] == 0:
        return image_in_space, None
    if palette_in_space.ndim == 1: palette_in_space = palette_in_space.unsqueeze(0)
    if C != palette_in_space.shape[1]:
         print(f"ERROR: Channel mismatch in apply_fixed_palette_get_labels: image({C}) vs palette({palette_in_space.shape[1]})")
         return image_in_space, None

    pixels = image_in_space.permute(0, 2, 3, 1).reshape(-1, C)
    # Ensure float32 for cdist
    distances = torch.cdist(pixels.float(), palette_in_space.float())
    labels = torch.argmin(distances, dim=1)
    quantized_pixels = palette_in_space[labels]
    quantized_image = quantized_pixels.view(B, H, W, C).permute(0, 3, 1, 2)

    # Return labels for the first batch item (if exists) for filtering
    labels_batch0 = labels.view(B, H * W)[0].clone() if B > 0 else None # Clone labels
    return quantized_image, labels_batch0

def filter_small_clusters(pixels_in_space, labels, centroids_in_space, min_area):
    """Filters K-Means clusters smaller than min_area."""
    if centroids_in_space is None or centroids_in_space.shape[0] <= 1 or labels is None:
        return labels.clone() if labels is not None else None, centroids_in_space.clone() if centroids_in_space is not None else None

    num_centroids = centroids_in_space.shape[0]
    device = pixels_in_space.device
    unique_labels, counts = torch.unique(labels, return_counts=True)

    # Map counts to original centroid indices
    label_to_count = {label.item(): count.item() for label, count in zip(unique_labels, counts)}

    valid_centroids_list = []
    new_to_old_map = [] # Map: new index -> old index
    old_to_new_map = -torch.ones(num_centroids, dtype=torch.long, device=device)

    new_idx_counter = 0
    for old_idx in range(num_centroids):
        count = label_to_count.get(old_idx, 0)
        if count >= min_area:
            valid_centroids_list.append(centroids_in_space[old_idx])
            new_to_old_map.append(old_idx)
            old_to_new_map[old_idx] = new_idx_counter
            new_idx_counter += 1

    num_valid_clusters = len(valid_centroids_list)

    # --- Handle edge cases ---
    if num_valid_clusters == num_centroids:
        return labels.clone(), centroids_in_space.clone() # No filtering needed

    if num_valid_clusters == 0:
        # All clusters too small. Keep the largest original cluster.
        if counts.numel() > 0:
            print(f"Warning: All clusters are smaller than min_area ({min_area}). Keeping the most populated original cluster.")
            largest_original_idx_in_counts = torch.argmax(counts)
            largest_original_label = unique_labels[largest_original_idx_in_counts].item() # Get Python int

            if 0 <= largest_original_label < num_centroids:
                new_centroids = centroids_in_space[largest_original_label].unsqueeze(0).clone()
                # Reassign all pixels to this single cluster
                new_labels = torch.zeros_like(labels)
                return new_labels, new_centroids
            else: # Should not happen if unique_labels are from labels
                print(f"Error: Largest cluster label index ({largest_original_label}) out of bounds ({num_centroids}). Returning original.")
                return labels.clone(), centroids_in_space.clone() # Return original as fallback
        else: # No labels found at all?
             print("Warning: No clusters found for filtering. Returning original.")
             return labels.clone(), centroids_in_space.clone()

    # --- Reassign pixels from invalid clusters ---
    new_centroids = torch.stack(valid_centroids_list).clone()
    new_labels = torch.full_like(labels, -1) # Initialize with -1

    # Pixels already in valid clusters: map their old label index to the new one
    for new_idx, old_idx in enumerate(new_to_old_map):
        mask = (labels == old_idx)
        new_labels[mask] = new_idx

    # Pixels in invalid clusters (where new_labels is still -1)
    invalid_pixel_mask = (new_labels == -1)
    if invalid_pixel_mask.any():
        pixels_to_reassign = pixels_in_space[invalid_pixel_mask]
        if pixels_to_reassign.shape[0] > 0:
            # Find nearest *new* centroid for these pixels
            distances = torch.cdist(pixels_to_reassign.float(), new_centroids.float())
            nearest_new_labels = torch.argmin(distances, dim=1)
            new_labels[invalid_pixel_mask] = nearest_new_labels

    # Sanity check: ensure no -1 labels remain
    if (new_labels == -1).any():
         print("Error: Some pixels were not reassigned during filtering!")
         # Fallback: Assign remaining -1 pixels to the first valid cluster (index 0)
         new_labels[new_labels == -1] = 0

    return new_labels, new_centroids

def filter_palette_by_usage(pixels_in_space, labels, palette_in_space, min_area):
    """Filters palette colors based on pixel count, reassigning pixels."""
    # Similar logic to _filter_small_clusters, but operates on a fixed palette
    if palette_in_space is None or palette_in_space.shape[0] <= 1 or labels is None:
        return labels.clone() if labels is not None else None, palette_in_space.clone() if palette_in_space is not None else None

    num_palette_colors = palette_in_space.shape[0]
    device = pixels_in_space.device
    unique_labels, counts = torch.unique(labels, return_counts=True) # Labels correspond to palette indices

    # Map counts to original palette indices
    label_to_count = {label.item(): count.item() for label, count in zip(unique_labels, counts)}

    valid_palette_list = []
    new_to_old_map = []
    old_to_new_map = -torch.ones(num_palette_colors, dtype=torch.long, device=device)

    new_idx_counter = 0
    for old_idx in range(num_palette_colors):
        count = label_to_count.get(old_idx, 0)
        if count >= min_area:
            valid_palette_list.append(palette_in_space[old_idx])
            new_to_old_map.append(old_idx)
            old_to_new_map[old_idx] = new_idx_counter
            new_idx_counter += 1

    num_valid_colors = len(valid_palette_list)

    if num_valid_colors == num_palette_colors:
        return labels.clone(), palette_in_space.clone() # No filtering needed

    if num_valid_colors == 0:
        # All palette colors used less than min_area. Keep the most used color.
        if counts.numel() > 0:
            print(f"Warning: All palette colors used less than min_area ({min_area}). Keeping the most populated original color.")
            largest_original_idx_in_counts = torch.argmax(counts)
            largest_original_label = unique_labels[largest_original_idx_in_counts].item()

            if 0 <= largest_original_label < num_palette_colors:
                new_palette = palette_in_space[largest_original_label].unsqueeze(0).clone()
                new_labels = torch.zeros_like(labels)
                return new_labels, new_palette
            else:
                 print(f"Error: Most used palette color index ({largest_original_label}) out of bounds ({num_palette_colors}). Returning original.")
                 return labels.clone(), palette_in_space.clone()
        else:
             print("Warning: No palette colors seem to be used. Returning original.")
             return labels.clone(), palette_in_space.clone()

    # Reassign pixels belonging to filtered-out palette colors
    new_palette = torch.stack(valid_palette_list).clone()
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
         print("Error: Some pixels were not reassigned during palette filtering!")
         new_labels[new_labels == -1] = 0 # Fallback

    print(f"Filtered palette colors by usage: {num_palette_colors} -> {new_palette.shape[0]}")
    return new_labels, new_palette

# === Основные Алгоритмы Квантования ===

def kmeans_plus_plus_initialization(pixels, num_colors):
    """Initializes centroids using K-Means++ algorithm."""
    num_pixels, num_features = pixels.shape
    device = pixels.device
    centroids = torch.empty((num_colors, num_features), dtype=pixels.dtype, device=device)
    if num_pixels == 0: return centroids # Handle empty input

    # 1. Choose one center uniformly at random
    first_centroid_idx = torch.randint(0, num_pixels, (1,), device=device).item() # Get index as int
    centroids[0] = pixels[first_centroid_idx]

    # Use float64 for squared distances for stability, especially if pixels are float16/bf16
    min_sq_distances = torch.full((num_pixels,), float('inf'), dtype=torch.float64, device=device)

    for c in range(1, num_colors):
        # Calculate squared distances from points to the *most recently added* center
        last_centroid = centroids[c-1:c].double() # Use float64 for centroid
        dist_sq = torch.sum((pixels.double() - last_centroid)**2, dim=1) # Use float64 for pixels
        min_sq_distances = torch.minimum(min_sq_distances, dist_sq)

        # Check if all distances are zero or non-finite
        finite_mask = torch.isfinite(min_sq_distances)
        if not finite_mask.any(): # No finite distances, maybe all points are the same?
            print("KMeans++ Warning: No finite distances found. Picking random point.")
            next_centroid_idx = torch.randint(0, num_pixels, (1,), device=device).item()
        else:
            # Use only finite distances for probability calculation
            finite_distances = min_sq_distances[finite_mask]
            min_sq_distances_sum = torch.sum(finite_distances)

            if min_sq_distances_sum <= 1e-12: # Use a small tolerance
                # If sum is zero, all finite distances are zero. Pick randomly among finite points.
                #print("KMeans++ Warning: Sum of finite distances is zero. Picking random finite point.")
                finite_indices = torch.where(finite_mask)[0]
                next_centroid_idx = finite_indices[torch.randint(0, len(finite_indices), (1,), device=device)].item()
            else:
                probabilities = finite_distances / min_sq_distances_sum
                # Ensure probabilities are non-negative and sum to 1 (approx)
                probabilities = torch.clamp(probabilities, min=0.0)
                probabilities /= probabilities.sum() # Renormalize just in case

                try:
                    # Sample from the indices corresponding to finite distances
                    finite_indices = torch.where(finite_mask)[0]
                    sampled_finite_idx = torch.multinomial(probabilities, 1).item()
                    next_centroid_idx = finite_indices[sampled_finite_idx].item() # Map back to original index
                except RuntimeError as multi_err:
                    print(f"Multinomial Error ({multi_err}). Sum={min_sq_distances_sum}. Probabilities (finite):", probabilities)
                    # Fallback: choose the finite point with max probability (max distance)
                    finite_indices = torch.where(finite_mask)[0]
                    next_centroid_idx = finite_indices[torch.argmax(probabilities)].item()

        centroids[c] = pixels[next_centroid_idx]
        # Optimization: Set distance of chosen point to 0 for next iteration? Might not be needed.
        # min_sq_distances[next_centroid_idx] = 0.0

    return centroids.to(pixels.dtype) # Return in original dtype

def kmeans_quantization(pixels_in_space, num_colors, max_iters):
    """Performs K-Means quantization in the provided color space."""
    # Works with `pixels_in_space` in the current `processing_space`
    # Uses Euclidean distance `torch.cdist`
    device = pixels_in_space.device
    dtype = pixels_in_space.dtype
    num_pixels, channels = pixels_in_space.shape

    if num_pixels == 0:
        return torch.empty(0, dtype=torch.long, device=device), \
               torch.empty((0, channels), dtype=dtype, device=device)

    # Handle cases where num_colors >= num_pixels or only 1 color needed
    unique_colors_in_space, inverse_indices = torch.unique(pixels_in_space, dim=0, return_inverse=True)
    num_unique = unique_colors_in_space.shape[0]

    if num_colors <= 1:
        centroid = pixels_in_space.mean(dim=0, keepdim=True)
        labels = torch.zeros(num_pixels, dtype=torch.long, device=device)
        return labels, centroid.to(dtype)
    elif num_unique <= num_colors:
        # Return unique colors as centroids and inverse_indices as labels
        return inverse_indices, unique_colors_in_space.to(dtype)

    # K-Means++ Initialization
    try:
        # Ensure pixels are float32 for robust initialization and cdist
        centroids = kmeans_plus_plus_initialization(pixels_in_space.float(), num_colors)
    except Exception as init_err:
        print(f"KMeans++ Init Error: {init_err}. Falling back to random init.")
        # Fallback: Random sample
        indices = torch.randperm(num_pixels, device=device)[:num_colors]
        centroids = pixels_in_space[indices].float() # Keep float

    pixels_float = pixels_in_space.float() # Work with float32

    for i in range(max_iters):
        # Assign labels
        distances = torch.cdist(pixels_float, centroids)
        labels = torch.argmin(distances, dim=1)

        # Update centroids using scatter_add_ for efficiency
        # Initialize new_centroids with zeros of the correct type (float32)
        new_centroids = torch.zeros_like(centroids, dtype=torch.float32)
        counts = torch.zeros(num_colors, dtype=torch.float32, device=device) # Use float for counts with scatter_add_

        # Expand labels for scatter_add_
        # Ensure labels are long type for indexing
        labels_expanded = labels.long().unsqueeze(1).expand_as(pixels_float)
        new_centroids.scatter_add_(0, labels_expanded, pixels_float)

        # Count points in each cluster using scatter_add_
        ones = torch.ones_like(pixels_float[:, 0]) # Shape (num_pixels,)
        counts.scatter_add_(0, labels.long(), ones) # Use long labels

        # Avoid division by zero for empty clusters
        counts_safe = counts.clamp(min=1.0) # Use 1.0 to avoid division by zero if counts are exactly zero
        new_centroids /= counts_safe.unsqueeze(1)

        # Handle empty clusters
        empty_cluster_mask = (counts < 0.5) # Check if count is effectively zero
        num_empty = empty_cluster_mask.sum().item()

        if num_empty > 0:
            # Re-initialize empty centroids - find points furthest from *existing* centroids
            non_empty_centroids = new_centroids[~empty_cluster_mask]
            if non_empty_centroids.shape[0] > 0 and num_pixels > num_empty: # Check if we have points and non-empty centroids
                all_dists = torch.cdist(pixels_float, non_empty_centroids)
                min_dists_to_non_empty, _ = torch.min(all_dists, dim=1)
                # Find points with the largest minimum distance
                try:
                    # Ensure k is not larger than the number of points
                    k_topk = min(num_empty, len(min_dists_to_non_empty))
                    if k_topk > 0:
                        furthest_points_indices = torch.topk(min_dists_to_non_empty, k=k_topk).indices
                        # Assign only up to num_empty points, repeat if necessary
                        assign_indices = furthest_points_indices.repeat( (num_empty + k_topk - 1) // k_topk )[:num_empty]
                        new_centroids[empty_cluster_mask] = pixels_float[assign_indices]
                    else: # No points to select from? Revert to random
                         print("KMeans Warning: Could not find furthest points for empty clusters. Re-initializing randomly.")
                         rand_indices = torch.randperm(num_pixels, device=device)[:num_empty]
                         new_centroids[empty_cluster_mask] = pixels_float[rand_indices]

                except Exception as e:
                     print(f"KMeans Warning: Error finding furthest points ({e}). Re-initializing empty clusters randomly.")
                     rand_indices = torch.randperm(num_pixels, device=device)[:num_empty]
                     new_centroids[empty_cluster_mask] = pixels_float[rand_indices]
            elif num_pixels > 0: # All clusters were empty or no points? Re-initialize all randomly
                 print("KMeans Warning: All clusters seem empty or error occurred. Re-initializing all centroids randomly.")
                 rand_indices = torch.randperm(num_pixels, device=device)[:num_colors]
                 new_centroids = pixels_float[rand_indices]
            # else: no pixels, do nothing

        # Check for convergence
        if torch.allclose(centroids, new_centroids, atol=1e-5, rtol=1e-4):
            #print(f"KMeans converged in {i+1} iterations.")
            centroids = new_centroids
            break

        centroids = new_centroids
    # else: # Loop finished without break
        #print(f"KMeans reached max iterations ({max_iters}).")

    # Final label assignment with the final centroids
    distances = torch.cdist(pixels_float, centroids)
    labels = torch.argmin(distances, dim=1)

    # Return centroids in the original input dtype
    return labels, centroids.to(dtype)

def median_cut_quantization(pixels_in_space, num_colors):
    """Performs Median Cut quantization in the provided color space."""
    device = pixels_in_space.device
    dtype = pixels_in_space.dtype
    num_pixels, channels = pixels_in_space.shape

    if num_pixels == 0:
        return torch.empty(0, dtype=torch.long, device=device), \
               torch.empty((0, channels), dtype=dtype, device=device)

    if num_colors <= 1:
        centroid = pixels_in_space.mean(dim=0, keepdim=True)
        labels = torch.zeros(num_pixels, dtype=torch.long, device=device)
        return labels, centroid.to(dtype)

    unique_colors_in_space, inverse_indices = torch.unique(pixels_in_space, dim=0, return_inverse=True)
    num_unique = unique_colors_in_space.shape[0]

    if num_unique <= num_colors:
         return inverse_indices, unique_colors_in_space.to(dtype)


    class Box:
        def __init__(self, data):
            self.data = data # (num_pixels_in_box, C)
            if data.shape[0] > 0:
                # Use float32 for calculations
                data_f = data.float()
                self.min_vals = torch.min(data_f, dim=0)[0]
                self.max_vals = torch.max(data_f, dim=0)[0]
                self.ranges = self.max_vals - self.min_vals
                self.ranges = torch.clamp(self.ranges, min=0.0) # Ensure non-negative range

                # Find dimension with max range, fallback to dim 0
                max_range_val = torch.max(self.ranges)
                if max_range_val > 1e-7: # Use tolerance
                    self.longest_dim = torch.argmax(self.ranges).item()
                else:
                    self.longest_dim = 0 # Fallback
                self.num_pixels = data.shape[0]
                # Store metric for sorting: Volume * Range in longest dim? Or just Range? Let's use Range * Num Pixels
                self.metric = self.ranges[self.longest_dim] * self.num_pixels
            else:
                # Initialize safely for empty box
                self.min_vals = torch.zeros(data.shape[1], device=data.device, dtype=torch.float32)
                self.max_vals = torch.zeros(data.shape[1], device=data.device, dtype=torch.float32)
                self.ranges = torch.zeros(data.shape[1], device=data.device, dtype=torch.float32)
                self.longest_dim = 0
                self.num_pixels = 0
                self.metric = -1.0 # Empty boxes should be sorted last

        def __lt__(self, other):
            # For sorting: higher metric comes first (we pop from end or sort reverse)
            return self.metric < other.metric

    initial_box = Box(pixels_in_space)
    boxes = [initial_box]

    while len(boxes) < num_colors:
        boxes.sort() # Sorts ascending by metric (__lt__)
        if not boxes: break # Should not happen if initial box was valid
        box_to_split = boxes.pop() # Get box with highest metric

        if box_to_split.num_pixels <= 1 or box_to_split.metric <= 1e-9: # Cannot split further
            boxes.append(box_to_split) # Put it back if it's the only one left?
            #print(f"MedianCut: Cannot split box further (pixels={box_to_split.num_pixels}, metric={box_to_split.metric:.2e}). Stopping at {len(boxes)} boxes.")
            break # Stop splitting

        dim_to_split = box_to_split.longest_dim
        # Sort pixels *within the box* along the longest dimension
        try:
            # Use float32 for sorting
            sorted_indices = torch.argsort(box_to_split.data[:, dim_to_split].float())
            sorted_pixels = box_to_split.data[sorted_indices]
        except Exception as sort_err:
            print(f"MedianCut: Error sorting pixels in box: {sort_err}. Skipping split.")
            boxes.append(box_to_split) # Put it back
            continue

        # Find median split point
        median_index = (sorted_pixels.shape[0] + 1) // 2

        # --- Refined split logic to handle identical values ---
        if median_index > 0 and median_index < sorted_pixels.shape[0]:
            val_at_median = sorted_pixels[median_index, dim_to_split]
            val_before_median = sorted_pixels[median_index - 1, dim_to_split]

            # If the values at the split point are identical, adjust the split index
            if torch.isclose(val_at_median, val_before_median, atol=1e-6): # Use tolerance for float
                # Find first index *after* median where value differs
                first_diff_idx = median_index
                while first_diff_idx < sorted_pixels.shape[0] and torch.isclose(sorted_pixels[first_diff_idx, dim_to_split], val_at_median, atol=1e-6):
                    first_diff_idx += 1

                if first_diff_idx < sorted_pixels.shape[0]:
                    median_index = first_diff_idx # Split after the block of identical values
                else:
                    # All remaining values are the same. Try splitting *before* the block.
                    last_diff_idx = median_index - 1
                    while last_diff_idx >= 0 and torch.isclose(sorted_pixels[last_diff_idx, dim_to_split], val_before_median, atol=1e-6):
                        last_diff_idx -= 1
                    # If we found a different value earlier, split after it
                    if last_diff_idx >= 0:
                         median_index = last_diff_idx + 1
                    # Else: Cannot find a good split point, leave median_index as is. Box might not split well.
        # --- End of refined split logic ---


        # Perform the split
        box1_data = sorted_pixels[:median_index]
        box2_data = sorted_pixels[median_index:]

        # Add new boxes only if they contain pixels
        if box1_data.shape[0] > 0: boxes.append(Box(box1_data))
        if box2_data.shape[0] > 0: boxes.append(Box(box2_data))

        # Safety check in case split somehow failed
        if box1_data.shape[0] == 0 and box2_data.shape[0] == 0 and box_to_split.num_pixels > 0:
             boxes.append(box_to_split) # Put original back


    # Calculate centroids (average color) for each final box
    final_centroids_list = []
    for box in boxes:
        if box.num_pixels > 0:
            # Calculate mean using float32 for stability
            centroid = box.data.float().mean(dim=0)
            final_centroids_list.append(centroid)

    if not final_centroids_list: # No valid boxes/centroids
        print("MedianCut Warning: No centroids generated. Assigning all pixels to label 0.")
        labels = torch.zeros(num_pixels, dtype=torch.long, device=device)
        if num_pixels > 0:
            centroids = pixels_in_space.float().mean(dim=0, keepdim=True).to(dtype)
        else:
            centroids = torch.empty((0, channels), dtype=dtype, device=device)
        return labels, centroids

    centroids = torch.stack(final_centroids_list).to(dtype) # Convert back to original dtype

    # Assign original pixels to the nearest centroid using cdist
    distances = torch.cdist(pixels_in_space.float(), centroids.float())
    labels = torch.argmin(distances, dim=1)

    return labels, centroids

def wu_quantization(pixels_rgb_0_1: torch.Tensor, num_colors: int):
    """Performs Wu quantization (expects and returns RGB [0,1])."""
    device = pixels_rgb_0_1.device
    dtype = pixels_rgb_0_1.dtype
    num_pixels, channels = pixels_rgb_0_1.shape

    if channels != 3:
        raise ValueError("Wu quantization requires RGB input (3 channels).")
    if num_pixels == 0:
        return torch.empty(0, dtype=torch.long, device=device), \
               torch.empty((0, 3), dtype=dtype, device=device)

    unique_colors_rgb, inverse_indices = torch.unique(pixels_rgb_0_1, dim=0, return_inverse=True)
    num_unique = unique_colors_rgb.shape[0]

    if num_colors <= 1:
        centroid = pixels_rgb_0_1.mean(dim=0, keepdim=True)
        labels = torch.zeros(num_pixels, dtype=torch.long, device=device)
        return labels, centroid.to(dtype)
    elif num_unique <= num_colors:
        return inverse_indices, unique_colors_rgb.to(dtype)


    SIZE = 33 # Histogram size (index 0 unused, 1-32 used)
    MAX_COLOR = 255.0 # Use float for scaling
    # Convert pixels to integer indices [1, 32]
    # Ensure input is float before scaling
    pixels_int = (pixels_rgb_0_1.float() * MAX_COLOR).round().long().clamp(0, 255)
    r_idx = (pixels_int[:, 0] >> 3) + 1
    g_idx = (pixels_int[:, 1] >> 3) + 1
    b_idx = (pixels_int[:, 2] >> 3) + 1
    # Combine indices for 3D histogram (use tuple for indexing)
    indices = (r_idx, g_idx, b_idx)

    # --- Calculate moments using scatter_add for unique indices ---
    unique_idx_flat, inverse_map, counts = torch.unique(
        r_idx * SIZE * SIZE + g_idx * SIZE + b_idx, return_inverse=True, return_counts=True
    )

    # Convert flat unique indices back to 3D
    unique_b = unique_idx_flat % SIZE
    unique_g = (unique_idx_flat // SIZE) % SIZE
    unique_r = unique_idx_flat // (SIZE * SIZE)
    unique_indices_3d = (unique_r, unique_g, unique_b)

    # Allocate moment tensors (use float64 for sums to avoid overflow)
    wt = torch.zeros((SIZE, SIZE, SIZE), dtype=torch.float64, device=device)
    mr = torch.zeros_like(wt)
    mg = torch.zeros_like(wt)
    mb = torch.zeros_like(wt)
    m2 = torch.zeros_like(wt)

    # Put counts into wt
    wt.index_put_(unique_indices_3d, counts.double())

    # Calculate sums for each unique bin using scatter_add_ on the inverse map
    pixels_long = pixels_int.long() # For sums
    pixels_double = pixels_int.double() # For squared sums

    sum_r = torch.zeros_like(unique_idx_flat, dtype=torch.float64).scatter_add_(0, inverse_map, pixels_double[:, 0])
    sum_g = torch.zeros_like(unique_idx_flat, dtype=torch.float64).scatter_add_(0, inverse_map, pixels_double[:, 1])
    sum_b = torch.zeros_like(unique_idx_flat, dtype=torch.float64).scatter_add_(0, inverse_map, pixels_double[:, 2])
    sum_sq = torch.zeros_like(unique_idx_flat, dtype=torch.float64).scatter_add_(0, inverse_map, (pixels_double**2).sum(dim=1))

    # Put sums into moment tensors
    mr.index_put_(unique_indices_3d, sum_r)
    mg.index_put_(unique_indices_3d, sum_g)
    mb.index_put_(unique_indices_3d, sum_b)
    m2.index_put_(unique_indices_3d, sum_sq)

    # --- Calculate 3D prefix sums (moment volumes) ---
    def prefix_sum_3d(tensor):
        out = tensor
        for d in range(3):
            out = torch.cumsum(out, dim=d)
        return out

    Vwt = prefix_sum_3d(wt)
    Vmr = prefix_sum_3d(mr)
    Vmg = prefix_sum_3d(mg)
    Vmb = prefix_sum_3d(mb)
    Vm2 = prefix_sum_3d(m2)

    # --- WuBox Class (needs access to Vwt, Vmr, etc.) ---
    class WuBox:
        # Ensure Vwt etc. are accessible (passed in or global within function scope)
        _Vwt, _Vmr, _Vmg, _Vmb, _Vm2 = Vwt, Vmr, Vmg, Vmb, Vm2

        def __init__(self, r0=0, r1=0, g0=0, g1=0, b0=0, b1=0):
            self.r0, self.r1 = r0, r1 # Inclusive indices [1, SIZE-1]
            self.g0, self.g1 = g0, g1
            self.b0, self.b1 = b0, b1
            self.calculate_stats() # Pass cumulative sums implicitly

        def vol_sum(self, M):
            # Helper to get volume sum from cumulative tensor M
            r0, r1 = self.r0, self.r1
            g0, g1 = self.g0, self.g1
            b0, b1 = self.b0, self.b1
            # Use V* tensors defined in the outer scope
            s = M[r1, g1, b1].clone()
            s -= M[r0-1, g1, b1] if r0 > 0 else 0
            s -= M[r1, g0-1, b1] if g0 > 0 else 0
            s -= M[r1, g1, b0-1] if b0 > 0 else 0
            s += M[r0-1, g0-1, b1] if r0 > 0 and g0 > 0 else 0
            s += M[r0-1, g1, b0-1] if r0 > 0 and b0 > 0 else 0
            s += M[r1, g0-1, b0-1] if g0 > 0 and b0 > 0 else 0
            s -= M[r0-1, g0-1, b0-1] if r0 > 0 and g0 > 0 and b0 > 0 else 0
            return s

        def calculate_stats(self):
            self.weight = self.vol_sum(WuBox._Vwt)
            if self.weight > 1e-9:
                self.r_sum = self.vol_sum(WuBox._Vmr)
                self.g_sum = self.vol_sum(WuBox._Vmg)
                self.b_sum = self.vol_sum(WuBox._Vmb)
                self.sq_sum = self.vol_sum(WuBox._Vm2)
                variance = self.sq_sum - (self.r_sum**2 + self.g_sum**2 + self.b_sum**2) / self.weight
                self.variance = max(0.0, variance.item())
                self.avg_r = self.r_sum / self.weight
                self.avg_g = self.g_sum / self.weight
                self.avg_b = self.b_sum / self.weight
            else:
                self.weight = 0.0
                self.r_sum = self.g_sum = self.b_sum = self.sq_sum = 0.0
                self.variance = 0.0
                self.avg_r = (self.r0 + self.r1) / 2.0 # Center index
                self.avg_g = (self.g0 + self.g1) / 2.0
                self.avg_b = (self.b0 + self.b1) / 2.0

        def __lt__(self, other):
            return self.variance < other.variance # Sort by variance ascending

    # --- Iterative Box Cutting ---
    initial_box = WuBox(1, SIZE - 1, 1, SIZE - 1, 1, SIZE - 1)
    boxes = [initial_box]

    if initial_box.weight <= 1e-9:
        print("Wu Warning: Initial box is empty.")
        return torch.empty(0, dtype=torch.long, device=device), torch.empty((0, 3), dtype=dtype, device=device)

    num_final_boxes = 1
    while num_final_boxes < num_colors:
        boxes.sort() # Highest variance last
        if not boxes: break
        box_to_split = boxes.pop()
        num_final_boxes -= 1

        if box_to_split.weight <= 1e-9 or box_to_split.variance < 1e-9 : # Don't split empty or zero-variance boxes
            boxes.append(box_to_split) # Put it back
            num_final_boxes += 1
            break # No more meaningful splits possible

        best_dir = -1
        best_pos = -1
        max_variance_decrease = -float('inf')

        # Iterate through dimensions R, G, B to find the best split
        for direction in range(3):
            if direction == 0: d0, d1 = box_to_split.r0, box_to_split.r1
            elif direction == 1: d0, d1 = box_to_split.g0, box_to_split.g1
            else: d0, d1 = box_to_split.b0, box_to_split.b1

            if d0 >= d1: continue # Cannot split if dimension has size 0 or 1

            # Iterate through possible cut positions *within* the box
            for cut_pos in range(d0 + 1, d1 + 1):
                if direction == 0:
                    box1 = WuBox(box_to_split.r0, cut_pos - 1, box_to_split.g0, box_to_split.g1, box_to_split.b0, box_to_split.b1)
                    box2 = WuBox(cut_pos, box_to_split.r1, box_to_split.g0, box_to_split.g1, box_to_split.b0, box_to_split.b1)
                elif direction == 1:
                    box1 = WuBox(box_to_split.r0, box_to_split.r1, box_to_split.g0, cut_pos - 1, box_to_split.b0, box_to_split.b1)
                    box2 = WuBox(box_to_split.r0, box_to_split.r1, cut_pos, box_to_split.g1, box_to_split.b0, box_to_split.b1)
                else:
                    box1 = WuBox(box_to_split.r0, box_to_split.r1, box_to_split.g0, box_to_split.g1, box_to_split.b0, cut_pos - 1)
                    box2 = WuBox(box_to_split.r0, box_to_split.r1, box_to_split.g0, box_to_split.g1, cut_pos, box_to_split.b1)

                if box1.weight > 1e-9 and box2.weight > 1e-9:
                    variance_decrease = box_to_split.variance - (box1.variance + box2.variance)
                    if variance_decrease > max_variance_decrease:
                        max_variance_decrease = variance_decrease
                        best_dir = direction
                        best_pos = cut_pos

        # Perform the best split if found
        if best_dir != -1:
            if best_dir == 0:
                box1 = WuBox(box_to_split.r0, best_pos - 1, box_to_split.g0, box_to_split.g1, box_to_split.b0, box_to_split.b1)
                box2 = WuBox(best_pos, box_to_split.r1, box_to_split.g0, box_to_split.g1, box_to_split.b0, box_to_split.b1)
            elif best_dir == 1:
                box1 = WuBox(box_to_split.r0, box_to_split.r1, box_to_split.g0, best_pos - 1, box_to_split.b0, box_to_split.b1)
                box2 = WuBox(box_to_split.r0, box_to_split.r1, best_pos, box_to_split.g1, box_to_split.b0, box_to_split.b1)
            else:
                box1 = WuBox(box_to_split.r0, box_to_split.r1, box_to_split.g0, box_to_split.g1, box_to_split.b0, best_pos - 1)
                box2 = WuBox(box_to_split.r0, box_to_split.r1, box_to_split.g0, box_to_split.g1, best_pos, box_to_split.b1)

            if box1.weight > 1e-9: boxes.append(box1); num_final_boxes += 1
            if box2.weight > 1e-9: boxes.append(box2); num_final_boxes += 1
            # Handle case where split results in only one valid box
            if box1.weight <= 1e-9 and box2.weight > 1e-9: boxes.append(box2); num_final_boxes += 1
            if box2.weight <= 1e-9 and box1.weight > 1e-9: boxes.append(box1); num_final_boxes += 1

        else:
            # If no split possible, put the box back and stop
            boxes.append(box_to_split)
            num_final_boxes += 1
            print(f"Warning: Wu quantization could not split box further. Stopping at {num_final_boxes} colors.")
            break

    # --- Generate palette and assign labels ---
    centroids_int_avg = torch.tensor([[box.avg_r, box.avg_g, box.avg_b] for box in boxes if box.weight > 1e-9],
                                     dtype=torch.float64, device=device)

    if centroids_int_avg.shape[0] == 0:
        print("Wu Warning: No valid centroids found after splitting.")
        return torch.empty(0, dtype=torch.long, device=device), torch.empty((0, 3), dtype=dtype, device=device)

    # Convert centroids back to [0, 1] float range
    centroids_rgb_0_1 = (centroids_int_avg / MAX_COLOR).clamp(0.0, 1.0).to(dtype) # Convert back to original dtype

    # Assign original pixels (float RGB 0-1) to the nearest final centroid (float RGB 0-1)
    distances = torch.cdist(pixels_rgb_0_1.float(), centroids_rgb_0_1.float())
    labels = torch.argmin(distances, dim=1)

    return labels, centroids_rgb_0_1

def octree_quantization_impl(pixels_rgb_0_1: torch.Tensor, num_colors: int):
    """Placeholder for Octree quantization."""
    print("Warning: Octree quantization not implemented, using K-Means fallback.")
    # Needs implementation of Octree data structure and reduction logic
    # For now, fall back to K-Means which also expects RGB [0,1]
    return kmeans_quantization(pixels_rgb_0_1, num_colors, max_iters=20)


# === НОВЫЙ АЛГОРИТМ: Stochastic Quantization (SQ) ===
def normalize_data(data_in_space, space, device):
    """Нормализует данные к диапазону [0, 1] для каждого канала."""
    if space not in COLOR_SPACE_RANGES:
        return data_in_space # Не нормализуем неизвестные пространства

    ranges = COLOR_SPACE_RANGES[space].to(device)
    min_vals = ranges[:, 0].view(1, -1) # Shape (1, C)
    max_vals = ranges[:, 1].view(1, -1) # Shape (1, C)
    range_size = max_vals - min_vals + 1e-8 # Добавляем epsilon

    # data_in_space может быть (N, C) или (B, N, C) или (C, H, W) или (B, C, H, W)
    original_shape = data_in_space.shape
    channels = original_shape[-1] if data_in_space.ndim == 2 else original_shape[-3]

    if channels != ranges.shape[0]:
        print(f"Warning: Channel mismatch during normalization. Data: {channels}, Expected for {space}: {ranges.shape[0]}. Skipping normalization.")
        return data_in_space

    if data_in_space.ndim == 2: # (N, C)
        normalized_data = (data_in_space - min_vals) / range_size
    elif data_in_space.ndim == 4: # (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = original_shape
        data_flat = data_in_space.permute(0, 2, 3, 1).reshape(B, H * W, C)
        normalized_data_flat = (data_flat - min_vals.unsqueeze(0)) / range_size.unsqueeze(0)
        normalized_data = normalized_data_flat.reshape(B, H, W, C).permute(0, 3, 1, 2) # Back to (B, C, H, W)
    elif data_in_space.ndim == 3: # (C, H, W) -> (H*W, C) ? Нестандартно, но попробуем
        C, H, W = original_shape
        data_flat = data_in_space.permute(1, 2, 0).reshape(H * W, C)
        normalized_data_flat = (data_flat - min_vals) / range_size
        normalized_data = normalized_data_flat.reshape(H, W, C).permute(2, 0, 1) # Back to (C, H, W)
    else: # Неизвестная форма
        print("Warning: Unsupported data shape for normalization. Skipping.")
        return data_in_space

    return normalized_data

def denormalize_data(data_normalized, space, device):
    """Денормализует данные из [0, 1] обратно в оригинальный диапазон."""
    if space not in COLOR_SPACE_RANGES:
        return data_normalized # Не денормализуем неизвестные пространства

    ranges = COLOR_SPACE_RANGES[space].to(device)
    min_vals = ranges[:, 0].view(1, -1) # Shape (1, C)
    max_vals = ranges[:, 1].view(1, -1) # Shape (1, C)
    range_size = max_vals - min_vals + 1e-8

    # data_normalized может быть (N, C)
    original_shape = data_normalized.shape
    channels = original_shape[-1]

    if channels != ranges.shape[0]:
        print(f"Warning: Channel mismatch during denormalization. Data: {channels}, Expected for {space}: {ranges.shape[0]}. Skipping denormalization.")
        return data_normalized

    if data_normalized.ndim == 2: # (N, C)
        denormalized_data = data_normalized * range_size + min_vals
    else: # Неизвестная форма
        print("Warning: Unsupported data shape for denormalization. Skipping.")
        return data_normalized

    return denormalized_data

def project_to_valid_range(data_in_space, space, device):
    """Ограничивает (clamp) данные допустимым диапазоном для пространства."""
    if space not in COLOR_SPACE_RANGES:
        return data_in_space # Не проецируем неизвестные пространства

    ranges = COLOR_SPACE_RANGES[space].to(device)
    min_vals = ranges[:, 0] # Shape (C,)
    max_vals = ranges[:, 1] # Shape (C,)

    # data_in_space может быть (N, C) или (B, C, H, W) или др.
    original_shape = data_in_space.shape
    if data_in_space.ndim == 2: # (N, C)
        projected_data = torch.max(torch.min(data_in_space, max_vals.unsqueeze(0)), min_vals.unsqueeze(0))
    elif data_in_space.ndim == 4: # (B, C, H, W)
        projected_data = torch.max(torch.min(data_in_space, max_vals.view(1, -1, 1, 1)), min_vals.view(1, -1, 1, 1))
    else:
        print(f"Warning: Unsupported shape {original_shape} for projection. Returning original.")
        projected_data = data_in_space

    return projected_data

# === Обновленный Stochastic Quantization (SQ) ===
def sq_quantization(pixels_in_space, num_colors, processing_space, # Добавили processing_space
                    iterations_factor=5, learning_rate_initial=0.1, learning_rate_decay_time=10000):
    """
    Performs Stochastic Quantization with normalization and adaptive learning rate.
    Operates on NORMALIZED data if space is not RGB, denormalizes centroids at the end.
    """
    device = pixels_in_space.device
    dtype = pixels_in_space.dtype
    num_pixels, channels = pixels_in_space.shape

    if num_pixels == 0:
        return torch.empty(0, dtype=torch.long, device=device), \
               torch.empty((0, channels), dtype=dtype, device=device)

    unique_colors_in_space, inverse_indices = torch.unique(pixels_in_space, dim=0, return_inverse=True)
    num_unique = unique_colors_in_space.shape[0]

    if num_colors <= 1:
        centroid = pixels_in_space.mean(dim=0, keepdim=True)
        labels = torch.zeros(num_pixels, dtype=torch.long, device=device)
        return labels, centroid.to(dtype)
    elif num_unique <= num_colors:
        return inverse_indices, unique_colors_in_space.to(dtype)

    # --- Нормализация данных (если не RGB) ---
    needs_normalization = processing_space != "RGB"
    if needs_normalization:
        print(f"SQ: Normalizing data from {processing_space} range to [0,1] for processing.")
        pixels_proc = normalize_data(pixels_in_space.float(), processing_space, device)
    else:
        pixels_proc = pixels_in_space.float() # Работаем в RGB [0,1]

    print(f"Starting Stochastic Quantization (K={num_colors}, iters={iterations_factor * num_pixels}, initial_lr={learning_rate_initial}, decay_t0={learning_rate_decay_time})...")

    # Инициализация (KMeans++ на обработанных/нормализованных данных)
    try:
        centroids_proc = kmeans_plus_plus_initialization(pixels_proc, num_colors)
    except Exception as init_err:
        print(f"SQ Init Error (KMeans++): {init_err}. Falling back to random init.")
        indices = torch.randperm(num_pixels, device=device)[:num_colors]
        centroids_proc = pixels_proc[indices]

    total_iterations = iterations_factor * num_pixels

    # Стохастические итерации
    for t in range(total_iterations):
        # Адаптивный learning rate: rho_t = initial_rho / (1 + t / t0)
        # Чтобы избежать слишком быстрого падения, можно использовать sqrt(1 + t/t0) или другую формулу
        # learning_rate = learning_rate_initial / (1.0 + t / learning_rate_decay_time)
        # Используем формулу, близкую к Adam/RMSProp decay:
        learning_rate = learning_rate_initial / math.sqrt(1.0 + t / learning_rate_decay_time)


        pixel_idx = torch.randint(0, num_pixels, (1,), device=device).item()
        current_pixel = pixels_proc[pixel_idx]

        distances = torch.norm(centroids_proc - current_pixel, p=2, dim=1)
        closest_centroid_idx = torch.argmin(distances).item()
        closest_centroid = centroids_proc[closest_centroid_idx]
        dist_norm = distances[closest_centroid_idx]

        if dist_norm > 1e-7:
            # r=3 -> ||xi - yk|| * (yk - xi)
            gradient = dist_norm * (closest_centroid - current_pixel)
            centroids_proc[closest_centroid_idx] -= learning_rate * gradient
        # else: gradient is zero, no update needed

        # Проекция на валидный диапазон [0, 1], так как работаем с нормализованными данными
        centroids_proc.clamp_(0.0, 1.0)

    print("Stochastic Quantization finished.")

    # --- Денормализация центроидов (если нужно) ---
    if needs_normalization:
        print(f"SQ: Denormalizing centroids back to {processing_space} range.")
        final_centroids = denormalize_data(centroids_proc, processing_space, device)
        # После денормализации, сделаем проекцию на реальный диапазон пространства
        final_centroids = project_to_valid_range(final_centroids, processing_space, device)
    else:
        final_centroids = centroids_proc # Уже в RGB [0,1]

    # Финальное назначение лейблов с использованием *оригинальных* пикселей и *финальных* (денормализованных) центроидов
    distances = torch.cdist(pixels_in_space.float(), final_centroids.float())
    labels = torch.argmin(distances, dim=1)

    return labels, final_centroids.to(dtype) # Возвращаем центроиды в исходном типе и пространстве


# === Обновленная основная функция квантования ===
def run_color_quantization(
    image_in_space,
    **quant_params # Принимаем все параметры через kwargs
    ):
    """
    Runs the selected color quantization method.
    Uses parameters passed via quant_params dictionary.
    """
    # Извлекаем параметры из словаря
    num_colors = quant_params.get("num_colors", 16) # Значение по умолчанию
    method = quant_params.get("method", "kmeans")
    min_pixel_area = quant_params.get("min_pixel_area", 1)
    processing_space = quant_params.get("processing_space", "RGB")
    auto_num_colors = quant_params.get("auto_num_colors", False)
    auto_k_range = quant_params.get("auto_k_range", 16)
    sample_size_dbi = quant_params.get("sample_size_dbi", 5000)
    # Параметры конкретных методов
    kmeans_max_iter = quant_params.get("kmeans_max_iter", 20)
    sq_iterations_factor = quant_params.get("sq_iterations_factor", 5)
    sq_learning_rate_initial = quant_params.get("sq_learning_rate_initial", 0.1)
    sq_learning_rate_decay_time = quant_params.get("sq_learning_rate_decay_time", 10000)

    batch_size, channels, height, width = image_in_space.shape
    device = image_in_space.device
    dtype = image_in_space.dtype

    image_flat = image_in_space.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

    quantized_images_batch = []
    final_centroids_list = []
    final_num_colors = num_colors # Начальное значение K

    # --- Определение K через DBI (если нужно) ---
    if auto_num_colors and batch_size > 0:
        print("Determining optimal number of colors using DBI...")
        # Используем RGB для DBI
        pixels_rgb_for_dbi = from_quantize_space(image_in_space[0].unsqueeze(0), processing_space).squeeze(0)
        pixels_flat_rgb_for_dbi = pixels_rgb_for_dbi.permute(1, 2, 0).reshape(height * width, channels)
        num_samples = min(sample_size_dbi, pixels_flat_rgb_for_dbi.shape[0])
        if num_samples > 1:
            pixels_sampled = pixels_flat_rgb_for_dbi[torch.randperm(pixels_flat_rgb_for_dbi.shape[0], device=device)[:num_samples]].float()
            try:
                best_k_dbi = determine_optimal_num_colors_dbi(pixels_sampled, 'kmeans', max_k=auto_k_range)
                final_num_colors = best_k_dbi
                print(f"Auto detected optimal colors (DBI): {final_num_colors}")
            except Exception as dbi_err:
                print(f"Error during DBI calculation: {dbi_err}. Using default K={num_colors}.")
                final_num_colors = num_colors
        else:
             print("Warning: Not enough pixels to sample for DBI. Using default K.")
             final_num_colors = num_colors

    # --- Обработка батча ---
    for i in range(batch_size):
        pixels = image_flat[i]
        if pixels.shape[0] == 0:
            quantized_images_batch.append(image_in_space[i].unsqueeze(0))
            final_centroids_list.append(torch.empty((0, channels), device=device, dtype=dtype))
            continue

        labels, centroids = None, None
        try:
            method_clean = method.strip().lower()
            print(f"DEBUG: run_color_quantization - Batch {i}, Method: '{method_clean}', K: {final_num_colors}")

            # --- Вызов методов квантования ---
            if method_clean == "kmeans":
                labels, centroids = kmeans_quantization(pixels, final_num_colors, kmeans_max_iter)
            elif method_clean == "median_cut":
                labels, centroids = median_cut_quantization(pixels, final_num_colors)
            elif method_clean == "wu":
                pixels_rgb = from_quantize_space(pixels.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), processing_space).squeeze()
                if pixels_rgb.ndim == 1: pixels_rgb = pixels_rgb.unsqueeze(0)
                labels, centroids_rgb = wu_quantization(pixels_rgb, final_num_colors)
                centroids = to_quantize_space(centroids_rgb.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), processing_space).squeeze()
                if centroids.ndim == 1: centroids = centroids.unsqueeze(0)
            elif method_clean == "octree":
                pixels_rgb = from_quantize_space(pixels.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), processing_space).squeeze()
                if pixels_rgb.ndim == 1: pixels_rgb = pixels_rgb.unsqueeze(0)
                labels, centroids_rgb = octree_quantization_impl(pixels_rgb, final_num_colors)
                centroids = to_quantize_space(centroids_rgb.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), processing_space).squeeze()
                if centroids.ndim == 1: centroids = centroids.unsqueeze(0)
            elif method_clean == "sq":
                labels, centroids = sq_quantization(
                    pixels,
                    final_num_colors,
                    processing_space, # Передаем пространство для нормализации/проекции
                    iterations_factor=sq_iterations_factor,
                    learning_rate_initial=sq_learning_rate_initial,
                    learning_rate_decay_time=sq_learning_rate_decay_time
                )
            else:
                raise ValueError(f"Unknown color quantization method (cleaned): '{method_clean}'")

            # --- Фильтр маленьких кластеров ---
            if centroids is not None and centroids.shape[0] > 1 and min_pixel_area > 1 and labels is not None:
                # Фильтруем *после* получения результата от основного метода
                original_centroid_count = centroids.shape[0]
                labels, centroids = filter_small_clusters(pixels, labels, centroids, min_pixel_area)
                if centroids.shape[0] < original_centroid_count:
                    print(f"Batch {i}: Filtered small clusters: {original_centroid_count} -> {centroids.shape[0]}")

            # --- Применение квантованных цветов ---
            if centroids is None or centroids.shape[0] == 0 or labels is None:
                 print(f"Warning: Quantization failed or produced no centroids for batch item {i}. Using original.")
                 quantized_image = image_in_space[i]
                 final_centroids_list.append(torch.empty((0, channels), device=device, dtype=dtype))
            else:
                 labels = labels.clamp(0, centroids.shape[0] - 1)
                 quantized_pixels = centroids[labels].to(dtype) # Применяем финальные центроиды
                 quantized_image = quantized_pixels.permute(1, 0).reshape(channels, height, width)
                 final_centroids_list.append(centroids.clone())

            quantized_images_batch.append(quantized_image.unsqueeze(0))

        except Exception as e:
            print(f"Error during color quantization for batch item {i} using '{method}' in space '{processing_space}': {e}")
            traceback.print_exc()
            quantized_images_batch.append(image_in_space[i].unsqueeze(0))
            final_centroids_list.append(torch.empty((0, channels), device=device, dtype=dtype))

    quantized_images_final = torch.cat(quantized_images_batch, dim=0)
    final_centroids_batch0 = final_centroids_list[0] if final_centroids_list else torch.empty((0, channels), device=device, dtype=torch.float32)

    # Возвращаем изображение и центроиды в processing_space
    return quantized_images_final, final_centroids_batch0.float()

def determine_optimal_num_colors_dbi(pixels_rgb_sampled, method_for_dbi, max_k=16, sample_size=5000, min_k=2):
     """ Determines the optimal number of colors (K) using the Davies-Bouldin Index. """
     # (код determine_optimal_num_colors_dbi без изменений)
     num_unique = torch.unique(pixels_rgb_sampled, dim=0).shape[0]
     if num_unique <= min_k:
         print(f"DBI: Found <= {min_k} unique colors ({num_unique}). Returning this count.")
         return max(1, num_unique)

     max_k = min(max_k, num_unique)
     if min_k > max_k: min_k = max_k
     if min_k < 2: min_k = 2 # DBI requires at least 2 clusters
     if min_k > max_k: return min_k

     K_range = range(min_k, max_k + 1)
     best_k = min_k
     best_dbi = float('inf')
     dbi_values = {}

     print(f"Calculating DBI for K={min_k} to {max_k} (sample size: {pixels_rgb_sampled.shape[0]})...")
     for k in K_range:
         try:
             # Use KMeans (as it's standard for DBI) regardless of the main method chosen
             # Ensure pixels are float32
             labels, centroids = kmeans_quantization(pixels_rgb_sampled.float(), k, max_iters=15)
             if centroids.shape[0] < 2: continue # Need at least 2 resulting clusters for DBI

             # Calculate DBI using float32 pixels and centroids
             dbi = calculate_dbi(pixels_rgb_sampled.float(), labels, centroids.float())
             dbi_values[k] = dbi
             #print(f"  K={k}, DBI={dbi:.4f}") # Optional progress print
             if math.isfinite(dbi) and dbi < best_dbi:
                 best_dbi = dbi
                 best_k = k
         except Exception as e:
             print(f"  Warning: Error calculating DBI for K={k}: {e}")
             traceback.print_exc() # Print traceback for debugging
             continue

     if best_dbi == float('inf'):
         print(f"DBI calculation failed to find a valid index. Returning default k={min_k}")
         return min_k

     #print(f"DBI calculation finished. Best K={best_k} with DBI={best_dbi:.4f}")
     return best_k
