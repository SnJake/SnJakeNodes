# --- START OF FILE pixelart/pixelo_ops.py ---

import torch
import kornia
import kornia.color as kc
import kornia.filters as kf
import kornia.morphology as km
import torch.nn.functional as F
import math

def _calculate_local_stats_unfold(image_gray, kernel_size):
    """Calculates local min, max, median using unfold (approximate median)."""
    B, C, H, W = image_gray.shape
    assert C == 1, "Input must be grayscale"
    device = image_gray.device
    padding = kernel_size // 2

    # Unfold creates patches view: (B, C * kernel_h * kernel_w, L) where L is number of patches
    patches = F.unfold(image_gray, kernel_size=kernel_size, padding=padding)
    # Reshape to (B, C, kernel_h * kernel_w, L) -> (B, kernel_h * kernel_w, L)
    patches = patches.view(B, C, kernel_size * kernel_size, -1).squeeze(1)

    local_min = torch.min(patches, dim=1, keepdim=True)[0]
    local_max = torch.max(patches, dim=1, keepdim=True)[0]
    # Approximate median using median of unfolded patches
    # Note: torch.median returns values and indices
    local_median = torch.median(patches, dim=1, keepdim=True)[0]

    # Fold back to image shape: (B, C, H, W)
    # Create an output tensor of the correct size
    output_shape = (B, 1, H, W)
    # Use fold with a kernel of ones for averaging unfolded patches - we need direct value assignment
    # Instead, reshape the L dimension back into H * W
    local_min_img = local_min.view(B, 1, H, W)
    local_max_img = local_max.view(B, 1, H, W)
    local_median_img = local_median.view(B, 1, H, W)

    return local_min_img, local_max_img, local_median_img

def contrast_aware_outline_expansion(image_rgb, median_kernel_size=5, morph_kernel_size=3):
    """
    Implements the Contrast-Aware Outline Expansion step.
    Args:
        image_rgb (torch.Tensor): Input RGB image (B, 3, H, W), float [0, 1].
        median_kernel_size (int): Kernel size for local stats calculation.
        morph_kernel_size (int): Kernel size for morphological operations.
    Returns:
        torch.Tensor: Processed RGB image (B, 3, H, W).
    """
    print(f"Applying PixelOE Outline Expansion (median_k={median_kernel_size}, morph_k={morph_kernel_size})...")
    device = image_rgb.device
    dtype = image_rgb.dtype
    image_rgb_float = image_rgb.float() # Work with float

    # 1. Weight Map Generation
    image_gray = kc.rgb_to_grayscale(image_rgb_float) # (B, 1, H, W)

    # Calculate Local Stats (Min, Max, Median)
    # Using Kornia filters might be simpler/faster if available and suitable
    # local_median = kf.median_blur(image_gray, median_kernel_size) # Kornia median blur
    # For min/max, Kornia morphology can be used, but might be slow
    # Using unfold is an alternative, but median is tricky. Let's try Kornia first.
    pad_size = median_kernel_size // 2
    # Pad manually for min/max pool as Kornia erosion/dilation work differently
    image_gray_padded = F.pad(image_gray, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    local_min = -F.max_pool2d(-image_gray_padded, kernel_size=median_kernel_size, stride=1)
    local_max = F.max_pool2d(image_gray_padded, kernel_size=median_kernel_size, stride=1)
    # Use median blur for median
    local_median = kf.median_blur(image_gray, median_kernel_size)

    # Calculate "Bright" and "Dark" distances
    bright_dist = torch.clamp(local_max - local_median, min=0)
    dark_dist = torch.clamp(local_median - local_min, min=0)
    total_dist = bright_dist + dark_dist + 1e-6 # Avoid division by zero

    # Weight 1: Prioritize brighter details in darker areas (higher weight for bright_dist when median is low)
    # Let's interpret "prioritize brighter details" as weight proportional to bright_dist / total_dist
    # And "in darker areas" as multiplying by (1 - local_median)
    weight1_raw = (bright_dist / total_dist) * (1.0 - local_median)

    # Weight 2: Based on distance between brighter/darker details (total contrast)
    # Let's interpret this as simply the total_dist normalized
    weight2_raw = (local_max - local_min) # Range is a good measure of distance/contrast

    # Combine weights (simple average for now, could be weighted sum)
    # Normalize weights first? Let's normalize the result.
    # weight_map = (weight1_raw + weight2_raw) / 2.0
    # Simpler approach: Higher weight means more dilation bias.
    # Use relative brightness/darkness distance
    # Weight > 0.5 favors dilation (bright details), Weight < 0.5 favors erosion (dark details)
    # Center at 0.5, scale by contrast (weight2_raw) maybe?
    weight_map = 0.5 + (bright_dist - dark_dist) / total_dist # Ranges from 0 to 1
    # Optional: Modulate by overall contrast? weight_map = 0.5 + (bright_dist - dark_dist) / total_dist * (local_max - local_min)

    # Normalize the final weight map to [0, 1]
    w_min = torch.min(weight_map.view(weight_map.shape[0], -1), dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    w_max = torch.max(weight_map.view(weight_map.shape[0], -1), dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    weight_map_norm = (weight_map - w_min) / (w_max - w_min + 1e-6)
    weight_map_norm = weight_map_norm.clamp(0.0, 1.0)

    # 2. Selective Morphological Operations
    morph_kernel = torch.ones(morph_kernel_size, morph_kernel_size, device=device)
    # Erode shrinks bright regions (operates per channel)
    image_eroded = km.erosion(image_rgb_float, morph_kernel)
    # Dilate expands bright regions
    image_dilated = km.dilation(image_rgb_float, morph_kernel)

    # Blend based on weight map (expand weight map to 3 channels)
    weight_map_rgb = weight_map_norm.repeat(1, 3, 1, 1)
    blended_image = image_eroded * (1.0 - weight_map_rgb) + image_dilated * weight_map_rgb

    # 3. Morphological Closing and Opening
    # Closing: Dilate then Erode (removes small dark spots)
    closed_image = km.closing(blended_image, morph_kernel)
    # Opening: Erode then Dilate (removes small bright spots)
    opened_image = km.opening(closed_image, morph_kernel)

    print("PixelOE Outline Expansion finished.")
    return opened_image.to(dtype) # Convert back to original dtype

def find_representative_pixel_lab(lab_patch):
    """
    Selects the representative LAB pixel from a patch based on L channel stats.
    Args:
        lab_patch (torch.Tensor): Input LAB patch (C=3, H, W).
    Returns:
        torch.Tensor: The selected LAB pixel (3,).
    """
    if lab_patch.numel() == 0:
         # Handle empty patch - return black or average? Let's return mid-gray LAB
         return torch.tensor([50.0, 0.0, 0.0], device=lab_patch.device, dtype=lab_patch.dtype)

    l_channel = lab_patch[0, :, :] # (H, W)
    a_channel = lab_patch[1, :, :]
    b_channel = lab_patch[2, :, :]

    if l_channel.numel() == 0:
        return torch.tensor([50.0, 0.0, 0.0], device=lab_patch.device, dtype=lab_patch.dtype)

    l_flat = l_channel.reshape(-1)
    a_flat = a_channel.reshape(-1)
    b_flat = b_channel.reshape(-1)

    # Calculate L channel stats
    l_min_val, l_min_idx = torch.min(l_flat, dim=0)
    l_max_val, l_max_idx = torch.max(l_flat, dim=0)
    l_median_val = torch.median(l_flat).values # Получаем 0-мерный тензор значения медианы
    # l_mean_val = torch.mean(l_flat.float()) # Not directly used in simplified logic

    # Find center pixel index (approximate)
    patch_h, patch_w = l_channel.shape
    center_y, center_x = patch_h // 2, patch_w // 2
    center_idx_flat = center_y * patch_w + center_x
    # Ensure center index is valid
    center_idx_flat = min(center_idx_flat, l_flat.shape[0] - 1)

    # Skewness check (simplified)
    skew_low = (l_median_val - l_min_val) > (l_max_val - l_median_val)
    skew_high = (l_max_val - l_median_val) > (l_median_val - l_min_val)

    selected_idx = center_idx_flat # Default to center pixel

    if skew_low:
        selected_idx = l_min_idx
        # print("Skew Low, selecting Min L pixel")
    elif skew_high:
        selected_idx = l_max_idx
        # print("Skew High, selecting Max L pixel")
    # else:
        # print("No significant skew, selecting Center pixel")

    # Get the full LAB vector of the selected pixel
    selected_lab_pixel = torch.stack([
        l_flat[selected_idx],
        a_flat[selected_idx],
        b_flat[selected_idx]
    ])

    return selected_lab_pixel

def contrast_aware_downsample(image_rgb, pixel_size):
    """
    Implements the Contrast-Aware Downsampling using LAB space.
    Args:
        image_rgb (torch.Tensor): Input RGB image (B, 3, H, W), float [0, 1].
        pixel_size (int): The size of the downsampling blocks.
    Returns:
        torch.Tensor: Downscaled RGB image (B, 3, H_new, W_new).
    """
    print(f"Applying PixelOE Contrast-Aware Downsampling (pixel_size={pixel_size})...")
    B, C, H, W = image_rgb.shape
    device = image_rgb.device
    dtype = image_rgb.dtype
    image_rgb_float = image_rgb.float()

    if pixel_size <= 0: pixel_size = 1

    target_h = max(1, H // pixel_size)
    target_w = max(1, W // pixel_size)

    # Convert to LAB
    try:
        image_lab = kc.rgb_to_lab(image_rgb_float) # (B, 3, H, W)
    except Exception as e:
        print(f"Error converting to LAB: {e}. Falling back to standard downsampling.")
        # Fallback to nearest neighbor downscale
        return F.interpolate(image_rgb, size=(target_h, target_w), mode='nearest')

    # Prepare output tensor
    downscaled_lab = torch.zeros((B, 3, target_h, target_w), device=device, dtype=image_lab.dtype)

    # Iterate over the target downscaled grid
    for b in range(B):
        for y_out in range(target_h):
            for x_out in range(target_w):
                # Define the input patch boundaries
                y_start = y_out * pixel_size
                y_end = min((y_out + 1) * pixel_size, H)
                x_start = x_out * pixel_size
                x_end = min((x_out + 1) * pixel_size, W)

                if y_start >= y_end or x_start >= x_end: continue # Skip empty patches

                # Extract the LAB patch for this output pixel
                lab_patch = image_lab[b, :, y_start:y_end, x_start:x_end] # (3, patch_h, patch_w)

                # --- L Channel Processing ---
                # Select the representative pixel based on L channel stats
                selected_lab_pixel = find_representative_pixel_lab(lab_patch)
                downscaled_lab[b, :, y_out, x_out] = selected_lab_pixel # Assign selected pixel

                # --- A and B Channel Processing (Original PixelOE description says median filter) ---
                # Let's stick to the logic of find_representative_pixel_lab selecting the *whole* LAB pixel
                # If we wanted separate median filtering for A/B:
                # if lab_patch[1,:,:].numel() > 0:
                #     a_median = torch.median(lab_patch[1, :, :].reshape(-1))[0]
                # else:
                #     a_median = torch.tensor(0.0, device=device, dtype=dtype)
                # if lab_patch[2,:,:].numel() > 0:
                #     b_median = torch.median(lab_patch[2, :, :].reshape(-1))[0]
                # else:
                #     b_median = torch.tensor(0.0, device=device, dtype=dtype)
                # # Assign L from find_pixel, A/B from median
                # downscaled_lab[b, 0, y_out, x_out] = selected_lab_pixel[0]
                # downscaled_lab[b, 1, y_out, x_out] = a_median
                # downscaled_lab[b, 2, y_out, x_out] = b_median

    # Convert back to RGB
    try:
        downscaled_rgb = kc.lab_to_rgb(downscaled_lab).clamp(0.0, 1.0)
    except Exception as e:
        print(f"Error converting back to RGB: {e}. Returning LAB (may cause issues downstream).")
        downscaled_rgb = downscaled_lab # Return LAB if conversion fails

    print("PixelOE Contrast-Aware Downsampling finished.")
    return downscaled_rgb.to(dtype) # Convert back to original dtype

# --- END OF FILE pixelart/pixelo_ops.py ---