import torch
import kornia
import kornia.filters

def box_blur(img, radius):
    """Applies a box blur using Kornia."""
    # (Скопируйте код _box_blur из вашего узла)
    kernel_size = (2 * radius + 1, 2 * radius + 1)
    # Use Kornia's box_blur with reflection padding for better edge handling
    try:
        blurred = kornia.filters.box_blur(img.float(), kernel_size, border_type='reflect')
        return blurred.to(img.dtype) # Convert back to original dtype
    except Exception as e:
        print(f"Warning: Kornia box_blur failed: {e}. Returning original image.")
        return img


def guided_filter(input_image, guide_image, radius, eps):
    """Applies a Guided Filter."""
    # (Скопируйте код _guided_filter из вашего узла)
    # Ensure inputs are float32 for calculations
    input_f = input_image.float()
    guide_f = guide_image.float()

    mean_I = box_blur(guide_f, radius)
    mean_P = box_blur(input_f, radius)
    mean_Ip = box_blur(guide_f * input_f, radius)
    mean_II = box_blur(guide_f * guide_f, radius)

    cov_Ip = mean_Ip - mean_I * mean_P
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_P - a * mean_I

    mean_a = box_blur(a, radius)
    mean_b = box_blur(b, radius)

    # Calculate output and convert back to original dtype
    q = (mean_a * guide_f + mean_b).to(input_image.dtype)
    return q

def apply_smoothing(image, smoothing, filter_type, edge_preservation, advanced_filter):
    """Applies smoothing and optional advanced filters."""
    # (Скопируйте код _apply_smoothing из вашего узла, заменив self._guided_filter/box_blur)
    if smoothing <= 0 and advanced_filter == "none":
        return image # No smoothing/filtering needed

    # Work with float32 for filtering operations
    smoothed_image = image.clone().float()
    image_float = image.float() # Keep original float version for guidance/details

    # --- Basic Smoothing Filter ---
    if smoothing > 0:
        # Determine kernel size based on smoothing strength
        # Ensure kernel size is odd and at least 3
        kernel_size = max(3, int(round(smoothing * 3) * 2 + 1))

        try:
            if filter_type == "gaussian":
                sigma = smoothing # Use smoothing value directly as sigma
                smoothed_image = kornia.filters.gaussian_blur2d(
                    smoothed_image, (kernel_size, kernel_size), (sigma, sigma), border_type='reflect'
                )
            elif filter_type == "median":
                smoothed_image = kornia.filters.median_blur(smoothed_image, (kernel_size, kernel_size))
            elif filter_type == "bilateral":
                # Set parameters for bilateral filter
                sigma_space = smoothing
                # Heuristic for sigma_color, often smaller than sigma_space
                sigma_color = max(0.1, smoothing * 0.5) # Ensure non-zero sigma_color
                smoothed_image = kornia.filters.bilateral_blur(
                    smoothed_image, kernel_size=kernel_size,
                    sigma_color=sigma_color,
                    sigma_space=(sigma_space, sigma_space),
                    border_type='reflect'
                )
            # else: unknown filter type, smoothed_image remains unchanged from clone
        except Exception as smooth_err:
            print(f"Warning: Basic smoothing filter '{filter_type}' failed: {smooth_err}. Skipping basic smooth.")
            smoothed_image = image_float # Revert to original if smoothing failed

    # --- Edge Preservation (Unsharp Masking, applied if basic smoothing occurred) ---
    # Only apply if smoothing actually happened and edge_preservation is on
    if smoothing > 0 and edge_preservation:
        # Calculate detail (difference between original and smoothed)
        detail = image_float - smoothed_image
        # Add detail back to the smoothed image. Factor controls strength.
        sharpen_factor = 1.0 # Adjust if needed
        processed_image = smoothed_image + detail * sharpen_factor
        processed_image = torch.clamp(processed_image, 0.0, 1.0) # Clamp result
    else:
        # If no smoothing or no edge preservation, use the result from smoothing step
        processed_image = smoothed_image

    # --- Advanced Filters (applied AFTER basic smoothing/sharpening) ---
    if advanced_filter != "none":
        adv_kernel_size = 5 # Common kernel size for these filters
        try:
            if advanced_filter == "guided":
                # Use original float image as the guide for better edge preservation
                guide = image_float
                radius = adv_kernel_size // 2
                eps = 0.01**2 # Standard epsilon value, adjust if needed
                processed_image = guided_filter(processed_image, guide, radius, eps) # Pass current image and guide
            elif advanced_filter == "kuwahara":
                processed_image = kornia.filters.kuwahara(processed_image, kernel_size=adv_kernel_size)
            # else: unknown filter, processed_image remains unchanged
        except Exception as adv_filter_err:
            print(f"Warning: Advanced filter '{advanced_filter}' failed: {adv_filter_err}. Skipping advanced filter.")
            # processed_image remains the result from previous steps

    # Clamp final result and convert back to original dtype
    return processed_image.clamp(0.0, 1.0).to(image.dtype)