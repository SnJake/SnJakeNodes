import torch
import kornia
import kornia.color as kc
import kornia.filters

def compute_adaptive_pixel_sizes(image, base_pixel_size, metric):
    """Computes a map of pixel sizes based on image content."""
    # (Скопируйте код _compute_adaptive_pixel_sizes из вашего узла)
    # ... (код без изменений) ...
    B, C, H, W = image.shape
    device = image.device
    image_float = image.float() # Ensure float

    if metric == "gradient":
        image_gray = kc.rgb_to_grayscale(image_float) # (B, 1, H, W)
        # Compute gradients using Sobel filters
        # kornia returns gradients as (B, C, 2, H, W) where C=1 for grayscale, 2 is (dy, dx)
        gradients = kornia.filters.spatial_gradient(image_gray, mode='sobel', order=1) # (B, 1, 2, H, W)
        # Magnitude: sqrt(dx^2 + dy^2) - Note the order in kornia output!
        gradient_magnitude = torch.sqrt(gradients[:, :, 1, :, :]**2 + gradients[:, :, 0, :, :]**2) # (B, 1, H, W)
        gradient_magnitude = gradient_magnitude.squeeze(1) # (B, H, W)

        # Normalize magnitude per image in batch to [0, 1]
        gmin = torch.amin(gradient_magnitude.view(B,-1), dim=1, keepdim=True)[0].view(B,1,1)
        gmax = torch.amax(gradient_magnitude.view(B,-1), dim=1, keepdim=True)[0].view(B,1,1)
        norm_gradient = (gradient_magnitude - gmin) / (gmax - gmin + 1e-8)

        # Invert: high gradient -> small pixel, low gradient -> large pixel
        inverted_metric = 1.0 - norm_gradient # (B, H, W)

    elif metric == "saturation":
        hsv = kc.rgb_to_hsv(image_float.clamp(0,1)) # Ensure input is valid RGB [0,1]
        saturation = hsv[:, 1, :, :] # (B, H, W) - Saturation is channel 1

        # Normalize saturation per image in batch to [0, 1]
        smin = torch.amin(saturation.view(B,-1), dim=1, keepdim=True)[0].view(B,1,1)
        smax = torch.amax(saturation.view(B,-1), dim=1, keepdim=True)[0].view(B,1,1)
        norm_saturation = (saturation - smin)/(smax - smin + 1e-8)

        # Invert: high saturation -> small pixel(?), low saturation -> large pixel
        inverted_metric = 1.0 - norm_saturation # (B, H, W)

    else: # Fallback to constant size based on base_pixel_size
        print(f"Warning: Unknown adaptive metric '{metric}'. Using constant size.")
        # This function just calculates the map, adaptive_downscale uses it.
        # Return a map indicating the base size everywhere.
        pixel_size_map = torch.full((B, H, W), base_pixel_size, dtype=torch.int, device=device)
        return pixel_size_map.clamp(min=1) # Ensure at least 1

    # Map inverted metric [0, 1] to pixel size range
    # Define min/max pixel size based on base_pixel_size
    min_pix_size = max(1, base_pixel_size // 3 + 1) # Ensure min is at least 1
    max_pix_size = max(min_pix_size + 1, base_pixel_size + base_pixel_size // 2) # Ensure max > min

    # Linear mapping: size = min + inverted * (max - min)
    pixel_size_map = inverted_metric * (max_pix_size - min_pix_size) + min_pix_size
    # Round to nearest integer and clamp
    pixel_size_map = torch.clamp(pixel_size_map.round().int(), min=1) # Ensure min size is 1

    return pixel_size_map


def adaptive_downscale(image, pixel_size_map):
    """Performs adaptive downscaling using the pixel size map (slow version)."""
    # (Скопируйте код _adaptive_downscale из вашего узла)
    # ... (код без изменений) ...
    batch_size, channels, height, width = image.shape
    device = image.device
    dtype = image.dtype
    # Create output tensor filled with zeros initially
    adaptive_image = torch.zeros_like(image, dtype=dtype)

    for b in range(batch_size):
        # Keep track of visited pixels to avoid redundant calculations
        visited = torch.zeros((height, width), dtype=torch.bool, device=device)

        # Iterate through the image, processing unvisited pixels
        for y in range(height):
            for x in range(width):
                if not visited[y, x]:
                    # Get the adaptive pixel block size for this starting pixel
                    size = pixel_size_map[b, y, x].item()
                    # Calculate the block boundaries, clamping to image dimensions
                    y_end = min(y + size, height)
                    x_end = min(x + size, width)

                    # Extract the region corresponding to the adaptive block
                    region = image[b, :, y:y_end, x:x_end]

                    # Calculate the average color of the region (use float for mean)
                    if region.numel() > 0: # Ensure region is not empty
                        avg_color = torch.mean(region.float(), dim=(1, 2), keepdim=True) # Shape (C, 1, 1)
                    else: # Should not happen with valid coords, but safety check
                        avg_color = image[b, :, y, x].unsqueeze(-1).unsqueeze(-1).float() # Use single pixel color

                    # Fill the corresponding block in the output image with the average color
                    adaptive_image[b, :, y:y_end, x:x_end] = avg_color.expand(-1, y_end - y, x_end - x).to(dtype)

                    # Mark the processed block as visited
                    visited[y:y_end, x:x_end] = True

    return adaptive_image