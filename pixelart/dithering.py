import torch

# --- Матрицы и Константы Дизеринга ---
# (Скопируйте BAYER_, HALFTONE_, ORDERED_PATTERNS, DIFFUSION_PATTERNS из вашего узла)
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
    # WhiteNoise handled separately
}


def apply_dithering(
    image_rgb_source,       # Original downscaled image in RGB [0,1]
    palette_rgb,            # Target RGB palette [0,1]
    dither_pattern,
    dither_strength,
    color_distance_threshold,
    apply_fixed_palette_func=None
    ):
    """Applies dithering to an image using a specified palette. Operates entirely in RGB."""
    B, C, H, W = image_rgb_source.shape
    device = image_rgb_source.device
    if apply_fixed_palette_func is None:
        from .quantization import apply_fixed_palette as apply_fixed_palette_func

    if palette_rgb is None or palette_rgb.shape[0] < 1:
        print("Warning: Dithering requires a palette. Skipping.")
        if palette_rgb is not None and palette_rgb.shape[0] > 0:
            # Use the passed function
            return apply_fixed_palette_func(image_rgb_source, palette_rgb)
        else:
            return image_rgb_source

    if palette_rgb.shape[0] < 2 and dither_pattern != "WhiteNoise": # Allow noise even for 1 color
        print(f"Warning: Dithering pattern '{dither_pattern}' requires >= 2 palette colors. Mapping to the single color.")
        return apply_fixed_palette_func(image_rgb_source, palette_rgb)

    # Work with float32 for accuracy
    dither_image_buffer = image_rgb_source.clone().float() # Buffer to accumulate errors/noise
    final_dithered_image = torch.zeros_like(image_rgb_source, dtype=torch.float32) # Output image

    # --- Ordered Dithering ---
    if dither_pattern in ORDERED_PATTERNS:
        bayer_matrix = ORDERED_PATTERNS[dither_pattern].to(device)
        matrix_size = bayer_matrix.shape[0]
        # Calculate repetitions needed to cover the image
        repeat_h = (H + matrix_size - 1) // matrix_size
        repeat_w = (W + matrix_size - 1) // matrix_size
        # Tile the matrix and crop to image size
        full_bayer = bayer_matrix.repeat(repeat_h, repeat_w)[:H, :W].unsqueeze(0).unsqueeze(0) # Add batch and channel dims

        num_palette_colors = palette_rgb.shape[0]
        # Scale the dither effect based on the number of colors
        dither_intensity_divisor = max(1, num_palette_colors - 1) # Avoid division by zero for 1 color palette
        dither_amount = (full_bayer * dither_strength) / dither_intensity_divisor

        # Add dither noise and clamp
        image_plus_dither = (dither_image_buffer + dither_amount).clamp(0.0, 1.0)
        # Quantize the noise-added image to the palette
        final_dithered_image = apply_fixed_palette_func(image_plus_dither, palette_rgb)

    # --- White Noise Dithering ---
    elif dither_pattern == "WhiteNoise":
        noise = (torch.rand_like(dither_image_buffer) - 0.5) # Centered noise [-0.5, 0.5]
        num_palette_colors = palette_rgb.shape[0]
        # Scale noise effect based on palette size and strength
        noise_scale = dither_strength / max(1, num_palette_colors - 1)
        image_plus_noise = (dither_image_buffer + noise * noise_scale).clamp(0.0, 1.0)
        final_dithered_image = apply_fixed_palette_func(image_plus_noise, palette_rgb)

    # --- Error Diffusion Dithering ---
    elif dither_pattern in DIFFUSION_PATTERNS:
        pattern = DIFFUSION_PATTERNS[dither_pattern]
        threshold_sq = -1.0
        if color_distance_threshold > 0:
            threshold_sq = color_distance_threshold ** 2
        palette_rgb_float = palette_rgb.float()

        output_batch = []
        for b in range(B):
            print(f"Applying {dither_pattern} Dithering (Strength: {dither_strength:.2f}) (Batch {b+1}/{B})...")
            current_image_buffer = dither_image_buffer[b].clone() # C, H, W
            current_output_image = torch.zeros_like(current_image_buffer) # C, H, W

            for y in range(H):
                # Serpentine scan direction
                x_range = range(W) if y % 2 == 0 else range(W - 1, -1, -1)

                for x in x_range:
                    old_pixel = current_image_buffer[:, y, x].clone().float() # Ensure float

                    # Find NEAREST color in the palette (RGB space assumed)
                    # Ensure palette is float
                    diff = palette_rgb_float - old_pixel.unsqueeze(0) # Use broadcasting
                    distances_sq = torch.sum(diff * diff, dim=1)
                    min_dist_sq, closest_index = torch.min(distances_sq, dim=0)
                    new_pixel = palette_rgb_float[closest_index]

                    current_output_image[:, y, x] = new_pixel # Assign palette color

                    quant_error = old_pixel - new_pixel # Calculate error

                    # Check distance threshold
                    should_diffuse = (threshold_sq < 0) or (min_dist_sq.item() <= threshold_sq)

                    if should_diffuse:
                        error_to_distribute = quant_error * dither_strength
                        for dx, dy, factor in pattern:
                            current_dx = dx if y % 2 == 0 else -dx # Adjust for serpentine
                            nx, ny = x + current_dx, y + dy
                            if 0 <= nx < W and 0 <= ny < H:
                                current_image_buffer[:, ny, nx] += error_to_distribute * factor
                                # Clamp buffer after accumulating error for a pixel's neighbors
            # Clamp buffer *outside* the inner loops? Might be better but riskier. Let's clamp inside.
            # Clamp the entire buffer once per batch item maybe? Seems safer to clamp inside.
            # Clamping inside the innermost loop as before:
                                current_image_buffer[:, ny, nx].clamp_(0.0, 1.0) # Clamp in-place

            output_batch.append(current_output_image.unsqueeze(0)) # Add batch dim back

        if output_batch:
            final_dithered_image = torch.cat(output_batch, dim=0)
        else:
            final_dithered_image = image_rgb_source # Fallback if batch was empty

    else: # Unknown pattern
        print(f"Warning: Unknown dither pattern '{dither_pattern}'. Mapping to palette without dithering.")
        final_dithered_image = apply_fixed_palette_func(image_rgb_source, palette_rgb)

    return final_dithered_image.clamp(0.0, 1.0)
