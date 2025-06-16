import torch
import torch.nn.functional as F
import math

# ==============================================================================
# Standalone bislerp function and its dependencies
# (Modified to match ComfyUI's utils.py logic)
# ==============================================================================

def bislerp(samples: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """
    Resizes a batch of samples (N, C, H, W) to a new width and height using
    Bilinear Slerp (Spherical Linear Interpolation for direction,
    Linear Interpolation for magnitude).

    Matches the logic from ComfyUI's `utils.py`.

    Args:
        samples (torch.Tensor): Input tensor of shape (N, C, H, W).
        width (int): Target width.
        height (int): Target height.

    Returns:
        torch.Tensor: Resized tensor of shape (N, C, height, width).
    """

    def slerp(b1: torch.Tensor, b2: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Performs Spherical Linear Interpolation (Slerp) between two batches
        of vectors, b1 and b2, according to ratio r.
        Matches ComfyUI's slerp implementation.

        Args:
            b1 (torch.Tensor): First batch of vectors (N, C).
            b2 (torch.Tensor): Second batch of vectors (N, C).
            r (torch.Tensor): Interpolation ratio (N, 1), values between 0 and 1.

        Returns:
            torch.Tensor: Interpolated batch of vectors (N, C).
        """
        if b1.shape != b2.shape or b1.shape[0] != r.shape[0] or r.shape[1] != 1:
             raise ValueError(f"Mismatched shapes for slerp: "
                              f"b1: {b1.shape}, b2: {b2.shape}, r: {r.shape}")

        c = b1.shape[-1]

        # norms
        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        # normalize (ComfyUI does not use eps here for division)
        # Handle potential division by zero if norms are 0, then correct
        # b1_normalized = b1 / b1_norms
        # b2_normalized = b2 / b2_norms
        # Using where to avoid NaN from 0/0 division if norm is 0
        # In PyTorch, 0/0 results in NaN.
        # If norm is 0, vector is 0. Normalized vector should be 0.
        b1_normalized = torch.where(b1_norms > 0, b1 / b1_norms, torch.zeros_like(b1))
        b2_normalized = torch.where(b2_norms > 0, b2 / b2_norms, torch.zeros_like(b2))


        # zero when norms are zero (ComfyUI's original explicit check)
        # This handles cases where norms might be extremely small but not exactly zero,
        # leading to large normalized vectors if not careful,
        # or if the previous division resulted in NaN for 0/0.
        # The `where` above should handle 0/0 correctly to 0.
        # This re-confirms for strict zero norms.
        b1_normalized[b1_norms.expand(-1, c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1, c) == 0.0] = 0.0
        
        # slerp
        dot = (b1_normalized * b2_normalized).sum(1)
        # ComfyUI does not clamp dot here. Potential for acos(>1) or acos(<-1) if precision issues.
        # However, to match 1:1, we omit the clamp.
        # dot = torch.clamp(dot, -1.0, 1.0) 

        omega = torch.acos(dot)
        so = torch.sin(omega)

        # Interpolation factors
        # ComfyUI does not use eps in the denominator for 'so'.
        # This can lead to div by zero if so is 0 (collinear vectors).
        # Edge cases below are intended to handle these situations.
        # r1 = torch.sin((1.0 - r.squeeze(1)) * omega) / (so + eps)
        # r2 = torch.sin(r.squeeze(1) * omega) / (so + eps)
        # res = r1.unsqueeze(1) * b1_normalized + r2.unsqueeze(1) * b2_normalized
        
        # ComfyUI's direct formula:
        # Create a mask for so == 0 to avoid division by zero if not caught by edge cases
        # This specific handling for so == 0 is not explicitly in ComfyUI's slerp,
        # it relies on the edge cases. For strict matching, we follow that.
        # If so is zero, omega is 0 or pi.
        # If omega is 0, dot is 1. If omega is pi, dot is -1.
        # These are handled by the edge cases.
        
        # If so is very near zero but not exactly, and not caught by edge cases,
        # this could still lead to issues.
        # For now, a direct translation of ComfyUI's math:
        factor1 = torch.sin((1.0 - r.squeeze(1)) * omega) / so
        factor2 = torch.sin(r.squeeze(1) * omega) / so
        
        res = factor1.unsqueeze(1) * b1_normalized + factor2.unsqueeze(1) * b2_normalized

        # Linearly interpolate magnitudes
        res *= (b1_norms * (1.0 - r) + b2_norms * r).expand(-1, c)

        # Handle edge cases:
        # 1. Vectors are nearly identical (dot product close to 1)
        # This also handles omega=0, so=0 case.
        same_mask = dot > (1.0 - 1e-5)
        res[same_mask] = b1[same_mask]

        # 2. Vectors are nearly opposite (dot product close to -1)
        # This also handles omega=pi, so=0 case.
        # ComfyUI: res[dot < 1e-5 - 1] which is dot < -0.99999
        opposite_mask = dot < (1e-5 - 1.0) # Equivalent to dot < (-1.0 + 1e-5)
        # For opposite vectors, ComfyUI falls back to linear interpolation.
        res[opposite_mask] = (b1 * (1.0 - r) + b2 * r)[opposite_mask]
        
        # Handle cases where 'so' was zero and not caught by edge cases (e.g. if 1e-5 threshold is too strict)
        # This might lead to NaNs if omega is an exact multiple of pi not caught by thresholds.
        # If `so` is 0, `factor1` and `factor2` would be NaN.
        # The edge case masks should ideally cover these.
        # If a NaN appears in `res` due to `0/0` in factors where `so` is 0, and it wasn't
        # covered by `same_mask` or `opposite_mask`, it would propagate.
        # For strict ComfyUI behavior, we assume the edge cases are sufficient.

        return res

    def generate_bilinear_data(length_old: int, length_new: int, device: torch.device):
        """
        Generates coordinates and ratios for 1D bilinear (linear) interpolation.
        Matches ComfyUI's `generate_bilinear_data` which uses F.interpolate 
        with default align_corners=False.

        Args:
            length_old (int): Original length.
            length_new (int): Target length.
            device (torch.device): Device for tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - ratios: Interpolation weights (1, 1, 1, length_new).
                - coords_1: Left neighbor indices (1, 1, 1, length_new).
                - coords_2: Right neighbor indices (1, 1, 1, length_new).
        """
        # --- ComfyUI's logic using F.interpolate (default align_corners=False) ---
        # Coords for the left/top pixel
        coords_1_float = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1,1,1,-1))
        # Note: F.interpolate with mode="bilinear" defaults to align_corners=False
        coords_1_float = F.interpolate(coords_1_float, size=(1, length_new), mode="bilinear", align_corners=False)
        ratios = coords_1_float - coords_1_float.floor() # Fractional part is the ratio for the right/bottom neighbor
        coords_1 = coords_1_float.floor().to(torch.int64) # Integer part is the index of the left/top neighbor

        # Coords for the right/bottom pixel
        coords_2_float = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1,1,1,-1)) + 1
        # Clamp the last coordinate for the right neighbor to avoid going out of bounds (before interpolation)
        # This means the rightmost pixel in the old grid, if chosen as a right neighbor, will effectively be itself.
        coords_2_float[:,:,:,-1] -= 1 
        coords_2_float = F.interpolate(coords_2_float, size=(1, length_new), mode="bilinear", align_corners=False)
        coords_2 = coords_2_float.floor().to(torch.int64) # Integer part is the index of the right/bottom neighbor

        # Ensure coords are within bounds [0, length_old - 1] after interpolation and floor.
        # F.interpolate might produce values slightly outside due to floating point,
        # or the logic might map near boundaries. Clamping is safe.
        coords_1 = torch.clamp(coords_1, 0, length_old - 1)
        coords_2 = torch.clamp(coords_2, 0, length_old - 1)
        
        # The ratio `ratios` is for interpolating towards coords_2.
        # So, final_value = v(coords_1) * (1 - ratios) + v(coords_2) * ratios
        # In slerp, r is the weight for b2 (the "second" or "right/bottom" point).
        # The `ratios` calculated from `coords_1_float - coords_1_float.floor()` is the weight for the
        # "ceil" or "right/bottom" coordinate.
        
        return ratios, coords_1, coords_2


    # --- Main bislerp logic starts here ---
    if samples.ndim != 4:
        raise ValueError(f"Input samples must be 4D (N, C, H, W), but got {samples.ndim}D")

    orig_dtype = samples.dtype
    samples = samples.float() # Perform calculations in float32 for precision
    n, c, h_orig, w_orig = samples.shape # Renamed h, w to h_orig, w_orig
    h_new, w_new = (height, width)

    if h_orig == h_new and w_orig == w_new:
        return samples.to(orig_dtype) # No resize needed

    # 1. Interpolate horizontally (along width dimension)
    # ComfyUI name original: ratios, coords_1, coords_2
    ratios_w, coords_1_w, coords_2_w = generate_bilinear_data(w_orig, w_new, samples.device)

    # Expand coordinates and ratios to match the input dimensions
    coords_1_w = coords_1_w.expand((n, c, h_orig, -1))
    coords_2_w = coords_2_w.expand((n, c, h_orig, -1))
    ratios_w = ratios_w.expand((n, 1, h_orig, -1)) # Ratio for slerp

    # Gather the left and right neighbors
    pass_1_w = samples.gather(dim=-1, index=coords_1_w)
    pass_2_w = samples.gather(dim=-1, index=coords_2_w)

    # Reshape for slerp
    pass_1_w = pass_1_w.movedim(1, -1).reshape(-1, c)
    pass_2_w = pass_2_w.movedim(1, -1).reshape(-1, c)
    ratios_w = ratios_w.movedim(1, -1).reshape(-1, 1)

    # Perform slerp
    result_w = slerp(pass_1_w, pass_2_w, ratios_w)

    # Reshape result back to image format
    result_w = result_w.reshape(n, h_orig, w_new, c).movedim(-1, 1)


    # 2. Interpolate vertically (along height dimension) using the results from step 1
    ratios_h, coords_1_h, coords_2_h = generate_bilinear_data(h_orig, h_new, samples.device)

    # Expand coordinates and ratios
    # Reshape for broadcasting and expansion: (1,1,length_new,1) for height-wise data
    coords_1_h = coords_1_h.reshape(1, 1, -1, 1).expand((n, c, -1, w_new))
    coords_2_h = coords_2_h.reshape(1, 1, -1, 1).expand((n, c, -1, w_new))
    ratios_h = ratios_h.reshape(1, 1, -1, 1).expand((n, 1, -1, w_new)) # Ratio for slerp

    # Gather the top and bottom neighbors from the horizontally interpolated result
    pass_1_h = result_w.gather(dim=-2, index=coords_1_h)
    pass_2_h = result_w.gather(dim=-2, index=coords_2_h)

    # Reshape for slerp
    pass_1_h = pass_1_h.movedim(1, -1).reshape(-1, c)
    pass_2_h = pass_2_h.movedim(1, -1).reshape(-1, c)
    ratios_h = ratios_h.movedim(1, -1).reshape(-1, 1)

    # Perform slerp
    result_h = slerp(pass_1_h, pass_2_h, ratios_h)

    # Reshape final result
    final_result = result_h.reshape(n, h_new, w_new, c).movedim(-1, 1)

    return final_result.to(orig_dtype) # Convert back to original dtype


# ==============================================================================
# Example Usage
# ==============================================================================
if __name__ == "__main__":
    # Ensure you have PyTorch installed: pip install torch

    # --- Parameters ---
    batch_size = 1 # Use 1 for easier comparison if printing values
    channels = 3
    input_height = 4
    input_width = 4
    output_height = 8
    output_width = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Using float32 for example to minimize dtype-related discrepancies during comparison
    # ComfyUI's slerp is sensitive, float16 can show more deviations or NaNs
    dtype = torch.float32
    # dtype = torch.float16 # Can be tested, but more prone to NaN/inf without safeguards

    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # --- Create dummy input data ---
    # Simple, predictable data
    input_tensor = torch.zeros(batch_size, channels, input_height, input_width, dtype=dtype, device=device)
    
    # Create some distinct vectors
    input_tensor[0, :, 0, 0] = torch.tensor([1, 0, 0], dtype=dtype, device=device) # Red
    input_tensor[0, :, 0, input_width-1] = torch.tensor([0, 1, 0], dtype=dtype, device=device) # Green
    input_tensor[0, :, input_height-1, 0] = torch.tensor([0, 0, 1], dtype=dtype, device=device) # Blue
    input_tensor[0, :, input_height-1, input_width-1] = torch.tensor([1, 1, 0], dtype=dtype, device=device) # Yellow
    
    # Add some non-unit norm vectors
    input_tensor[0, :, 1, 1] = torch.tensor([2, 0, 0], dtype=dtype, device=device)
    input_tensor[0, :, 1, 2] = torch.tensor([0, 3, 3], dtype=dtype, device=device) # Non-unit norm
    input_tensor[0, :, 2, 1] = torch.tensor([0, 0, 0], dtype=dtype, device=device) # Zero vector
    input_tensor[0, :, 2, 2] = torch.tensor([-1, 0, 0], dtype=dtype, device=device) # Opposite to [1,0,0]


    print(f"\nInput tensor shape: {input_tensor.shape}")
    # print("Input tensor (first item, all channels):")
    # print(input_tensor[0])

    # --- Perform bislerp ---
    output_tensor = bislerp(input_tensor, output_width, output_height)

    # --- Check output ---
    print(f"\nOutput tensor shape: {output_tensor.shape}")
    # print("Output tensor (first item, all channels, first 4x4 corner):")
    # print(output_tensor[0, :, :4, :4])


    expected_shape = (batch_size, channels, output_height, output_width)
    assert output_tensor.shape == expected_shape, \
        f"Output shape mismatch! Expected {expected_shape}, got {output_tensor.shape}"
    assert output_tensor.dtype == dtype, \
        f"Output dtype mismatch! Expected {dtype}, got {output_tensor.dtype}"
    assert output_tensor.device == input_tensor.device, \
        f"Output device mismatch! Expected {input_tensor.device}, got {output_tensor.device}"

    # Check for NaNs or Infs, which might occur if slerp is unstable
    if torch.isnan(output_tensor).any():
        print("\nWARNING: Output tensor contains NaNs!")
    if torch.isinf(output_tensor).any():
        print("\nWARNING: Output tensor contains Infs!")
        
    print("\nBislerp executed.")

    # Optional: Compare with standard bilinear interpolation for context
    # Note: This bislerp is NOT the same as standard F.interpolate bilinear.
    try:
        # Standard bilinear uses align_corners=False by default for 'bilinear' mode if not PyTorch 1.10+
        # For consistency with generate_bilinear_data in this script (which uses align_corners=False)
        output_bilinear_pytorch = F.interpolate(
            input_tensor, # Already float32 for this example
            size=(output_height, output_width),
            mode='bilinear',
            align_corners=False # To match the coordinate generation
        ).to(dtype)

        print(f"\nStandard PyTorch Bilinear (align_corners=False) output shape: {output_bilinear_pytorch.shape}")
        
        diff = torch.abs(output_tensor - output_bilinear_pytorch).mean()
        print(f"Mean Absolute Difference between this bislerp and PyTorch bilinear (align_corners=False): {diff.item():.6f}")
        # This difference is expected to be non-zero.

    except Exception as e:
        print(f"\nCould not perform standard bilinear comparison: {e}")