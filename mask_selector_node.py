import torch
from server import PromptServer
from comfy_execution.graph import ExecutionBlocker

class ImageMaskSelector:
    """
    This node takes an image and a mask as input.
    It checks if the mask contains any non-zero values (gray or white).
    - If the mask has values greater than 0, the image is sent to output 1 and the mask is sent to a separate output.
    - If the mask contains only 0 (black), the image is sent to output 2.
    """

    CATEGORY = "😎 SnJake/Utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "mask": ("MASK", {}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("output_1", "output_2", "mask_out")
    FUNCTION = "process"

    def process(self, image, mask):
        """
        Processes the image and mask to determine the output based on mask content.

        Args:
            image: Input image tensor.
            mask: Input mask tensor.

        Returns:
            A tuple containing the output image(s) and mask based on the mask's content.
        """

        # Check if the mask contains any non-zero values.
        non_zero_pixels = torch.any(mask > 0)

        if non_zero_pixels:
            # If the mask has values greater than 0, send the image to output 1 and the mask to mask_out.
            # Block output 2 using ExecutionBlocker.
            return (image, ExecutionBlocker(None), mask)
        else:
            # If the mask contains only 0, send the image to output 2.
            # Block output 1 using ExecutionBlocker.
            return (ExecutionBlocker(None), image, ExecutionBlocker(None))