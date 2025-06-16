import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class ColorPaletteImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color_palette_str": ("STRING", {
                    "default": "",
                    "widget_type": "text_box",
                    "multiline": False,
                    "placeholder": "Enter HEX colors separated by commas"
                }),
                "square_size": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 500,
                    "step": 1,
                    "display": "slider"
                }),
                "columns": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                "background_color": ("STRING", {"default": "#FFFFFF"}),
                "show_color_codes": ("BOOLEAN", {"default": True}),
                "font_size": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("palette_image",)
    FUNCTION = "execute"
    CATEGORY = "😎 SnJake/PixelArt"

    def execute(self, color_palette_str, square_size, columns, background_color, show_color_codes, font_size):
        # Parse the color palette string into a list of HEX codes
        colors = [c.strip() for c in color_palette_str.strip().split(',') if c.strip()]
        num_colors = len(colors)

        if num_colors == 0:
            # Return a blank image or raise an error
            raise ValueError("No colors provided in color_palette_str.")

        # Calculate the number of rows needed based on the number of columns
        rows = (num_colors + columns - 1) // columns

        # Create a new image with the calculated size
        canvas_width = columns * square_size
        canvas_height = rows * square_size
        image = Image.new("RGB", (canvas_width, canvas_height), background_color)
        draw = ImageDraw.Draw(image)

        # Load a font for drawing text
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # Draw each color as a square
        for idx, color in enumerate(colors):
            row = idx // columns
            col = idx % columns
            x0 = col * square_size
            y0 = row * square_size
            x1 = x0 + square_size
            y1 = y0 + square_size

            # Draw the color square
            draw.rectangle([x0, y0, x1, y1], fill=color)

            # Optionally add the color code text
            if show_color_codes:
                text = color.upper()
                # Use draw.textbbox to get text dimensions
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_x = x0 + (square_size - text_width) / 2
                text_y = y0 + (square_size - text_height) / 2

                # Add a contrasting outline around the text for readability
                outline_color = "white" if self._is_dark_color(color) else "black"
                self._draw_text_with_outline(draw, text_x, text_y, text, font, fill=outline_color)

        # Convert the PIL image to a tensor
        image_tensor = self._pil_to_tensor(image)

        return (image_tensor,)

    def _pil_to_tensor(self, image):
        # Convert PIL image to a torch tensor in ComfyUI format (batch_size, height, width, channels)
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)
        return image

    def _is_dark_color(self, hex_color):
        # Determine if a color is dark based on its luminance
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2 ,4))
        luminance = (0.299*r + 0.587*g + 0.114*b) / 255
        return luminance < 0.5

    def _draw_text_with_outline(self, draw, x, y, text, font, fill):
        # Draw text with an outline for better visibility
        outline_color = "black" if fill == "white" else "white"
        outline_width = 1
        # Draw outline
        for adj in [-outline_width, 0, outline_width]:
            for adj2 in [-outline_width, 0, outline_width]:
                if adj != 0 or adj2 != 0:
                    draw.text((x + adj, y + adj2), text, font=font, fill=outline_color)
        # Draw text
        draw.text((x, y), text, font=font, fill=fill)