import numpy as np
from sklearn.cluster import KMeans

class ColorPaletteCompressionNode:
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
                "num_basic_colors": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "display": "slider"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("compressed_color_palette_str",)
    FUNCTION = "execute"
    CATEGORY = "😎 SnJake/PixelArt"

    def execute(self, color_palette_str, num_basic_colors):
        # Parse the color palette string into a list of HEX color codes
        colors = [c.strip() for c in color_palette_str.strip().split(',') if c.strip()]
        num_colors = len(colors)

        if num_colors == 0:
            raise ValueError("No colors provided in color_palette_str.")

        # Convert HEX colors to RGB
        color_values = []
        for hex_color in colors:
            hex_color = hex_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2 ,4))
            color_values.append([r, g, b])

        color_values = np.array(color_values)

        # Use KMeans clustering to reduce the number of colors
        n_clusters = min(num_basic_colors, num_colors)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(color_values)
        centroids = kmeans.cluster_centers_

        # Convert centroids to integer RGB values
        centroids = np.clip(centroids.round(), 0, 255).astype(int)

        # Convert centroids back to HEX codes
        compressed_colors = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in centroids]

        # Create the compressed color palette string
        compressed_color_palette_str = ', '.join(compressed_colors)

        return (compressed_color_palette_str,)
