import torch
import torch.nn.functional as F

class RegionMergingNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "max_stack_size": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "display": "slider"
                }),
                "min_region_size": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "display": "slider"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "execute"
    CATEGORY = "😎 SnJake/PixelArt"

    def execute(self, image, max_stack_size, min_region_size):
        batch_size, height, width, channels = image.shape

        # Ensure image is in the correct range
        image = image.clamp(0, 1)

        # Permute the image to (batch_size, channels, height, width)
        image = image.permute(0, 3, 1, 2)

        output_images = []
        for b in range(batch_size):
            img = image[b]  # Shape: (channels, height, width)

            # Reshape to (height * width, channels)
            pixels = img.view(channels, -1).permute(1, 0)

            # Map unique colors to labels
            colors, inverse_indices = torch.unique(pixels, dim=0, return_inverse=True)

            # Create label image
            label_image = inverse_indices.view(height, width)

            # Initialize label IDs
            label_id = 1
            label_ids = torch.zeros_like(label_image, dtype=torch.int32)

            # Stack to keep track of pixels to visit
            visited = torch.zeros_like(label_image, dtype=torch.bool)

            # Direction vectors for 4-connectivity
            directions = torch.tensor([[0, -1], [-1, 0], [0, 1], [1, 0]], device=img.device)

            # Dictionary to hold region information
            region_sizes = {}
            region_colors = {}
            
            for y in range(height):
                for x in range(width):
                    if not visited[y, x]:
                        # Start a new region
                        current_label = label_id
                        color_label = label_image[y, x]
                        stack = [(y, x)]
                        visited[y, x] = True
                        region_coords = []
                        while stack:
                            cy, cx = stack.pop()
                            region_coords.append((cy, cx))
                            # Check neighbors
                            for dy, dx in directions:
                                ny, nx = cy + dy.item(), cx + dx.item()
                                if 0 <= ny < height and 0 <= nx < width:
                                    if not visited[ny, nx] and label_image[ny, nx] == color_label:
                                        visited[ny, nx] = True
                                        stack.append((ny, nx))
                        # Assign label to region
                        for ry, rx in region_coords:
                            label_ids[ry, rx] = current_label
                        region_sizes[current_label] = len(region_coords)
                        region_colors[current_label] = color_label
                        label_id += 1

            # Now, for each small region, merge it with neighboring large regions
            merged = False
            for current_label in range(1, label_id):
                region_size = region_sizes[current_label]
                if region_size <= max_stack_size:
                    # Get region mask
                    region_mask = (label_ids == current_label)
                    # Find neighbors
                    dilated_mask = F.max_pool2d(region_mask.float().unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)[0, 0]
                    border_mask = (dilated_mask > 0) & (~region_mask)
                    neighbor_labels = label_ids[border_mask]
                    neighbor_labels = neighbor_labels[neighbor_labels != 0]
                    neighbor_labels = neighbor_labels.unique()
                    # Filter neighbors by min_region_size
                    valid_neighbors = []
                    for neighbor_label in neighbor_labels:
                        neighbor_size = region_sizes[neighbor_label.item()]
                        if neighbor_size >= min_region_size:
                            valid_neighbors.append(neighbor_label)
                    if valid_neighbors:
                        # Merge into the largest neighbor
                        largest_neighbor = max(valid_neighbors, key=lambda l: region_sizes[l.item()])
                        label_ids[region_mask] = largest_neighbor
                        region_sizes[largest_neighbor.item()] += region_sizes[current_label]
                        region_sizes[current_label] = 0
                        merged = True

            # Map labels back to colors
            output_img = torch.zeros_like(img)
            for label in torch.unique(label_ids):
                if label == 0:
                    continue
                mask = (label_ids == label)
                color_label = region_colors[label.item()]
                color = colors[color_label]
                for c in range(channels):
                    output_img[c][mask] = color[c]

            output_images.append(output_img.unsqueeze(0))

        # Stack output images
        output_images = torch.cat(output_images, dim=0)

        # Permute back to (batch_size, height, width, channels)
        output_images = output_images.permute(0, 2, 3, 1)

        return (output_images,)
