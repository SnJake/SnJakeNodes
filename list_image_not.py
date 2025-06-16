import os

class ScanImageFolder2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("filenames", "image_paths")
    CATEGORY = "😎 SnJake/Utils"
    FUNCTION = "scan_folder"

    def scan_folder(self, folder_path):
        image_extensions = (".jpg", ".jpeg", ".png", ".webp", ".tif")
        filenames = []
        image_paths = []

        if not os.path.isdir(folder_path):
            return ["Error: указанный путь не является папкой."], ["Error: указанный путь не является папкой."]

        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(image_extensions):
                full_path = os.path.join(folder_path, filename)
                filenames.append(filename)
                image_paths.append(full_path)

        return "\n".join(filenames), "\n".join(image_paths)