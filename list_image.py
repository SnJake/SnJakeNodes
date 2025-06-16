import os

class ScanImageFolder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("image_paths",)
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "😎 SnJake/Utils"
    FUNCTION = "scan_folder"

    def scan_folder(self, folder_path):
        image_extensions = (".jpg", ".jpeg", ".png", ".webp", ".tif")
        image_paths = []

        if not os.path.isdir(folder_path):
            print("Error: specified path is not a folder.")
            return [""]

        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(image_extensions):
                full_path = os.path.join(folder_path, filename)
                image_paths.append(full_path)

        return (image_paths, )