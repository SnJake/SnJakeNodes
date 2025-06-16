import os
import re
from PIL import Image
import numpy as np
import torch

def load_image(image_path):
    try:
        i = Image.open(image_path)
        i = i.convert("RGB")
        image = np.array(i).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return image
    except Exception as e:
        print(f"Error loading image: {image_path}")
        print(e)
        return None

class ConcatenateImagesByDirectory:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_paths": ("STRING", {"multiline": True, "forceInput": True}),
                # Желательно, чтобы пользователь вводил базовое имя без расширения,
                # например "0" или "asd". Если будет введён полный путь или имя с расширением,
                # код обработает его.
                "base_image_name": ("STRING", {"default": "0", "placeholder": "например, 0 или asd"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "concatenate_images"
    CATEGORY = "😎 SnJake/Utils"
    INPUT_IS_LIST = True

    def concatenate_images(self, image_paths, base_image_name):
        # Если base_image_name является списком, берем первый элемент
        if isinstance(base_image_name, list):
            base_image_name = base_image_name[0]
        # Если base_image_name содержит путь или расширение, извлекаем только базовое имя
        base_image_name = os.path.splitext(os.path.basename(base_image_name))[0]
        
        # Группируем пути по директориям
        image_groups = {}
        for path in image_paths:
            path = os.path.normpath(path)
            directory = os.path.dirname(path)
            if directory not in image_groups:
                image_groups[directory] = []
            image_groups[directory].append(path)

        final_images = []

        # Обрабатываем каждую директорию
        for directory, paths in image_groups.items():
            base_image_path = None
            other_images = []

            # Формируем ожидаемые имена базового изображения (ищем jpg и png)
            expected_names = [
                f"{base_image_name.lower()}.jpg",
                f"{base_image_name.lower()}.png"
            ]

            # Поиск базового изображения в директории
            try:
                directory_files = os.listdir(directory)
            except Exception as e:
                print(f"Warning: не удалось прочитать директорию: {directory}")
                continue

            for file_name in directory_files:
                if file_name.lower() in expected_names:
                    base_image_path = os.path.join(directory, file_name)
                    break

            if base_image_path is None:
                print(f"Warning: No base image '{expected_names[0]}' or '{expected_names[1]}' found in directory: {directory}")
                continue

            # Определяем, какие изображения являются дополнительными (кроме базового)
            for path in paths:
                if os.path.normpath(path) == os.path.normpath(base_image_path):
                    continue
                else:
                    other_images.append(path)

            # Сортировка дополнительных изображений по числовой части имени файла
            def get_numeric_part(path):
                match = re.search(r"(\d+)\.(jpg|png|jpeg|webp|bmp|gif)$", path, re.IGNORECASE)
                return int(match.group(1)) if match else float('inf')

            other_images.sort(key=get_numeric_part)

            # Загружаем базовое изображение
            base_image = load_image(base_image_path)
            if base_image is None:
                print(f"Warning: could not load base image: {base_image_path}")
                continue

            # Если дополнительных изображений нет, можно вернуть базовое изображение (по необходимости)
            if len(other_images) == 0:
                final_images.append(base_image)
            else:
                # Конкатенируем каждое дополнительное изображение с базовым (по горизонтали)
                for path in other_images:
                    other_image = load_image(path)
                    if other_image is not None:
                        concatenated_image = torch.cat((base_image, other_image), dim=2)
                        final_images.append(concatenated_image)

        # Если итоговых изображений нет, возвращаем пустой список, чтобы избежать дальнейших ошибок
        return (final_images,)
