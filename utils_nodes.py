import os
import glob
import random
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import numpy as np
import torch
import json

from pathlib import Path

from comfy_execution.graph import ExecutionBlocker
from comfy_execution.graph_utils import GraphBuilder

class BatchLoadImages:
    """
    Узел для пакетной загрузки изображений из папки без кеширования.
    Параметры и логика индексации/циклической выдачи сохранены,
    но кэширования (cached_paths) больше нет.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["single_image", "incremental_image", "random"], {}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32-1}),
                "index": ("INT", {"default": 0, "min": 0, "max": 150000}),
                "label": ("STRING", {"default": "Batch 001"}),
                "path": ("STRING", {"default": ""}),
                "pattern": ("STRING", {"default": "*"}),
                "allow_RGBA_output": (["false", "true"], {"default": "false"}),

                # <-- переключатель "по кругу"
                "allow_cycle": (["true", "false"], {"default":"true", "label_on":"Cycle On", "label_off":"Cycle Off"}),
            },
            "optional": {
                "filename_text_extension": (["true", "false"], {"default":"true"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "filename_text")
    FUNCTION = "load_batch_images"
    CATEGORY = "😎 SnJake/Utils"

    # ----------------------------------------------------------
    # Счётчики для 'incremental_image' (label -> index)
    # ----------------------------------------------------------
    incremental_counters = {}

    def load_batch_images(
        self, 
        path, 
        pattern="*", 
        index=0, 
        mode="single_image",
        seed=0, 
        label="Batch 001", 
        allow_RGBA_output="false", 
        filename_text_extension="true",
        allow_cycle="true"
    ):
        # 1) Собираем список картинок БЕЗ кэширования
        all_files = self._scan_directory(path, pattern)
        if not all_files:
            print(f"[BatchLoadImages] Папка '{path}' пуста или нет подходящих файлов по паттерну '{pattern}'")
            return (None, None)

        # 2) Логика выбора индекса
        if mode == "single_image":
            if index < 0 or index >= len(all_files):
                print(f"[BatchLoadImages] Запрошен index={index}, но в папке только {len(all_files)} файлов.")
                return (None, None)
            chosen_index = index

        elif mode == "incremental_image":
            if label not in self.incremental_counters:
                self.incremental_counters[label] = 0
            chosen_index = self.incremental_counters[label]

            # Если достигли конца
            if chosen_index >= len(all_files):
                if allow_cycle == "true":
                    # «По кругу»: сбрасываем в 0
                    print(f"[BatchLoadImages] Для label='{label}' индекс достиг конца ({chosen_index}). Сбрасываем в 0 (cycling).")
                    chosen_index = 0
                    self.incremental_counters[label] = 0
                else:
                    # Выходим с ошибкой
                    print(f"[BatchLoadImages] Изображения в папке закончились для label='{label}'. Останавливаемся.")
                    return (None, None)

            # Готовим индекс на след. раз
            self.incremental_counters[label] += 1

        else:  # mode == 'random'
            random.seed(seed)
            chosen_index = random.randint(0, len(all_files) - 1)

        # 3) Открываем картинку
        img_path = all_files[chosen_index]
        image_tensor = self._load_as_tensor(img_path, allow_RGBA_output == "true")

        # 4) Если нужно убрать расширение у filename
        filename = os.path.basename(img_path)
        if filename_text_extension == "false":
            filename = os.path.splitext(filename)[0]

        return (image_tensor, filename)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Если mode != single_image, возвращаем NaN, чтобы ComfyUI всегда перезапрашивал
        (иначе "random"/"incremental" могут не обновиться).
        """
        if kwargs["mode"] != "single_image":
            return float("NaN")
        else:
            path    = kwargs["path"]
            index   = kwargs["index"]
            pattern = kwargs["pattern"]
            mode    = kwargs["mode"]
            # Для single_image достаточно триггерить пересчёт, когда что-то меняется
            return (path, pattern, mode, index)

    # --------------------------------------------------------------------
    # Служебные методы
    # --------------------------------------------------------------------
    def _scan_directory(self, directory_path, pattern):
        exts = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tiff"]
        files = []
        for file_name in glob.glob(os.path.join(directory_path, pattern), recursive=True):
            if os.path.splitext(file_name)[1].lower() in exts:
                files.append(os.path.abspath(file_name))
        files.sort()
        return files

    def _load_as_tensor(self, file_path, allow_rgba=False):
        from PIL import Image, ImageOps
        import numpy as np
        import torch

        pil_img = Image.open(file_path)
        pil_img = ImageOps.exif_transpose(pil_img)

        # Приводим к RGB, если не разрешён RGBA
        if not allow_rgba and pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        elif allow_rgba and pil_img.mode == "RGBA":
            # Оставляем RGBA, если пользователь разрешил
            pass
        else:
            # Если вдруг формат P, LA и т.д., приводим к RGB
            if pil_img.mode not in ["RGB", "RGBA"]:
                pil_img = pil_img.convert("RGB")

        np_img = np.array(pil_img).astype(np.float32) / 255.0
        # batch dimension
        tensor = torch.from_numpy(np_img)[None, ]
        return tensor





class LoadSingleImageFromPath:
    """
    Узел для загрузки ОДНОГО изображения по ПОЛНОМУ пути, включая имя и формат.
    Пример входа:  /home/user/images/test.png
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"
    CATEGORY = "😎 SnJake/Utils"

    def load_image(self, image_path):
        if not os.path.exists(image_path):
            print(f"[LoadSingleImageFromPath] Файл '{image_path}' не найден!")
            return (None,)

        pil_img = Image.open(image_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        # Переводим всё к RGB на всякий случай
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        np_img = np.array(pil_img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_img)[None,]
        return (tensor,)




class SaveImageToPath:
    """
    Узел для сохранения полученного изображения в указанный путь.
    Пример полного пути: D:\Stable diffusion\result_7.png
    Если директории не существует, она будет создана автоматически.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "save_path": ("STRING", {"default": "D:\\Stable diffusion\\result_7.png"}),
                "save_workflow": ("BOOLEAN", {"default": True, "tooltip": "Сохранять ли workflow (метаданные) внутри изображения."}), # Новый параметр
            },
            "hidden": { # Скрытые входы для получения информации о workflow от ComfyUI
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_image"
    CATEGORY = "😎 SnJake/Utils"
    OUTPUT_NODE = True

    def save_image(self, image, save_path, save_workflow, prompt=None, extra_pnginfo=None):
        if image is None:
            print("[SaveImageToPath] Нет входного изображения!")
            return ()

        # Используем pathlib для нормализации пути
        try:
            path_obj = Path(save_path.strip().strip('"').strip("'"))
            # Если требуется, можно преобразовать путь к абсолютному
            full_path = path_obj.resolve()
            print(f"[SaveImageToPath] Полный путь: {full_path}")
        except Exception as e:
            print(f"[SaveImageToPath] Ошибка при обработке пути: {e}")
            return ()

        # Создаем директорию, если не существует
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[SaveImageToPath] Ошибка при создании директории {full_path.parent}: {e}")
            return ()

        # Преобразуем torch.Tensor -> numpy -> PIL Image
        try:
            # Берем первое изображение из батча, если их несколько
            # ComfyUI обычно передает батч изображений, даже если он состоит из одного элемента.
            # Индексация [0] предполагает, что мы сохраняем только первое изображение из батча,
            # или что узел предназначен для работы с одним изображением за раз.
            # Если нужно сохранять все изображения из батча, логику нужно будет изменить (например, цикл и модификация имени файла).
            # Для данного примера, предполагаем сохранение одного изображения.
            img_tensor = image[0] 
            np_img = img_tensor.cpu().numpy()
            
            # Транспонирование, если формат [C,H,W]
            if len(np_img.shape) == 3 and np_img.shape[0] in [1, 3, 4]: # (C, H, W)
                np_img = np.transpose(np_img, (1, 2, 0)) # (H, W, C)
            
            np_img = (np_img * 255.0).clip(0, 255).astype(np.uint8)
            
            # Определяем режим в зависимости от числа каналов
            if np_img.ndim == 2: # Grayscale
                mode = "L"
            elif np_img.shape[2] == 3: # RGB
                mode = "RGB"
            elif np_img.shape[2] == 4: # RGBA
                mode = "RGBA"
            elif np_img.shape[2] == 1: # Одноканальное, но не L (может быть маска)
                np_img = np_img.squeeze(axis=2) # Удаляем последнюю размерность
                mode = "L" # Сохраняем как Grayscale
            else:
                print(f"[SaveImageToPath] Неподдерживаемое количество каналов: {np_img.shape[2]}")
                return ()
                
            pil_img = Image.fromarray(np_img, mode=mode)
        except Exception as e:
            print(f"[SaveImageToPath] Ошибка при конвертации изображения: {e}")
            return ()

        # Подготовка метаданных для сохранения workflow
        metadata_to_save = None
        if save_workflow:
            metadata_to_save = PngInfo()
            if prompt is not None:
                metadata_to_save.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None and isinstance(extra_pnginfo, dict):
                for k, v in extra_pnginfo.items():
                    metadata_to_save.add_text(k, json.dumps(v))
        
        # Сохраняем изображение
        try:
            pil_img.save(str(full_path), pnginfo=metadata_to_save)
            print(f"[SaveImageToPath] Изображение успешно сохранено: {full_path}")
            if save_workflow:
                if prompt or extra_pnginfo:
                     print(f"[SaveImageToPath] Workflow data has been included in the image.")
                else:
                     print(f"[SaveImageToPath] Workflow saving was enabled, but no workflow data (prompt/extra_pnginfo) was available to save.")
            else:
                print(f"[SaveImageToPath] Workflow data was not saved to the image (option disabled).")
        except Exception as e:
            print(f"[SaveImageToPath] Ошибка при сохранении изображения: {e}")

        return ()



class ImageRouter:
    CATEGORY = "😎 SnJake/Utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,  # Максимальное количество выходов
                    "step": 1,
                    "tooltip": "Выберите номер выхода для перенаправления изображения."
                }),
                "image_in": ("IMAGE", {
                    "tooltip": "Входящее изображение для перенаправления."
                }),
            },
            "optional": {
                "max_outputs": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 10,  # Максимальное количество выходов не должно превышать RETURN_TYPES
                    "step": 1,
                    "tooltip": "Максимальное количество выходов. Не должно превышать 10."
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = tuple(["IMAGE"] * 10)  # Определяем 10 выходов типа IMAGE
    RETURN_NAMES = tuple([f"output{i+1}" for i in range(10)])  # Имена выходов: output1, output2, ..., output10
    FUNCTION = "switch_image"

    def switch_image(self, select, image_in, max_outputs=10, unique_id=None, prompt=None, **kwargs):
        """
        Перенаправляет входящее изображение в выбранный выход.
        Остальные выходы блокируются.
        """
        # Ограничиваем значение max_outputs до 10
        max_outputs = max(1, min(max_outputs, 10))

        # Проверяем, что значение select находится в допустимом диапазоне
        if not 1 <= select <= max_outputs:
            raise ValueError(f"Значение 'select' ({select}) должно быть в диапазоне от 1 до {max_outputs}.")

        outputs = []
        for i in range(1, max_outputs + 1):
            if i == select:
                outputs.append(image_in)
            else:
                # Блокируем остальные выходы
                outputs.append(ExecutionBlocker(None))

        # Заполняем оставшиеся выходы значением None, если max_outputs меньше 10
        while len(outputs) < 10:
            outputs.append(None)

        return tuple(outputs)





class StringToNumber:
    """
    Узел, который берёт строку (STRING) и пробует сконвертировать в int и float.
    На выходе два значения: int_value, float_value
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {"default": "123"})
            }
        }

    RETURN_TYPES = ("INT","FLOAT")
    RETURN_NAMES = ("as_int","as_float")
    FUNCTION = "convert"
    CATEGORY = "😎 SnJake/Utils"

    def convert(self, input_string):
        try:
            i_val = int(input_string)
        except:
            i_val = 0
            print(f"[StringToNumber] Не удалось преобразовать '{input_string}' к int. Ставим 0.")

        try:
            f_val = float(input_string)
        except:
            f_val = 0.0
            print(f"[StringToNumber] Не удалось преобразовать '{input_string}' к float. Ставим 0.0.")

        return (i_val, f_val)




class StringReplace:
    """
    Узел, который заменяет в исходном тексте (source_string) подстроку (old_string)
    на (new_string). Пример:
      source_string = "Hello"
      old_string    = "ell"
      new_string    = "bob"
      => "Hbobo"
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_string": ("STRING", {"default": "Hello"}),
                "old_string":    ("STRING", {"default": "ell"}),
                "new_string":    ("STRING", {"default": "bob"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("replaced_string",)
    FUNCTION = "string_replace"
    CATEGORY = "😎 SnJake/Utils"

    def string_replace(self, source_string, old_string, new_string):
        if source_string is None:
            return ("",)
        result = source_string.replace(old_string, new_string)
        return (result,)





class RandomIntNode:
    CATEGORY = "😎 SnJake/Utils"
    FUNCTION = "generate"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("random_int",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_value": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1, "tooltip": "Минимальное значение"}),
                "max_value": ("INT", {"default": 10, "min": -10000, "max": 10000, "step": 1, "tooltip": "Максимальное значение"})
            }
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Возвращает NaN, чтобы узел всегда пересчитывался и не использовался кэш
        return float("NaN")

    def generate(self, min_value, max_value):
        result = random.randint(min_value, max_value)
        return (result,)


class RandomFloatNode:
    CATEGORY = "😎 SnJake/Utils"
    FUNCTION = "generate"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("random_float",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_value": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01, "tooltip": "Минимальное значение"}),
                "max_value": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01, "tooltip": "Максимальное значение"})
            }
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Отключаем кэширование, возвращая NaN
        return float("NaN")

    def generate(self, min_value, max_value):
        value = random.uniform(min_value, max_value)
        # Округляем до 2 знаков после запятой, чтобы, например, 0.53228 превратилось в 0.53
        result = round(value, 2)
        return (result,)