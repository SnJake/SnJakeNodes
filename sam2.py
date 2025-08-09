import os
import sys
import torch
import numpy as np
from PIL import Image

# Добавляем директорию с нодой в системный путь для импорта 'sam2'
sys.path.append(os.path.dirname(__file__))

# Импорты из ComfyUI
import folder_paths
from comfy.utils import ProgressBar

# Импорты из SAM-2 и других библиотек
try:
    from sam2 import sam2_image_predictor
    from sam2 import build_sam
    from sam2.build_sam import build_sam2
    from safetensors.torch import load_file as load_safetensors
except ImportError as e:
    print(f"SnJakeNodes Error: Не удалось импортировать библиотеку SAM2. Убедитесь, что папка 'sam2' находится в директории ноды и зависимости установлены (hydra-core, omegaconf, safetensors). Ошибка: {e}")


# --- Глобальные переменные и вспомогательные функции ---

# Указываем ComfyUI, где искать модели SAM-2
SAM2_MODEL_DIR = os.path.join(folder_paths.models_dir, "sam2")
folder_paths.add_model_folder_path("sam2", SAM2_MODEL_DIR)

# Сопоставление имен файлов моделей с файлами конфигурации
MODEL_CONFIG_MAP = {
    "hiera_tiny": "sam2.1_hiera_t.yaml",
    "hiera_small": "sam2.1_hiera_s.yaml",
    "hiera_base_plus": "sam2.1_hiera_b+.yaml",
    "hiera_large": "sam2.1_hiera_l.yaml",
}

def get_sam2_model_names():
    """Сканирует директорию с моделями и возвращает список доступных чекпоинтов."""
    if not os.path.exists(SAM2_MODEL_DIR):
        os.makedirs(SAM2_MODEL_DIR)
        return []
    return [f for f in os.listdir(SAM2_MODEL_DIR) if f.endswith((".pt", ".safetensors"))]

def tensor_to_numpy_image(tensor):
    """Конвертирует тензор (B, H, W, C) в массив numpy (H, W, C) uint8."""
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    # Конвертация из float [0,1] в uint8 [0,255]
    img = tensor.cpu().numpy()[0]
    return np.clip(img * 255, 0, 255).astype(np.uint8)


# --- Реализация нод ---

class Sam2Loader:
    """Нода для загрузки модели SAM-2 и ее конфигурации."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_sam2_model_names(),),
                "device": (["cuda", "cpu"],),
            }
        }

    RETURN_TYPES = ("SAM2_MODEL",)
    RETURN_NAMES = ("sam2_model",)
    FUNCTION = "load_model"
    CATEGORY = "SnJake/SAM2"

    def load_model(self, model_name, device):
        ckpt_path = os.path.join(SAM2_MODEL_DIR, model_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Чекпоинт модели не найден: {ckpt_path}")

        # Поиск соответствующего файла конфигурации
        config_name = next((v for k, v in MODEL_CONFIG_MAP.items() if k in model_name), None)
        if config_name is None:
            raise ValueError(f"Не удалось найти конфигурацию для модели: {model_name}")

        config_file_path = os.path.join(os.path.dirname(__file__), "sam2", "configs", "sam2.1", config_name)
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_file_path}")

        print(f"SnJake SAM2: Загрузка модели '{model_name}' с конфигурацией '{config_name}'")

        # Загрузка state_dict в зависимости от типа файла
        if model_name.endswith(".safetensors"):
            sd = load_safetensors(ckpt_path, device="cpu")
        else:
            sd = torch.load(ckpt_path, map_location="cpu")

        model_sd = sd.get("model", sd)

        # --- Кастомная загрузка чекпоинта ---
        # Мы переопределяем внутреннюю функцию загрузки в build_sam,
        # чтобы использовать наш уже загруженный state_dict.
        original_load_checkpoint = build_sam._load_checkpoint
        def new_load_checkpoint(model, ckpt_path):
            missing_keys, unexpected_keys = model.load_state_dict(model_sd, strict=False)
            if missing_keys:
                print(f"SAM2 Warning (Пропущенные ключи): {missing_keys}")
            if unexpected_keys:
                print(f"SAM2 Warning (Неожиданные ключи): {unexpected_keys}")
            print("SnJake SAM2: State dict чекпоинта успешно загружен.")

        build_sam._load_checkpoint = new_load_checkpoint

        # Используем hydra для чтения конфигурации и инстанцирования модели
        from hydra import initialize_config_dir, compose
        from omegaconf import OmegaConf

        config_dir = os.path.dirname(config_file_path)
        with initialize_config_dir(config_dir=os.path.abspath(config_dir), version_base=None):
            cfg = compose(config_name=os.path.basename(config_file_path))
            OmegaConf.resolve(cfg)
            
            # Эта функция вызовет наш new_load_checkpoint
            sam_model = build_sam2(
                config_file=config_file_path,
                ckpt_path=ckpt_path, # формально передаем, но он не используется
                device=device,
                mode="eval",
            )
        
        # Восстанавливаем оригинальную функцию
        build_sam._load_checkpoint = original_load_checkpoint
        print("SnJake SAM2: Модель успешно загружена.")
        return (sam_model,)


class Sam2ImageInference:
    """Нода для выполнения сегментации на изображении с помощью SAM-2."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam2_model": ("SAM2_MODEL",),
                "image": ("IMAGE",),
                "positive_points": ("MASK",),
                "negative_points": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "multimask_output": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "predict"
    CATEGORY = "SnJake/SAM2"

    def predict(self, sam2_model, image, positive_points, negative_points, threshold, multimask_output):
        predictor = sam2_image_predictor.SAM2ImagePredictor(
            sam_model=sam2_model,
            mask_threshold=threshold,
        )
        
        img_np = tensor_to_numpy_image(image)
        
        print("SnJake SAM2: Установка изображения в предиктора...")
        predictor.set_image(img_np)
        
        # Обработка промптов (точек)
        pos_coords = (positive_points[0] > 0).nonzero(as_tuple=False).cpu().numpy()[:, [1, 0]]
        neg_coords = (negative_points[0] > 0).nonzero(as_tuple=False).cpu().numpy()[:, [1, 0]]

        if pos_coords.shape[0] == 0 and neg_coords.shape[0] == 0:
            print("SnJake SAM2 Warning: Точки для сегментации не предоставлены. Возвращается пустая маска.")
            h, w, _ = img_np.shape
            return (torch.zeros((1, h, w), dtype=torch.float32, device="cpu"),)

        point_coords = np.concatenate([pos_coords, neg_coords], axis=0) if neg_coords.size > 0 else pos_coords
        pos_labels = np.ones(pos_coords.shape[0], dtype=int)
        neg_labels = np.zeros(neg_coords.shape[0], dtype=int)
        point_labels = np.concatenate([pos_labels, neg_labels], axis=0)
        
        print(f"SnJake SAM2: Предсказание с {pos_coords.shape[0]} позитивными и {neg_coords.shape[0]} негативными точками...")
        masks_np, iou_preds, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output,
            return_logits=False, # возвращаем уже бинарные маски
        )

        # masks_np имеет форму (C, H, W), где C - количество предсказанных масок
        # Выбираем маску с наилучшим предсказанным IoU
        best_mask_idx = np.argmax(iou_preds)
        final_mask_np = masks_np[best_mask_idx]

        # Конвертируем маску в тензор float, как ожидает ComfyUI
        final_mask_tensor = torch.from_numpy(final_mask_np.astype(np.float32)).unsqueeze(0)
        
        return (final_mask_tensor,)
