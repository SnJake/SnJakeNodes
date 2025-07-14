import json
from collections import Counter
from safetensors import safe_open
import folder_paths # Импорт для работы с путями в ComfyUI

class LoraMetadataParser:
    """
    Узел для извлечения, подсчета и отображения тегов из метаданных файла LoRA.
    Он находит информацию о частоте использования тегов, агрегирует данные
    и возвращает отформатированную строку с тегами и их количеством,
    отсортированную по убыванию.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Определяет входные параметры для узла.
        В данном случае это выпадающий список со всеми найденными LoRA-файлами.
        """
        return {
            "required": {
                # Создаем выпадающий список (COMBO) из файлов в папке 'loras'
                "lora_name": (folder_paths.get_filename_list("loras"), ),
            }
        }

    # Тип возвращаемых данных - строка
    RETURN_TYPES = ("STRING",)
    # Имя выходного сокета
    RETURN_NAMES = ("tags_with_count",)
    # Имя функции, которая будет выполняться
    FUNCTION = "get_lora_tags"

    # Категория, в которой узел будет отображаться в меню ComfyUI
    CATEGORY = "😎 SnJake/Utils"

def get_lora_tags(self, lora_name: str):
    """
    Основная функция узла.
    Читает метаданные LoRA, извлекает и форматирует частоту тегов.
    Теги со специальными символами (например, \arknights) будут экранированы.
    """
    # Получаем полный путь к файлу LoRA
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path:
        return (f"Ошибка: Файл LoRA не найден: {lora_name}",)

    try:
        # Безопасно открываем файл .safetensors на CPU, чтобы не занимать VRAM
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
    except Exception as e:
        return (f"Ошибка при чтении файла .safetensors: {e}",)

    if metadata is None:
        return ("Метаданные не найдены в файле LoRA.",)

    # Извлекаем строку с частотой тегов
    tag_freq_str = metadata.get("ss_tag_frequency")
    if not tag_freq_str:
        return ("Ключ 'ss_tag_frequency' не найден в метаданных.",)

    try:
        # Преобразуем JSON-строку в объект Python
        tag_freq_data = json.loads(tag_freq_str)
    except json.JSONDecodeError:
        return ("Ошибка: Не удалось разобрать JSON из 'ss_tag_frequency'.",)

    # Используем Counter для удобного подсчета и суммирования тегов
    all_tags = Counter()
    
    for _dataset_key, tags_dict in tag_freq_data.items():
        if isinstance(tags_dict, dict):
             all_tags.update(tags_dict)

    if not all_tags:
        return ("Теги не найдены в 'ss_tag_frequency'.",)

    output_lines = []
    # Сортируем теги по количеству (по убыванию)
    for tag, count in all_tags.most_common():
        # Сначала проводим базовую очистку, как и раньше
        clean_tag = tag.replace('(', '').replace(')', '').strip()

        # ----- НОВАЯ ЛОГИКА ДЛЯ ЭКРАНИРОВАНИЯ -----
        # Проверяем, есть ли в теге обратный слэш
        if '\\' in clean_tag:
            words = clean_tag.split()
            # Обрабатываем каждое слово: если оно начинается с '\', оборачиваем его
            processed_words = [f'\\({word[1:]}\\)' if word.startswith('\\') else word for word in words]
            final_tag = ' '.join(processed_words)
        else:
            # Если слэша нет, оставляем тег как есть
            final_tag = clean_tag
        # -----------------------------------------
        
        output_lines.append(f"{final_tag}: {count}")
    
    output_string = "\n".join(output_lines)

    return (output_string,)
