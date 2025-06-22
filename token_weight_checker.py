import torch
import re

class TokenWeightChecker:
    """
    Нода для проверки "веса" токенов в модели CLIP.
    Принимает на вход CLIP, а не MODEL.
    Под "весом" понимается L2-норма (величина) вектора эмбеддинга токена.
    Более высокое значение может указывать на то, что токен несет больше "изученной" информации.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ИЗМЕНЕНО: Теперь нода напрямую принимает CLIP
                "clip": ("CLIP",),
                "tokens": ("STRING", {
                    "multiline": True,
                    "default": "cat, dog, astronaut, a beautiful landscape"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("weights_string",)
    FUNCTION = "check_token_weights"
    CATEGORY = "😎 SnJake/Utils"

    # ИЗМЕНЕНО: Функция теперь принимает 'clip' вместо 'model'
    def check_token_weights(self, clip, tokens):
        output_lines = []
        
        # Получаем токенизатор и эмбеддинги напрямую из объекта CLIP
        try:
            tokenizer = clip.tokenizer
            embedding_layer = clip.transformer.text_model.embeddings.token_embedding
            embeddings = embedding_layer.weight
        except AttributeError:
            return ("Ошибка: структура CLIP-модели не соответствует ожидаемой. Не удалось найти токенизатор или эмбеддинги.",)

        # Обрабатываем введенные токены: разделяем по запятым и новым строкам
        token_list = [t.strip() for t in re.split('[,\\n]', tokens) if t.strip()]

        if not token_list:
            return ("Нет токенов для проверки. Введите токены в текстовое поле.",)

        output_lines.append(f"Проверка весов для {len(token_list)} токенов:")
        output_lines.append("-" * 20)

        for token_str in token_list:
            # Токенизируем строку
            # .encode() добавляет токены начала и конца строки (e.g., [49406, ID, 49407] для SD1.5)
            token_ids = tokenizer.encode(token_str)

            # Проверяем, состоит ли ввод из одного "значимого" токена (исключая start/end)
            # Для SDXL может быть больше служебных токенов
            meaningful_tokens = token_ids[1:-1]
            if not meaningful_tokens:
                 output_lines.append(f"'{token_str}': [Не удалось токенизировать]")
                 continue

            if len(meaningful_tokens) > 1:
                # Если строка разбивается на несколько токенов, обрабатываем каждый
                sub_tokens_info = []
                for token_id in meaningful_tokens:
                    # Убедимся, что ID в пределах словаря
                    if token_id >= len(embeddings):
                        sub_tokens_info.append(f"ID {token_id} [вне словаря]")
                        continue
                    
                    sub_token_str = tokenizer.decode([token_id])
                    # Проверяем, что токен не является UNK (неизвестным)
                    if token_id == tokenizer.unk_token_id:
                        weight_info = "[Неизвестный токен]"
                    else:
                        token_vector = embeddings[token_id]
                        weight = torch.linalg.norm(token_vector).item()
                        weight_info = f"{weight:.4f}"
                    sub_tokens_info.append(f"'{sub_token_str}' ({weight_info})")
                
                output_lines.append(f"'{token_str}' -> {' '.join(sub_tokens_info)}")

            else:
                # Обработка одиночного токена
                token_id = meaningful_tokens[0]
                if token_id >= len(embeddings):
                    output_lines.append(f"'{token_str}': ID {token_id} [вне словаря]")
                    continue

                if token_id == tokenizer.unk_token_id:
                    output_lines.append(f"'{token_str}': [Неизвестный токен]")
                else:
                    token_vector = embeddings[token_id]
                    weight = torch.linalg.norm(token_vector).item()
                    output_lines.append(f"'{token_str}': {weight:.4f}")
        
        # Соединяем все строки в один текстовый блок
        result_string = "\n".join(output_lines)
        
        return (result_string,)
