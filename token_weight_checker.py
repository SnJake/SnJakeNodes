import torch
import re

class TokenWeightChecker:
    """
    Универсальная нода для проверки "веса" токенов в моделях CLIP.
    Автоматически определяет и обрабатывает одиночные и множественные
    текстовые энкодеры (например, в SDXL, SD3, Flux).
    Под "весом" понимается L2-норма (величина) вектора эмбеддинга токена.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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

    def _process_encoder(self, name, encoder, token_list):
        """Вспомогательная функция для обработки одного энкодера."""
        lines = [f"--- Анализ для: {name} ---"]
        
        try:
            # Получаем токенизатор и эмбеддинги
            tokenizer = encoder.tokenizer
            
            # У разных моделей разная структура
            if hasattr(encoder, 'transformer') and hasattr(encoder.transformer, 'text_model'): # Для OpenCLIP (clip-l, clip-g)
                embeddings = encoder.transformer.text_model.embeddings.token_embedding.weight
            elif hasattr(encoder, 'transformer') and hasattr(encoder.transformer, 'get_input_embeddings'): # Для T5
                embeddings = encoder.transformer.get_input_embeddings().weight
            else:
                lines.append("Не удалось найти таблицу эмбеддингов для этой модели.")
                return lines

            for token_str in token_list:
                token_ids = tokenizer.encode(token_str)
                meaningful_tokens = token_ids[1:-1] # Исключаем токены начала/конца

                if not meaningful_tokens:
                    lines.append(f"'{token_str}': [Не удалось токенизировать]")
                    continue

                if len(meaningful_tokens) > 1:
                    sub_tokens_info = []
                    for token_id in meaningful_tokens:
                        if token_id >= len(embeddings):
                            sub_tokens_info.append(f"id:{token_id}[вне словаря]")
                            continue
                        
                        sub_token_str = tokenizer.decode([token_id])
                        if token_id == getattr(tokenizer, 'unk_token_id', -1):
                            weight_info = "[UNK]"
                        else:
                            weight = torch.linalg.norm(embeddings[token_id]).item()
                            weight_info = f"{weight:.4f}"
                        sub_tokens_info.append(f"'{sub_token_str}'({weight_info})")
                    lines.append(f"'{token_str}' -> {' '.join(sub_tokens_info)}")
                else:
                    token_id = meaningful_tokens[0]
                    if token_id >= len(embeddings):
                        lines.append(f"'{token_str}': id:{token_id}[вне словаря]")
                        continue
                        
                    if token_id == getattr(tokenizer, 'unk_token_id', -1):
                        lines.append(f"'{token_str}': [Неизвестный токен]")
                    else:
                        weight = torch.linalg.norm(embeddings[token_id]).item()
                        lines.append(f"'{token_str}': {weight:.4f}")
            
        except Exception as e:
            lines.append(f"Ошибка при обработке энкодера '{name}': {e}")
            import traceback
            lines.append(traceback.format_exc())
            
        return lines

    def check_token_weights(self, clip, tokens):
        # Собираем все энкодеры, которые есть в объекте clip
        encoders_to_check = {}
        possible_encoder_names = ['clip_l', 'clip_g', 't5xxl']
        
        for name in possible_encoder_names:
            if hasattr(clip, name) and getattr(clip, name) is not None:
                encoders_to_check[name.upper()] = getattr(clip, name)
        
        # Если ни одного вложенного энкодера не найдено,
        # предполагаем, что сам 'clip' является энкодером (для SD 1.5)
        if not encoders_to_check:
            if hasattr(clip, 'tokenizer') and hasattr(clip, 'transformer'):
                 encoders_to_check['CLIP'] = clip
            else:
                return ("Ошибка: не удалось определить структуру CLIP. Это не стандартный CLIP-объект и не контейнер (SDXL/SD3).",)

        # Обрабатываем введенные токены
        token_list = [t.strip() for t in re.split('[,\\n]', tokens) if t.strip()]
        if not token_list:
            return ("Нет токенов для проверки.",)

        # Собираем результаты по всем энкодерам
        final_output = []
        for name, encoder in encoders_to_check.items():
            result_lines = self._process_encoder(name, encoder, token_list)
            final_output.extend(result_lines)
            final_output.append("") # Добавляем пустую строку для разделения

        return ("\n".join(final_output),)
