import torch
import re
import traceback

class TokenWeightChecker:
    """
    Универсальная нода для проверки "веса" токенов.
    Динамически находит все текстовые энкодеры внутри объекта CLIP,
    что делает ее совместимой с SD1.5, SDXL, SD3, Flux и другими архитектурами.
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
        """Вспомогательная функция для анализа одного энкодера."""
        lines = [f"--- Анализ для: {name} ---"]
        
        try:
            tokenizer = getattr(encoder, 'tokenizer', None)
            if not tokenizer:
                lines.append("Ошибка: Токенизатор не найден.")
                return lines

            # Универсальный способ получить эмбеддинги
            if hasattr(encoder, 'transformer') and hasattr(encoder.transformer, 'text_model') and hasattr(encoder.transformer.text_model, 'embeddings'):
                embeddings = encoder.transformer.text_model.embeddings.token_embedding.weight
            elif hasattr(encoder, 'transformer') and hasattr(encoder.transformer, 'get_input_embeddings'):
                embeddings = encoder.transformer.get_input_embeddings().weight
            else:
                lines.append("Ошибка: Не удалось найти таблицу эмбеддингов для этой модели.")
                return lines

            for token_str in token_list:
                # Пропускаем пустые строки
                if not token_str.strip():
                    continue

                token_ids = tokenizer.encode(token_str)
                # Убираем токены начала/конца, которые обычно имеют ID 49406 и 49407
                meaningful_tokens = [tid for tid in token_ids if tid not in [tokenizer.bos_token_id, tokenizer.eos_token_id, getattr(tokenizer, 'pad_token_id', -1)]]
                # Fallback для старых токенизаторов
                if not meaningful_tokens and len(token_ids) > 2:
                    meaningful_tokens = token_ids[1:-1]
                
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
            lines.append(f"Исключение при обработке энкодера '{name}': {e}")
            lines.append(traceback.format_exc())
            
        return lines

    def check_token_weights(self, clip, tokens):
        encoders_to_check = {}

        # 1. Интроспекция: ищем вложенные энкодеры
        for attr_name in dir(clip):
            if attr_name.startswith('_'):
                continue
            
            try:
                attr_value = getattr(clip, attr_name)
                if hasattr(attr_value, 'tokenizer') and hasattr(attr_value, 'transformer'):
                    encoders_to_check[attr_name.upper()] = attr_value
            except Exception:
                continue

        # 2. Если ничего не нашли, проверяем сам объект clip
        if not encoders_to_check:
            if hasattr(clip, 'tokenizer') and hasattr(clip, 'transformer'):
                encoders_to_check['CLIP'] = clip
        
        # 3. Если снова неудача, сообщаем об ошибке
        if not encoders_to_check:
            error_msg = ("Ошибка: не удалось найти ни одного текстового энкодера.\n"
                         "Убедитесь, что на вход подан корректный CLIP-объект.")
            return (error_msg,)

        token_list = [t.strip() for t in re.split('[,\\n]', tokens) if t.strip()]
        if not token_list:
            return ("Введите токены для анализа.",)

        final_output = []
        for name, encoder in encoders_to_check.items():
            result_lines = self._process_encoder(name, encoder, token_list)
            final_output.extend(result_lines)
            final_output.append("")

        return ("\n".join(final_output),)
