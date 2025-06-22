import torch
import re
import traceback

class TokenWeightChecker:
    """
    Универсальная нода для проверки "веса" токенов.
    Динамически находит все текстовые энкодеры внутри объекта CLIP,
    включая обработку объектов-оберток (wrappers), что делает ее 
    совместимой с SD1.5, SDXL, SD3, Flux и другими архитектурами.
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
                lines.append("Ошибка: Токенизатор не найден в этом энкодере.")
                return lines

            # Универсальный способ получить эмбеддинги
            if hasattr(encoder, 'transformer') and hasattr(encoder.transformer, 'text_model') and hasattr(encoder.transformer.text_model, 'embeddings'):
                embeddings = encoder.transformer.text_model.embeddings.token_embedding.weight
            elif hasattr(encoder, 'transformer') and hasattr(encoder.transformer, 'get_input_embeddings'):
                embeddings = encoder.transformer.get_input_embeddings().weight
            else:
                lines.append("Ошибка: Не удалось найти таблицу эмбеддингов для этой модели.")
                return lines

            # Определяем ID служебных токенов
            special_ids = set()
            for token_name in ['bos_token_id', 'eos_token_id', 'pad_token_id', 'unk_token_id']:
                token_id = getattr(tokenizer, token_name, None)
                if token_id is not None:
                    special_ids.add(token_id)

            for token_str in token_list:
                if not token_str.strip():
                    continue

                token_ids = tokenizer.encode(token_str)
                meaningful_tokens = [tid for tid in token_ids if tid not in special_ids]
                
                if not meaningful_tokens:
                    # Если после фильтрации ничего не осталось, возможно это один токен, который был отфильтрован
                    # или токенизация не удалась. Выведем как есть для отладки.
                    decoded_tokens = [f"'{tokenizer.decode([tid])}'({tid})" for tid in token_ids]
                    lines.append(f"'{token_str}' -> [Нет значащих токенов] Raw: {' '.join(decoded_tokens)}")
                    continue

                if len(meaningful_tokens) > 1:
                    sub_tokens_info = []
                    for token_id in meaningful_tokens:
                        if token_id >= len(embeddings):
                            sub_tokens_info.append(f"id:{token_id}[вне словаря]")
                            continue
                        
                        sub_token_str = tokenizer.decode([token_id])
                        weight = torch.linalg.norm(embeddings[token_id]).item()
                        sub_tokens_info.append(f"'{sub_token_str}'({weight:.4f})")
                    lines.append(f"'{token_str}' -> {' '.join(sub_tokens_info)}")
                else:
                    token_id = meaningful_tokens[0]
                    if token_id >= len(embeddings):
                        lines.append(f"'{token_str}': id:{token_id}[вне словаря]")
                        continue
                    
                    weight = torch.linalg.norm(embeddings[token_id]).item()
                    lines.append(f"'{token_str}': {weight:.4f}")
            
        except Exception as e:
            lines.append(f"Исключение при обработке энкодера '{name}': {e}")
            lines.append(traceback.format_exc())
            
        return lines

    def check_token_weights(self, clip, tokens):
        encoders_to_check = {}
        
        # Список объектов для инспекции. Начинаем с самого clip
        # и добавляем его .target, если он существует.
        objects_to_inspect = [clip]
        if hasattr(clip, 'target'):
            objects_to_inspect.append(clip.target)

        for obj in objects_to_inspect:
            if obj is None: continue
            # Ищем энкодеры как атрибуты объекта
            for attr_name in dir(obj):
                if attr_name.startswith('_'):
                    continue
                
                try:
                    attr_value = getattr(obj, attr_name)
                    # Проверяем, похож ли атрибут на текстовый энкодер
                    if hasattr(attr_value, 'tokenizer') and hasattr(attr_value, 'transformer'):
                         # Не добавляем дубликаты
                        if id(attr_value) not in [id(v) for v in encoders_to_check.values()]:
                            encoders_to_check[attr_name.upper()] = attr_value
                except Exception:
                    continue
        
        # Если после всех поисков ничего не нашли, проверяем, не является ли сам 'clip' энкодером
        if not encoders_to_check:
            if hasattr(clip, 'tokenizer') and hasattr(clip, 'transformer'):
                encoders_to_check['CLIP'] = clip

        if not encoders_to_check:
            return ("Ошибка: не удалось найти ни одного текстового энкодера. Убедитесь, что на вход подан корректный CLIP-объект.",)

        token_list = [t.strip() for t in re.split('[,\\n]', tokens) if t.strip()]
        if not token_list:
            return ("Введите токены для анализа.",)

        final_output = []
        for name, encoder in sorted(encoders_to_check.items()):
            result_lines = self._process_encoder(name, encoder, token_list)
            final_output.extend(result_lines)
            final_output.append("")

        return ("\n".join(final_output),)
