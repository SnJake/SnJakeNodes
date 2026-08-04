#
# text_utils_nodes.py
#

class SnJakeRemoveTextRange:
    FUNCTION = "remove_text_range"
    CATEGORY = "😎 SnJake/Utils"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "start_word": ("STRING", {"default": ""}),
                "end_word": ("STRING", {"default": ""}),
            }
        }

    def remove_text_range(self, text, start_word, end_word):
        if not start_word or not end_word:
            return (text,)

        start = text.find(start_word)
        if start == -1:
            return (text,)

        end = text.find(end_word, start + len(start_word))
        if end == -1:
            return (text,)

        return (text[:start] + text[end + len(end_word):],)


class SnJakeTextConcatenate:
    """
    Нода для соединения (конкатенации) до четырех строк текста с использованием
    указанного разделителя (делиметера). Пустые строки игнорируются.
    """
    
    FUNCTION = "concatenate"
    CATEGORY = "😎 SnJake/Utils"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_text",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_1": ("STRING", {"default": "", "multiline": True}),
                "text_2": ("STRING", {"default": "", "multiline": True}),
                "text_3": ("STRING", {"default": "", "multiline": True}),
                "text_4": ("STRING", {"default": "", "multiline": True}),
                "delimiter": ("STRING", {"default": ", "}), # Разделитель по умолчанию - запятая с пробелом
            },
            "optional": {
            }
        }

    def concatenate(self, text_1, text_2, delimiter, text_3="", text_4=""):
        # Собираем все текстовые части в один список
        parts = [text_1, text_2, text_3, text_4]
        
        # Фильтруем список, чтобы убрать пустые строки
        non_empty_parts = [part for part in parts if part]
        
        # Соединяем непустые части с помощью указанного разделителя
        result = delimiter.join(non_empty_parts)
        
        return (result,)

class SnJakeMultilineText:
    """
    Простая нода для удобного ввода большого многострочного текста.
    """

    FUNCTION = "get_text"
    CATEGORY = "😎 SnJake/Utils"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "",
                    "multiline": True, # Это свойство делает поле ввода большим
                }),
            }
        }

    def get_text(self, text):
        # Просто возвращаем текст, который ввел пользователь
        return (text,)
