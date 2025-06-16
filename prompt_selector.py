class MultilinePromptSelector:
    CATEGORY = "😎 SnJake/Utils"
    FUNCTION = "choose_prompt"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Selected Prompt",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Многострочное текстовое поле для ввода промптов
                "prompts": ("STRING", {
                    "multiline": True,
                    "placeholder": "Введите промпты, каждый с новой строки"
                }),
                # Числовое поле для выбора строки (1-индексация)
                "select": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 999999,
                    "step": 1,
                    "tooltip": "Номер строки для выбора"
                }),
            },
            "hidden": {
                # Уникальный идентификатор узла (не используется в логике, но требуется)
                "unique_id": "UNIQUE_ID"
            }
        }

    def choose_prompt(self, prompts, select, unique_id):
        # Разбиваем текст на строки без фильтрации пустых
        lines = prompts.splitlines()
        # Если запрошенная строка отсутствует – возвращаем пустую строку
        if select < 1 or select > len(lines):
            return ("",)
        chosen = lines[select - 1]
        # Если выбранная строка пуста (после удаления пробелов), узел ничего не выводит (пустой результат)
        if not chosen.strip():
            return ("",)
        return (chosen,)
