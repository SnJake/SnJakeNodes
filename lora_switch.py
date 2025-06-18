class LoraSwitch6:
    """
    Нода-переключатель для 6 пар MODEL и CLIP.
    Выбирает активную пару входов (model_N, clip_N) на основе значения 'select'
    и передает их на выходы.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Определяет типы входов для ноды.
        - 'select': INT виджет для выбора активной пары.
        - 6 опциональных пар входов для MODEL и CLIP.
        """
        # Создаем словарь для опциональных входов динамически
        optional_inputs = {}
        for i in range(1, 7):
            optional_inputs[f"model_{i}"] = ("MODEL",)
            optional_inputs[f"clip_{i}"] = ("CLIP",)

        return {
            "required": {
                "select": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 6, 
                    "step": 1
                }),
            },
            "optional": optional_inputs
        }

    # Типы данных, которые нода возвращает
    RETURN_TYPES = ("MODEL", "CLIP")
    # Имена выходных сокетов
    RETURN_NAMES = ("model", "clip")

    # Название функции, которая будет выполняться
    FUNCTION = "switch_pair"

    # Категория, в которой нода появится в меню "Add Node"
    CATEGORY = "Switches"

    def switch_pair(self, select, **kwargs):
        """
        Основная логика ноды.
        'select' - это значение из INT виджета.
        'kwargs' будет содержать все опциональные входы, которые были подключены.
        """
        # Формируем имена ключей для выбранной пары
        model_key = f"model_{select}"
        clip_key = f"clip_{select}"

        # Получаем объекты model и clip из kwargs.
        # Если вход не подключен, kwargs.get вернет None.
        selected_model = kwargs.get(model_key, None)
        selected_clip = kwargs.get(clip_key, None)
        
        # Проверяем, что оба входа для выбранной пары подключены.
        # Если нет, ComfyUI в любом случае выдаст ошибку на следующей ноде,
        # которая ожидает эти данные, что является стандартным поведением.
        if selected_model is None:
            print(f"[LoraSwitch6] Внимание: Вход '{model_key}' не подключен, но выбран. На выход будет передан 'None'.")
        
        if selected_clip is None:
            print(f"[LoraSwitch6] Внимание: Вход '{clip_key}' не подключен, но выбран. На выход будет передан 'None'.")

        # Возвращаем выбранную пару. Важно вернуть кортеж.
        return (selected_model, selected_clip)
