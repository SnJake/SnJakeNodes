TELEPORT_DATA = {}

class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True
    def __ne__(self, _):
        return False

any_type = AlwaysEqualProxy("*")

class SnJake_TeleportSet:
    CATEGORY = "😎 SnJake/Utils"
    FUNCTION = "set_data"
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("signal_passthrough",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": (any_type, {}),
                "constant": ("STRING", {"default": "default_pipe"}),
            }
        }

    def set_data(self, signal, constant: str):
        if not constant.strip():
            raise ValueError("[SnJake_TeleportSet] Имя 'constant' не может быть пустым.")
        TELEPORT_DATA[constant] = signal
        return (signal,)

class SnJake_TeleportGet:
    CATEGORY = "😎 SnJake/Utils"
    FUNCTION = "get_data"
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("signal",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Этот список будет динамически генерироваться на бэкенде
                "constant": ([],),
            },
            # Скрытый вход, который дает нам доступ ко всему графу
            "hidden": {"prompt": "PROMPT"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Принудительно пересчитываем узел, чтобы он всегда получал свежие данные
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, prompt, **kwargs):
        # *** КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ДЛЯ ВАЛИДАЦИИ ***
        # Перед выполнением основного метода валидации мы находим все
        # Set-узлы в графе и динамически обновляем список доступных 'constant'.
        if not prompt:
            return True # Граф еще не полностью построен

        available_constants = []
        # Итерируемся по всем узлам в графе (prompt)
        for node_id, node_data in prompt.items():
            # Проверяем, является ли узел нашим Set-узлом
            if node_data.get("class_type") == "SnJake_TeleportSet":
                # Извлекаем значение 'constant' из его входов
                constant_value = node_data.get("inputs", {}).get("constant")
                if constant_value and isinstance(constant_value, str):
                    available_constants.append(constant_value)

        # Убираем дубликаты и сортируем
        available_constants = sorted(list(set(available_constants)))
        
        if not available_constants:
            available_constants.append("(no channels found)")

        # Теперь мы можем проверить, есть ли выбранное значение в актуальном списке
        selected_constant = kwargs.get("constant")
        if selected_constant not in available_constants:
             return f"Value not in list: constant '{selected_constant}' not in {available_constants}. " \
                    f"Possible race condition or the Set node hasn't been evaluated."

        return True

    def get_data(self, constant: str, prompt):
        # Валидация уже прошла, теперь получаем данные
        if constant not in TELEPORT_DATA:
            raise KeyError(
                f"[SnJake_TeleportGet] Канал '{constant}' не найден. "
                f"Убедитесь, что узел 'SnJake_TeleportSet' с этим каналом выполняется до данного узла."
            )
        signal = TELEPORT_DATA.get(constant)
        return (signal,)

# Очистка хранилища (остается без изменений)
try:
    import execution
    # Проверяем, не был ли метод уже "обернут"
    if not hasattr(execution.PromptQueue, '_snjake_patched'):
        original_execute = execution.PromptQueue.prototype_execute
        def clear_teleport_data_before_execution(func):
            def wrapper(*args, **kwargs):
                global TELEPORT_DATA
                TELEPORT_DATA.clear()
                return func(*args, **kwargs)
            return wrapper
        execution.PromptQueue.prototype_execute = clear_teleport_data_before_execution(original_execute)
        execution.PromptQueue._snjake_patched = True
except Exception as e:
    print(f"[SnJake Teleport Nodes] Warning: Could not patch PromptQueue for automatic data clearing. {e}")
