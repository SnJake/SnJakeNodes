# /ComfyUI/custom_nodes/snjake_nodes/teleport_nodes.py

# Статическое хранилище для передачи данных между узлами в рамках одного выполнения.
# ВНИМАНИЕ: Это хранилище не является потокобезопасным и очищается перед каждым новым запуском.
# Его использование может привести к непредсказуемым результатам, если порядок выполнения
# графа не гарантирует, что Set-узел всегда выполняется перед Get-узлом.
TELEPORT_DATA = {}

class AlwaysEqualProxy(str):
    """
    Прокси-класс для типа данных. Он обходит стандартную проверку типов ComfyUI,
    позволяя соединять любой выход с входом, использующим этот тип.
    Это необходимо для создания универсальных входов/выходов "any".
    """
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False

any_type = AlwaysEqualProxy("*")

class SnJake_TeleportSet:
    """
    Узел для отправки ("телепортации") данных по именованному каналу.
    Он сохраняет входящий сигнал в глобальном хранилище и пропускает его через себя
    для возможности дальнейшего последовательного соединения.
    """
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
        
        # Сохраняем данные в статическом хранилище
        TELEPORT_DATA[constant] = signal
        
        # Возвращаем сигнал для сквозной передачи
        return (signal,)

class SnJake_TeleportGet:
    """
    Узел для получения ("телепортации") данных из именованного канала.
    Извлекает данные, ранее сохраненные узлом SnJake_TeleportSet с тем же 'constant'.
    """
    CATEGORY = "😎 SnJake/Utils"
    FUNCTION = "get_data"
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("signal",)

    @classmethod
    def INPUT_TYPES(cls):
        # JS на стороне клиента будет динамически обновлять этот список.
        # В Python мы просто предоставляем базовую структуру.
        return {
            "required": {
                "constant": (["default_pipe"],),
            }
        }

    def get_data(self, constant: str):
        if constant not in TELEPORT_DATA:
            # Это исключение подчеркивает проблему порядка выполнения.
            raise KeyError(
                f"[SnJake_TeleportGet] Канал '{constant}' не найден. "
                f"Убедитесь, что узел 'SnJake_TeleportSet' с этим каналом "
                f"выполняется до данного узла."
            )
        
        signal = TELEPORT_DATA.get(constant)
        return (signal,)

# Очистка хранилища перед каждым новым запуском графа.
# Мы "обезьяньим патчем" добавляем логику очистки в `PromptQueue`.
try:
    import execution
    def clear_teleport_data_before_execution(func):
        def wrapper(*args, **kwargs):
            global TELEPORT_DATA
            TELEPORT_DATA.clear()
            return func(*args, **kwargs)
        return wrapper

    execution.PromptQueue.prototype_execute = clear_teleport_data_before_execution(execution.PromptQueue.prototype_execute)
except Exception as e:
    print(f"[SnJake Teleport Nodes] Warning: Could not patch PromptQueue for automatic data clearing. {e}")
