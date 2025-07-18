class SnJake_TeleportSet:
    """
    ВИРТУАЛЬНЫЙ УЗЕЛ. Вся логика находится в JS.
    Этот узел просто объявляет входы/выходы для UI.
    """
    CATEGORY = "😎 SnJake/Utils"
    FUNCTION = "do_nothing"
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("signal_passthrough",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": ("*",),
                "constant": ("STRING", {"default": "default_pipe"}),
            }
        }

    def do_nothing(self, signal, **kwargs):
        # Возвращаем сигнал для сквозного соединения
        return (signal,)

class SnJake_TeleportGet:
    """
    ВИРТУАЛЬНЫЙ УЗЕЛ. Вся логика находится в JS.
    Этот узел объявляет входы/выходы для UI.
    """
    CATEGORY = "😎 SnJake/Utils"
    FUNCTION = "do_nothing"
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("signal",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Важно: объявляем как STRING, чтобы избежать валидации на стороне бэкенда.
                # JS превратит это в выпадающий список.
                "constant": ("STRING", {"default": "default_pipe"}),
            }
        }

    def do_nothing(self, **kwargs):
        # Этот узел ничего не делает. Данные будут получены "виртуально".
        return (None,)
