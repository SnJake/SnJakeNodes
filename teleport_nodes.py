# Глобальное хранилище для данных во время выполнения.
TELEPORT_DATA = {}

class AlwaysEqualProxy(str):
    """Класс-заглушка для типа ANY, чтобы ComfyUI не ругался на типы."""
    def __eq__(self, _): return True
    def __ne__(self, _): return False

any_type = AlwaysEqualProxy("*")

class SnJake_TeleportSet:
    CATEGORY = "😎 SnJake/Utils"
    RETURN_TYPES = () # У Set ноды нет реального выхода, она просто отправляет данные
    FUNCTION = "set_value"
    OUTPUT_NODE = True # Важно, чтобы нода всегда выполнялась

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": (any_type, {}),
                "constant": ("STRING", {"default": "default"}),
            }
        }

    def set_value(self, signal, constant):
        constant_clean = constant.strip()
        if constant_clean:
            TELEPORT_DATA[constant_clean] = signal
        # Ничего не возвращаем, так как RETURN_TYPES пуст
        return ()


class SnJake_TeleportGet:
    CATEGORY = "😎 SnJake/Utils"
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("signal",)
    FUNCTION = "get_value"

    @classmethod
    def INPUT_TYPES(cls):
        # JavaScript заполнит этот список динамически
        return {
            "required": {
                "constant": (["default"],),
            }
        }

    def get_value(self, constant):
        value = TELEPORT_DATA.get(constant, None)
        if value is None:
            # Предупреждение, если нода Get выполняется до ноды Set
            print(f"\033[93mWarning: [Teleport Get] Сигнал для канала '{constant}' не найден.\033[0m")
        return (value,)
