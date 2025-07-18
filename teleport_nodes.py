# /ComfyUI/custom_nodes/snjake_nodes/teleport_nodes.py

class SnJake_TeleportSet:
    # Этот узел теперь виртуальный. Его логика находится в JavaScript.
    CATEGORY = "😎 SnJake/Utils"
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("signal_passthrough",)
    FUNCTION = "do_nothing" # Функция-пустышка

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": ("*",),
                "constant": ("STRING", {"default": "default_pipe"}),
            }
        }

    def do_nothing(self, signal, constant):
        # Этот узел ничего не делает и просто возвращает то, что получил.
        # Вся "магия" происходит на стороне клиента (JS).
        return (signal,)

class SnJake_TeleportGet:
    # Этот узел также виртуальный.
    CATEGORY = "😎 SnJake/Utils"
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("signal",)
    FUNCTION = "do_nothing" # Функция-пустышка

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                 # Мы определяем это как COMBO, чтобы JS мог его найти и заполнить
                "constant": (["default_pipe"],),
            }
        }
    
    def do_nothing(self, constant):
        # Этот узел ничего не возвращает. Его выход будет виртуально
        # соединен с входом Set-узла через JavaScript.
        return (None,)
