#
# switch_nodes.py
#

class SnJakeSwitch:
    """
    Базовый класс для переключателей. Он направляет один из двух входов 
    (on_true или on_false) на выход в зависимости от значения boolean-переключателя 'select'.
    """
    FUNCTION = "do_switch"
    CATEGORY = "😎 SnJake/Utils"

    @classmethod
    def INPUT_TYPES(cls):
        # Тип данных для входов определяется в дочерних классах через RETURN_TYPES
        return {
            "required": {
                "select": ("BOOLEAN", {"default": True, "label_on": "true", "label_off": "false"}),
                "on_true": (cls.RETURN_TYPES[0], {}),
                "on_false": (cls.RETURN_TYPES[0], {}),
            }
        }

    def do_switch(self, select, on_true, on_false):
        if select:
            return (on_true,)
        else:
            return (on_false,)

class SnJakeImageSwitch(SnJakeSwitch):
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

class SnJakeMaskSwitch(SnJakeSwitch):
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

class SnJakeLatentSwitch(SnJakeSwitch):
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

class SnJakeConditioningSwitch(SnJakeSwitch):
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

class SnJakeStringSwitch(SnJakeSwitch):
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)

    @classmethod
    def INPUT_TYPES(cls):
        # Переопределяем для добавления многострочного ввода текста
        return {
            "required": {
                "select": ("BOOLEAN", {"default": True, "label_on": "true", "label_off": "false"}),
                "on_true": ("STRING", {"default": "", "multiline": True}),
                "on_false": ("STRING", {"default": "", "multiline": True}),
            }
        }

class AlwaysEqualProxy(str):
    """
    Прокси-класс, который всегда возвращает True при сравнении.
    Используется для создания "any" типа (*), который принимает любые данные.
    """
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False

any_type = AlwaysEqualProxy("*")

class SnJakeAnySwitch:
    """
    Специальный переключатель, который принимает и отдает данные абсолютно любого типа.
    """
    FUNCTION = "do_switch"
    CATEGORY = "😎 SnJake/Utils"
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("BOOLEAN", {"default": True, "label_on": "true", "label_off": "false"}),
                "on_true": (any_type, {"forceInput": True}),
                "on_false": (any_type, {"forceInput": True}),
            }
        }

    def do_switch(self, select, on_true, on_false):
        if select:
            return (on_true,)
        else:
            return (on_false,)
