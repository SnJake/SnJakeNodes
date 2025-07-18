class SnJake_TeleportSet:
    """
    Python-заглушка для виртуального узла Teleport Set.
    Вся логика находится в /js/snjake_teleport_ui.js
    """
    CATEGORY = "😎 SnJake/Utils"
    FUNCTION = "do_nothing"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def do_nothing(self, **kwargs):
        return (list(kwargs.values()),)


class SnJake_TeleportGet:
    """
    Python-заглушка для виртуального узла Teleport Get.
    Вся логика находится в /js/snjake_teleport_ui.js
    """
    CATEGORY = "😎 SnJake/Utils"
    FUNCTION = "do_nothing"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def do_nothing(self, **kwargs):
        return ([],)
