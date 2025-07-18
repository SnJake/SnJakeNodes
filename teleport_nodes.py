class SnJake_TeleportSet:
    """
    Python-заглушка для виртуального узла Teleport Set.
    Вся логика реализована в /js/snjake_teleport_ui.js
    """
    CATEGORY = "😎 SnJake/Utils"
    FUNCTION = "do_nothing"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}} # JS добавит входы динамически

    def do_nothing(self, **kwargs):
        return ()


class SnJake_TeleportGet:
    """
    Python-заглушка для виртуального узла Teleport Get.
    Вся логика реализована в /js/snjake_teleport_ui.js
    """
    CATEGORY = "😎 SnJake/Utils"
    FUNCTION = "do_nothing"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}} # JS добавит входы динамически

    def do_nothing(self, **kwargs):
        return ()
