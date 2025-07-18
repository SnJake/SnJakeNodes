# /ComfyUI/custom_nodes/snjake_teleport_nodes/teleport_nodes.py

import server
from aiohttp import web

# --- Глобальные переменные ---
TELEPORT_DATA = {}
TELEPORT_CONSTANTS = {"default"} # Начинаем с одного значения, чтобы избежать ошибок

# --- API Эндпоинты для JavaScript ---

# Этот эндпоинт будет вызываться Get-нодой для получения актуального списка каналов
@server.PromptServer.instance.routes.get("/snjake/get_teleport_constants")
async def get_teleport_constants(request):
    return web.json_response(sorted(list(TELEPORT_CONSTANTS)))

# Этот эндпоинт будет вызываться Set-нодой, когда пользователь вводит новый канал
@server.PromptServer.instance.routes.post("/snjake/add_teleport_constant")
async def add_teleport_constant(request):
    try:
        data = await request.json()
        constant = data.get("constant")
        if constant and isinstance(constant, str):
            constant_clean = constant.strip()
            if constant_clean:
                TELEPORT_CONSTANTS.add(constant_clean)
                return web.json_response({"status": "ok", "message": f"Added '{constant_clean}'"})
        return web.json_response({"status": "error", "message": "Invalid constant"}, status=400)
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=500)


# --- Классы нод ---

class AlwaysEqualProxy(str):
    def __eq__(self, _): return True
    def __ne__(self, _): return False

any_type = AlwaysEqualProxy("*")

class SnJake_TeleportSet:
    CATEGORY = "😎 SnJake/Utils"
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("signal_passthrough",)
    FUNCTION = "set_value"
    OUTPUT_NODE = True

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
            TELEPORT_CONSTANTS.add(constant_clean)
        return (signal,)


class SnJake_TeleportGet:
    CATEGORY = "😎 SnJake/Utils"
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("signal",)
    FUNCTION = "get_value"

    @classmethod
    def INPUT_TYPES(cls):
        # Теперь мы просто создаем COMBO. JavaScript заполнит его данными.
        return {
            "required": {
                "constant": (["default"],),
            }
        }

    def get_value(self, constant):
        value = TELEPORT_DATA.get(constant, None)
        if value is None:
            print(f"\033[93mWarning: [Teleport Get] Signal for channel '{constant}' not found.\033[0m")
        return (value,)
