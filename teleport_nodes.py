# Глобальное хранилище для "телепортируемых" сигналов.
# Это простой и эффективный способ обмена данными между нодами в рамках одного запуска.
TELEPORT_DATA = {}

# Глобальное множество для хранения всех имен констант.
# Используется для динамического создания списка в ноде Get.
# Инициализируем с одним значением, чтобы избежать ошибок с пустым списком при первой загрузке.
TELEPORT_CONSTANTS = {"default_constant"}


class AlwaysEqualProxy(str):
    """
    Класс-заглушка для типа данных "ANY" (*).
    Объект этого класса всегда равен любому другому объекту,
    что позволяет обходить стандартную проверку типов в ComfyUI.
    """
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False

# Создаем экземпляр для использования в качестве типа "ANY"
any_type = AlwaysEqualProxy("*")


class SnJake_TeleportSet:
    """
    Нода для отправки (установки) данных в "телепорт" под уникальным именем.
    """
    # Категория, в которой нода будет отображаться в меню ComfyUI
    CATEGORY = "😎 SnJake/Utils" # Помещаем в Utils, как более подходящее место

    # Имена и типы выходных данных
    RETURN_TYPES = (any_type,)
    # Также возвращаем сигнал для возможности сквозного соединения (chaining)
    RETURN_NAMES = ("signal_passthrough",) 

    FUNCTION = "set_value"
    OUTPUT_NODE = True # Важно, чтобы эта нода выполнялась, даже если ее выход не подключен

    @classmethod
    def INPUT_TYPES(cls):
        """Определение входных данных для ноды."""
        return {
            "required": {
                # Вход для любого сигнала
                "signal": (any_type, {}),
                # Поле для ввода имени константы (канала телепорта)
                "constant": ("STRING", {"default": "constant_name"}),
            }
        }

    def set_value(self, signal, constant):
        """
        Сохраняет полученный сигнал в глобальном хранилище.
        """
        constant_clean = constant.strip()
        if not constant_clean:
            # Предупреждение, если пользователь оставил имя пустым
            print("\033[93mWarning: [Teleport Set] Имя константы не может быть пустым. Сигнал не был сохранен.\033[0m")
            return (signal,) # Возвращаем сигнал как есть

        # Сохраняем данные и имя константы
        TELEPORT_DATA[constant_clean] = signal
        TELEPORT_CONSTANTS.add(constant_clean)

        print(f"Info: [Teleport Set] Сигнал сохранен в канале '{constant_clean}'.")

        # Возвращаем исходный сигнал, чтобы можно было строить цепочки дальше
        return (signal,)


class SnJake_TeleportGet:
    """
    Нода для получения (извлечения) данных из "телепорта" по имени.
    """
    CATEGORY = "😎 SnJake/Utils"

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("signal",)
    FUNCTION = "get_value"

    @classmethod
    def INPUT_TYPES(cls):
        """
        Определение входных данных. Список констант для выпадающего меню
        генерируется динамически из всех используемых в графе Set-нод.
        """
        # Преобразуем множество имен констант в отсортированный список для выпадающего меню (COMBO)
        constants_list = sorted(list(TELEPORT_CONSTANTS))
        return {
            "required": {
                # Выпадающий список с доступными каналами
                "constant": (constants_list, ),
            }
        }

    def get_value(self, constant):
        """
        Извлекает сигнал из глобального хранилища по имени константы.
        """
        # Получаем данные. Если данных нет, возвращаем None (ComfyUI обработает это).
        value = TELEPORT_DATA.get(constant, None)

        if value is None:
            # Предупреждение, если Set-нода еще не выполнилась
            print(f"\033[93mWarning: [Teleport Get] Сигнал для канала '{constant}' не найден. Возможно, соответствующая нода Set еще не была выполнена.\033[0m")

        # Возвращаем найденное значение в виде кортежа
        return (value,)
