# ComfyUI/custom_nodes/MyDateTimeUtils/datetime_node.py
import datetime

class DateTimeToStringNode:
    @classmethod
    def INPUT_TYPES(cls):
        """
        Определяет типы входных данных для ноды.
        Эта нода не имеет входов.
        """
        return {
            "required": {}  # Нет обязательных входов
        }

    RETURN_TYPES = ("STRING",)  # Нода возвращает один выход типа STRING
    RETURN_NAMES = ("datetime_str",)  # Имя выходного сокета
    FUNCTION = "get_current_datetime_string"  # Метод, который будет вызван
    CATEGORY = "😎 SnJake/Utils"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        """
        Сообщает ComfyUI, что выходные данные этой ноды изменяются при каждом выполнении.
        Это необходимо, так как дата и время всегда новые.
        Возвращение float("NaN") гарантирует пересчет.
        """
        return float("NaN")

    def get_current_datetime_string(self):
        """
        Основная функция ноды. Получает текущую дату и время,
        форматирует их в строку и возвращает.
        """
        now = datetime.datetime.now()
        
        # Формат: YYYY-MM-DD_HH-MM-SS_ms (год-месяц-день_час-минута-секунда_миллисекунды)
        # Пример: 2023-10-27_15-30-45_123
        # Этот формат изначально использует подчеркивания и дефисы вместо пробелов,
        # что удовлетворяет требованию "Вместо пробелов должны быть нижние подчеркивания".
        # Если бы выбранный формат мог содержать пробелы (например, из datetime.ctime()),
        # то потребовалось бы .replace(" ", "_").
        datetime_string = now.strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]  # Обрезаем микросекунды до миллисекунд
        
        return (datetime_string,) # Возвращаем результат в виде кортежа