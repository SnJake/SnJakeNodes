import random
import sys

class SnJakeRandomNumberGenerator:
    
    FUNCTION = "generate_numbers"
    CATEGORY = "😎 SnJake/Utils"
    # Для ясности я переименовал выходы
    RETURN_TYPES = ("INT", "FLOAT", "INT")
    RETURN_NAMES = ("seed_out", "float_out", "int_out")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "min_float": ("FLOAT", {"default": 0.0, "min": -sys.maxsize, "max": sys.maxsize, "step": 0.01, "round": 0.001}),
                "max_float": ("FLOAT", {"default": 1.0, "min": -sys.maxsize, "max": sys.maxsize, "step": 0.01, "round": 0.001}),
                "min_int": ("INT", {"default": 0, "min": -sys.maxsize, "max": sys.maxsize}),
                "max_int": ("INT", {"default": 1024, "min": -sys.maxsize, "max": sys.maxsize}),
            }
        }

    def generate_numbers(self, seed, min_float, max_float, min_int, max_int):
        # Гарантируем, что min не больше max. Если это так, меняем их местами.
        if min_float > max_float:
            min_float, max_float = max_float, min_float
        if min_int > max_int:
            min_int, max_int = max_int, min_int

        # Устанавливаем seed, чтобы генерация была предсказуемой
        random.seed(seed)
        
        # Генерируем числа в заданных пользователем диапазонах
        generated_float = random.uniform(min_float, max_float)
        generated_int = random.randint(min_int, max_int)
        
        # Возвращаем seed (как первый выход) и сгенерированные числа
        return (seed, generated_float, generated_int)
