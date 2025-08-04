import random
import sys

class SnJakeRandomNumberGenerator:
    """
    Эта нода генерирует случайные числа на основе входного seed.
    Она не имеет входных сокетов, только виджеты для управления.
    """
    
    FUNCTION = "generate_numbers"
    CATEGORY = "😎 SnJake/Utils"
    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "INT")
    RETURN_NAMES = ("seed", "number", "float", "int")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True, # Добавляет опцию 'control_after_generate'
                }),
            }
        }

    def generate_numbers(self, seed):
        # Устанавливаем seed для предсказуемости генерации
        random.seed(seed)
        
        # Генерируем случайный float в диапазоне от 0.0 до 100.0 для выхода "number"
        output_number = random.uniform(0.0, 100.0)

        # Генерируем стандартный float от 0.0 до 1.0
        output_float = random.random()
        
        # Генерируем случайное целое число в максимально возможном диапазоне
        output_int = random.randint(0, sys.maxsize)

        # Возвращаем кортеж со всеми значениями
        return (seed, output_number, output_float, output_int)
