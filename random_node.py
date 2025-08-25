class SnJakeNumberNode:
    FUNCTION = "get_number"
    CATEGORY = "😎 SnJake/Utils"
    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("int", "float")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True, # Опция для инкремента, декремента, рандомизации
                }),
            }
        }

    def get_number(self, seed):
        # Просто преобразуем входное значение в нужные типы и возвращаем.
        output_int = int(seed)
        output_float = float(seed)
        
        return (output_int, output_float)
