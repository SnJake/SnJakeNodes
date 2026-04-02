from .switch_nodes import any_type


class SnJakeExecutionCounter:
    CATEGORY = "😎 SnJake/Utils"
    FUNCTION = "count_execution"
    RETURN_TYPES = (any_type, "INT")
    RETURN_NAMES = ("output", "current_count")

    counters = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_value": (any_type, {"forceInput": True}),
                "max_count": ("INT", {"default": 25, "min": 1, "max": 2**31 - 1}),
                "label": ("STRING", {"default": "Execution Counter"}),
                "reset": ("BOOLEAN", {"default": False, "label_on": "reset", "label_off": "keep"}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force execution on every queue run so the counter always advances.
        return float("NaN")

    def count_execution(self, input_value, max_count, label, reset):
        if reset or label not in self.counters:
            self.counters[label] = 0

        self.counters[label] += 1
        current_count = self.counters[label]

        if current_count >= max_count:
            raise RuntimeError(
                f"[SnJakeExecutionCounter] Limit reached for '{label}': "
                f"{current_count}/{max_count}. Workflow stopped."
            )

        return (input_value, current_count)
