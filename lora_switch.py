from comfy_execution.graph import ExecutionBlocker


class LoraSwitchDynamic:
    """
    Динамический переключатель для N пар MODEL и CLIP.
    Выбирает активную пару на основе 'select' и передает ее на выход.
    Количество пар задается в виджете 'pairs' и обновляется по кнопке в интерфейсе.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 999}),
                # Этот виджет будет управлять количеством входов в JS
                "pairs": ("INT", {"default": 6, "min": 1, "max": 99}),
            },
            # Мы определим 1-ю пару по умолчанию, остальные добавит JS
            "optional": {
                "model_1": ("MODEL",),
                "clip_1": ("CLIP",),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "switch_pair"
    CATEGORY = "Switches"

    def switch_pair(self, select, pairs, **kwargs):
        """
        Логика ноды. 'pairs' здесь не используется, но должен быть в аргументах.
        """
        model_key = f"model_{select}"
        clip_key = f"clip_{select}"

        selected_model = kwargs.get(model_key)
        selected_clip = kwargs.get(clip_key)
        
        if selected_model is None:
            print(f"[LoraSwitchDynamic] Warning: Input '{model_key}' is not connected but selected. Passing through 'None'.")
        
        if selected_clip is None:
            print(f"[LoraSwitchDynamic] Warning: Input '{clip_key}' is not connected but selected. Passing through 'None'.")

        return (selected_model, selected_clip)




class LoraBlocker:
    """
    Блокирует или пропускает сигнал MODEL и CLIP.
    Если 'select' РАВЕН 'pass_on_select', сигнал проходит.
    В противном случае, выполнение последующих нод блокируется.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "select": ("INT", {"default": 1, "min": 1, "max": 999}),
                "pass_on_select": ("INT", {"default": 1, "min": 1, "max": 999, "tooltip": "Значение, при котором сигнал должен пройти"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "block_or_pass"
    CATEGORY = "Switches"

    def block_or_pass(self, model, clip, select, pass_on_select):
        # Если текущий выбор совпадает с тем, при котором нужно пропустить сигнал
        if select == pass_on_select:
            print(f"[LoraBlocker] Pass -> (select: {select}, pass_on: {pass_on_select})")
            return (model, clip)
        else:
            # В противном случае блокируем выполнение
            print(f"[LoraBlocker] Block -> (select: {select}, pass_on: {pass_on_select})")
            # ExecutionBlocker(None) молча остановит выполнение этой ветки
            return (ExecutionBlocker(None), ExecutionBlocker(None))
