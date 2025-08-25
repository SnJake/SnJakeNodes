class LoraSwitchDynamic:
    """
    Динамический переключатель с ленивыми вычислениями.
    Сообщает движку, какую ветку графа нужно вычислить,
    делая ноды-блокировщики ненужными.
    """
    
    @classmethod
    def INPUT_TYPES(cls):

        optional_inputs = {}
        for i in range(1, 7): 
            optional_inputs[f"model_{i}"] = ("MODEL", {"lazy": True})
            optional_inputs[f"clip_{i}"] = ("CLIP", {"lazy": True})

        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 99}),
                "pairs": ("INT", {"default": 6, "min": 1, "max": 99}),
            },
            "optional": optional_inputs
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "switch_pair"
    CATEGORY = "😎 SnJake/LoRA"

    def check_lazy_status(self, select, **kwargs):
        """
        Этот метод вызывается ДО основного, чтобы определить, какие 'lazy' входы нужно вычислить.
        """
        # `select` - это простое число (int), а не список. Убираем [0].
        selected_index = select
        
        needed_model = f"model_{selected_index}"
        needed_clip = f"clip_{selected_index}"
        
        return [needed_model, needed_clip]

    def switch_pair(self, select, pairs, **kwargs):
        """
        Эта функция теперь будет вызвана ПОСЛЕ того, как ComfyUI вычислит
        только те входы, которые мы запросили в check_lazy_status.
        """
        # `select` - это простое число (int), а не список. Убираем [0].
        selected_index = select
        
        model_key = f"model_{selected_index}"
        clip_key = f"clip_{selected_index}"

        selected_model = kwargs.get(model_key)
        selected_clip = kwargs.get(clip_key)
        
        print(f"[LoraSwitchDynamic] Switching to pair #{selected_index}. Passing model: {type(selected_model)}, clip: {type(selected_clip)}")

        return (selected_model, selected_clip)
