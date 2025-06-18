from comfy_execution.graph import ExecutionBlocker


class LoraSwitchDynamic:
    """
    Динамический переключатель с ленивыми вычислениями.
    Сообщает движку, какую ветку графа нужно вычислить,
    делая ноды-блокировщики ненужными.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # ВАЖНО: Мы добавляем флаг "lazy": True ко всем опциональным входам.
        # Это говорит ComfyUI не вычислять их, пока нода не попросит.
        optional_inputs = {}
        for i in range(1, 7): # Создадим 6 пар по умолчанию, JS сможет добавить еще
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
        # Получаем номер выбранной пары
        selected_index = select[0] # select приходит как список
        
        # Формируем имена нужных нам входов
        needed_model = f"model_{selected_index}"
        needed_clip = f"clip_{selected_index}"
        
        # Возвращаем список имен тех входов, которые нужны для выполнения
        return [needed_model, needed_clip]

    def switch_pair(self, select, pairs, **kwargs):
        """
        Эта функция теперь будет вызвана ПОСЛЕ того, как ComfyUI вычислит
        только те входы, которые мы запросили в check_lazy_status.
        """
        selected_index = select[0]
        
        model_key = f"model_{selected_index}"
        clip_key = f"clip_{selected_index}"

        # Теперь мы можем быть уверены, что в kwargs есть нужные нам данные
        selected_model = kwargs.get(model_key)
        selected_clip = kwargs.get(clip_key)
        
        print(f"[LoraSwitchDynamic] Switching to pair #{selected_index}. Passing model: {type(selected_model)}, clip: {type(selected_clip)}")

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
            },
            # Добавляем скрытый вход для получения ID ноды
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "block_or_pass"
    CATEGORY = "😎 SnJake/LoRA"

    def block_or_pass(self, model, clip, select, pass_on_select, unique_id):
        if select == pass_on_select:
            print(f"[LoraBlocker ID: {unique_id}] Pass -> (select: {select}, pass_on: {pass_on_select})")
            return (model, clip)
        else:
            print(f"[LoraBlocker ID: {unique_id}] Block -> (select: {select}, pass_on: {pass_on_select})")
            # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
            # Возвращаем ОДИН объект ExecutionBlocker. ComfyUI сам заблокирует все выходы.
            return ExecutionBlocker(None)
