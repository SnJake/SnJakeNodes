class SnJakeRandomPromptWindowSelector:
    FUNCTION = "choose_random_prompt"
    CATEGORY = "😎 SnJake/Utils"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)

    MAX_WINDOWS = 20

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "windows_visible": ("INT", {
                "default": 3,
                "min": 1,
                "max": cls.MAX_WINDOWS,
                "step": 1,
                "tooltip": "Internal widget for JS visibility control."
            })
        }

        for i in range(1, cls.MAX_WINDOWS + 1):
            required[f"prompt_{i}"] = ("STRING", {
                "default": "",
                "multiline": True,
                "placeholder": f"Prompt #{i}"
            })

        return {"required": required}

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Disable ComfyUI cache for this node so a new random prompt is chosen every run.
        return float("NaN")

    def choose_random_prompt(
        self,
        windows_visible,
        prompt_1="",
        prompt_2="",
        prompt_3="",
        prompt_4="",
        prompt_5="",
        prompt_6="",
        prompt_7="",
        prompt_8="",
        prompt_9="",
        prompt_10="",
        prompt_11="",
        prompt_12="",
        prompt_13="",
        prompt_14="",
        prompt_15="",
        prompt_16="",
        prompt_17="",
        prompt_18="",
        prompt_19="",
        prompt_20="",
    ):
        import random

        count = max(1, min(self.MAX_WINDOWS, int(windows_visible)))
        prompts = [
            prompt_1,
            prompt_2,
            prompt_3,
            prompt_4,
            prompt_5,
            prompt_6,
            prompt_7,
            prompt_8,
            prompt_9,
            prompt_10,
            prompt_11,
            prompt_12,
            prompt_13,
            prompt_14,
            prompt_15,
            prompt_16,
            prompt_17,
            prompt_18,
            prompt_19,
            prompt_20,
        ]

        visible_non_empty = [p for p in prompts[:count] if isinstance(p, str) and p.strip()]
        if not visible_non_empty:
            return ("",)

        return (random.choice(visible_non_empty),)
