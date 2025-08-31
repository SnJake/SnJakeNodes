class LoraSwitchDynamic:
    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {}
        for i in range(1, 100):
            optional_inputs[f"model_{i}"] = ("MODEL", {"lazy": True})
            optional_inputs[f"clip_{i}"] = ("CLIP", {"lazy": True})

        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 99}),
                "pairs": ("INT", {"default": 6, "min": 1, "max": 99}),
            },
            "optional": optional_inputs,
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "switch_pair"
    CATEGORY = "😎 SnJake/LoRA"

    def check_lazy_status(self, select, **kwargs):
        idx = int(select)
        if idx < 1 or idx > 99:
            return []
        return [f"model_{idx}", f"clip_{idx}"]

    def switch_pair(self, select, pairs, **kwargs):
        idx = int(select)
        if idx < 1:
            idx = 1
        if idx > int(pairs):
            idx = int(pairs)

        m_key = f"model_{idx}"
        c_key = f"clip_{idx}"

        model = kwargs.get(m_key, None)
        clip = kwargs.get(c_key, None)

        if model is None or clip is None:
            raise ValueError(
                f"[LoraSwitchDynamic] Пара #{idx} не подключена: "
                f"{m_key}={type(model)}, {c_key}={type(clip)}"
            )

        print(f"[LoraSwitchDynamic] pair #{idx} -> model:{type(model)}, clip:{type(clip)}")
        return (model, clip)
