import hashlib
import json
from collections import OrderedDict


class SnJakeExecutionCounter:
    CATEGORY = "😎 SnJake/Utils"
    FUNCTION = "count_execution"
    OUTPUT_NODE = True
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("current_count",)

    # Keep counters isolated per workflow and node instance.
    counters = OrderedDict()
    MAX_WORKFLOW_STATES = 128

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_count": ("INT", {"default": 25, "min": 1, "max": 2**31 - 1}),
                "label": ("STRING", {"default": "Execution Counter"}),
                "reset": ("BOOLEAN", {"default": False, "label_on": "reset", "label_off": "keep"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force execution on every queue run so the counter always advances.
        return float("NaN")

    @classmethod
    def _workflow_key(cls, prompt=None, extra_pnginfo=None):
        if isinstance(extra_pnginfo, dict):
            workflow = extra_pnginfo.get("workflow")
            if isinstance(workflow, dict):
                workflow_id = workflow.get("id")
                if workflow_id:
                    return f"workflow:{workflow_id}"

        prompt_payload = prompt if isinstance(prompt, dict) else {}
        prompt_json = json.dumps(prompt_payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        prompt_hash = hashlib.sha256(prompt_json.encode("utf-8")).hexdigest()
        return f"prompt:{prompt_hash}"

    @classmethod
    def _get_workflow_counters(cls, workflow_key):
        workflow_counters = cls.counters.get(workflow_key)
        if workflow_counters is None:
            workflow_counters = {}
            cls.counters[workflow_key] = workflow_counters
        else:
            cls.counters.move_to_end(workflow_key)

        while len(cls.counters) > cls.MAX_WORKFLOW_STATES:
            cls.counters.popitem(last=False)

        return workflow_counters

    def count_execution(self, max_count, label, reset, unique_id=None, prompt=None, extra_pnginfo=None):
        workflow_key = self._workflow_key(prompt=prompt, extra_pnginfo=extra_pnginfo)
        workflow_counters = self._get_workflow_counters(workflow_key)

        node_key = str(unique_id) if unique_id is not None else label

        if reset or node_key not in workflow_counters:
            workflow_counters[node_key] = 0

        workflow_counters[node_key] += 1
        current_count = workflow_counters[node_key]

        if current_count > max_count:
            raise RuntimeError(
                f"[SnJakeExecutionCounter] Limit reached for '{label}': "
                f"{current_count}/{max_count}. Workflow stopped."
            )

        return (current_count,)
