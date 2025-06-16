import requests
import json
import torch
import re
from PIL import Image
import numpy as np
import base64
from io import BytesIO

SYSTEM_TEMPLATE = """
# Coding a Python Function
        # Coding a Python Function
        You are an expert python coder who specializes in writing custom nodes for ComfyUI.

        ## Available Python Modules
        It is not required to use any of these libraries, but if you do use any import in your code it must be on this list:
        [[IMPORTS]]

        ## Input Data
        Here is some important information about the input data:
        - input_data_1: [[INPUT1]]
        - input_data_2: [[INPUT2]]
        The type of input_data_1 is: [[INPUT_TYPE]]
        [[CONNECTIONS]][[EXAMPLES]][[CODEBLOCK]]
        ## Coding Instructions
        - Your job is to code the user's requested node given the inputs and desired output type.
        - Respond with only a brief plan and the code in one function named generated_function that takes two kwargs named 'input_data_1' and 'input_data_2'.
        - All functions you must define should be inner functions of generated_function.
        - You may briefly plan your code in plain text, but after write only the code contents of the function itself inside of a `python` code block.
        - Do include needed available imports in your code before the function.
        - If the request is simple enough to do without imports, like math, just do that.
        - If an input is a Tensor and the output is a Tensor, it should be the same shape unless otherwise specified by the user.
        - Image tensors come in the shape (batch, width, height, rgb_channels), if outputting an image, use the same shape as the input image tensor.
            - To know the tensor is an image, it will come with the last dimension as 3
            - An example image tensor for a single 512x786 image: (1, 512, 786, 3)
            - An animation is a tensor with a larger batch of images of the same shape
        - Your resulting code should be as compute efficient as possible.
        - Make sure to deallocate memory for anything which uses it.
        - You may not use `open` or fetch files from the internet.
        - If there is a code block above, insure that the the generated_function args and kwargs match the example below.
        - **Crucially, ensure the output of the `generated_function` has the same Python type as the input `input_data_1`.**

        ### Example Generated function:
        User: output a list of all prime numbers up to the input number
        ```python
        import math
        def generated_function(input_data_1=None, input_data_2=None):
            def is_prime(n):
                if n <= 1:
                    return False
                for i in range(2, int(math.sqrt(n))+1):
                    if n % i == 0:
                        return False
                return True
            primes = []
            num = input_data_1
            while num > 1:
                if is_prime(num):
                    primes.append(num)
                num -= 1
            return primes
        ```

        ## Write the Code
"""

def extract_code(text):
    """Extracts Python code from a string."""
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1)
    return None

class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False

any_type = AlwaysEqualProxy("*")

# Узел для облачного OpenAI Compatible API (например, GPT-4o)
class OpenAICompatibleNode:
    CATEGORY = "😎 SnJake/AnyNode"
    FUNCTION = "process"
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_task": ("STRING", {"multiline": True, "default": "Process the given input data."}),
                "api_address": ("STRING", {"default": "https://api.openai.com/v1/chat/completions"}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
              "input_data": (any_type, {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT"
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def process(self, user_task, api_address, api_key, input_data=None, **kwargs):
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        input_type_str = self.get_input_type_str(input_data)
        messages = [
            {"role": "system", "content": SYSTEM_TEMPLATE.replace("[[INPUT]]", input_type_str)},
            {"role": "user", "content": user_task + "\nInput Data: " + str(input_data)}
        ]
        payload = {
            "model": "gpt-4o",
            "messages": messages,
            "temperature": 1
        }
        try:
            print("OpenAICompatibleNode: Sending payload to", api_address)
            response = requests.post(api_address, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            response_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            code = extract_code(response_content)

            if code:
                print("OpenAICompatibleNode: Executing generated code.")
                local_vars = {"input_data": input_data, "Image": Image, "np": np, "torch": torch}
                exec(code, {}, local_vars)

                if 'generated_function' in local_vars:
                    output = local_vars['generated_function'](input_data)
                else:
                    output = input_data
            else:
                print("OpenAICompatibleNode: No code found in response, returning original input.")
                output = input_data
            
            if output is not None:
                return (output,)
            else:
              return (input_data,)

        except Exception as e:
            print("OpenAICompatibleNode: Exception occurred:", e)
            return (input_data,)

    def get_input_type_str(self, input_data):
        if isinstance(input_data, torch.Tensor):
            if input_data.ndim == 4 and input_data.shape[-1] == 3:
                return "an image (torch.Tensor)"
            else:
                return "a tensor (torch.Tensor)"
        elif isinstance(input_data, str):
            return "a string (str)"
        elif isinstance(input_data, int):
            return "an integer (int)"
        elif isinstance(input_data, float):
            return "a float (float)"
        elif isinstance(input_data, list):
            return "a list (list)"
        elif isinstance(input_data, dict):
            return "a dictionary (dict)"
        else:
            return "an unknown type"
            

# Узел для локальных OpenAI Compatible API
class LocalOpenAICompatibleNode:
    CATEGORY = "😎 SnJake/AnyNode"
    FUNCTION = "process"
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_task": ("STRING", {"multiline": True, "default": "Process the given input data."}),
                "api_address": ("STRING", {"default": "http://localhost:5000/v1/chat/completions"}),
                "api_key": ("STRING", {"default": ""})
            },
            "optional": {
                "input_data": (any_type, {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT"
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def process(self, user_task, api_address, api_key, input_data=None, **kwargs):
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        input_type_str = self.get_input_type_str(input_data)
        messages = [
            {"role": "system", "content": SYSTEM_TEMPLATE.replace("[[INPUT]]", input_type_str)},
            {"role": "user", "content": user_task + "\nInput Data: " + str(input_data)}
        ]
        payload = {
            "model": "gpt-4o",
            "messages": messages,
            "temperature": 1
        }

        try:
            print("LocalOpenAICompatibleNode: Sending payload to", api_address)
            response = requests.post(api_address, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            response_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            code = extract_code(response_content)

            if code:
                print("LocalOpenAICompatibleNode: Executing generated code.")
                local_vars = {"input_data": input_data, "Image": Image, "np": np, "torch": torch}
                exec(code, {}, local_vars)

                if 'generated_function' in local_vars:
                    output = local_vars['generated_function'](input_data)

                else:
                    output = input_data
            else:
                print("LocalOpenAICompatibleNode: No code found in response, returning original input.")
                output = input_data

            if output is not None:
                return (output,)
            else:
              return (input_data,)
        except Exception as e:
            print("LocalOpenAICompatibleNode: Exception occurred:", e)
            return (input_data,)

    def get_input_type_str(self, input_data):
        if isinstance(input_data, torch.Tensor):
            if input_data.ndim == 4 and input_data.shape[-1] == 3:
                return "an image (torch.Tensor)"
            else:
                return "a tensor (torch.Tensor)"
        elif isinstance(input_data, str):
            return "a string (str)"
        elif isinstance(input_data, int):
            return "an integer (int)"
        elif isinstance(input_data, float):
            return "a float (float)"
        elif isinstance(input_data, list):
            return "a list (list)"
        elif isinstance(input_data, dict):
            return "a dictionary (dict)"
        else:
            return "an unknown type"