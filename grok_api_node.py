import base64
import os
from io import BytesIO

import requests
from PIL import Image


XAI_RESPONSES_URL = "https://api.x.ai/v1/responses"
GROK_MODELS = [
    "grok-4.5",
    "grok-4.3",
    "grok-4.20-reasoning",
    "grok-4.20-non-reasoning",
    "grok-4.20-multi-agent",
]


def _image_to_data_url(image):
    pixels = image[0].detach().cpu().clamp(0, 1).mul(255).byte().numpy()
    image_pil = Image.fromarray(pixels)
    buffer = BytesIO()
    image_pil.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _get_response_text(data):
    parts = []
    for output in data.get("output", []):
        if output.get("type") != "message":
            continue
        for content in output.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                parts.append(content["text"])
    return "\n".join(parts)


class SnJakeGrokApi:
    FUNCTION = "generate"
    CATEGORY = "😎 SnJake/API"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (GROK_MODELS,),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "api_key": ("STRING", {"default": "", "placeholder": "xAI API key or XAI_API_KEY environment variable"}),
                "reasoning_effort": (["default", "low", "medium", "high"],),
                "max_output_tokens": ("INT", {"default": 4096, "min": 1, "max": 32768}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    def generate(self, model, prompt, api_key, reasoning_effort, max_output_tokens, image=None):
        api_key = api_key.strip() or os.getenv("XAI_API_KEY", "").strip()
        if not api_key:
            return ("xAI API error: API key is required.",)

        if image is None:
            input_data = prompt
        else:
            input_data = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": _image_to_data_url(image), "detail": "high"},
                        {"type": "input_text", "text": prompt},
                    ],
                }
            ]

        payload = {
            "model": model,
            "input": input_data,
            "max_output_tokens": max_output_tokens,
            "store": False,
        }
        if reasoning_effort != "default":
            payload["reasoning"] = {"effort": reasoning_effort}

        try:
            response = requests.post(
                XAI_RESPONSES_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=3600,
            )
        except requests.exceptions.RequestException as error:
            return (f"xAI API request failed: {error}",)

        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            return (f"xAI API returned HTTP {response.status_code} with an invalid JSON response.",)

        if not response.ok:
            error = data.get("error", {})
            message = error.get("message") if isinstance(error, dict) else str(error)
            return (f"xAI API error ({response.status_code}): {message or 'Unknown error'}",)

        text = _get_response_text(data)
        if not text:
            return ("xAI API error: response did not contain text.",)
        return (text,)
