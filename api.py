import json
import requests
import torch
import base64
from io import BytesIO
from PIL import Image

class VLMApiNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "use_api_key": ("BOOLEAN", {
                    "default": False, 
                    "label_on": "Use GPT-4o API", 
                    "label_off": "Use local server"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "placeholder": "Enter your OpenAI key (e.g. sk-...)"
                }),
                "api_url": ("STRING", {"default": "http://127.0.0.1:5000/v1/chat/completions"}),
                "custom_instruction": ("STRING", {"multiline": True, "default": "Describe this image in detail."}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 150, "min": 1, "max": 2048}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "vlm_analyze"
    CATEGORY = "😎 SnJake/VLM"

    def vlm_analyze(
        self,
        image,
        use_api_key,
        api_key,
        api_url,
        custom_instruction,
        top_k,
        top_p,
        max_new_tokens,
        temperature,
        min_p
    ):
        """
        Если use_api_key=True, запрашиваем GPT-4o по API,
        иначе идём на локальный адрес (например, ваш локальный LLM сервер).
        """

        # Извлекаем image[0] из batch, переводим в PIL и кодируем в base64
        image_tensor = image[0].cpu().numpy()
        image_pil = Image.fromarray((image_tensor * 255).astype('uint8'))
        buffered = BytesIO()
        image_pil.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Формируем сообщение: текст + инлайн "image_url" с base64
        messages_payload = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": custom_instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    }
                ]
            }
        ]

        if use_api_key:
            # ====== GPT-4o запрос =======
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}',
            }
            payload = {
                "model": "gpt-4o",  # или другой GPT-4o модель
                "messages": messages_payload,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_new_tokens,  # устаревшее поле, но для примера
            }

            response = None  # Инициализируем заранее
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()

                data = response.json()
                # Обычно структура ответа:
                # {
                #   "id": ...,
                #   "object": "chat.completion",
                #   "choices": [
                #       {
                #           "message": {
                #               "role": "assistant",
                #               "content": "...финальный текст..."
                #           }
                #       }
                #   ]
                #   ...
                # }
                if "choices" in data and len(data["choices"]) > 0:
                    text = data["choices"][0]["message"]["content"]
                else:
                    text = "No content returned."
                return (text,)

            except requests.exceptions.RequestException as e:
                return (f"Error during OpenAI API request: {e}",)
            except (json.JSONDecodeError, KeyError) as e:
                return (f"Error parsing OpenAI response: {e}",)

        else:
            # ====== Локальный сервер =======
            headers = {
                'Content-Type': 'application/json',
            }
            payload = {
                "messages": messages_payload,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": max_new_tokens,
                "min_p": min_p,
            }

            response = None  # Инициализируем заранее
            try:
                response = requests.post(api_url, headers=headers, json=payload)
                response.raise_for_status()

                # Предположим, что локальный сервер тоже возвращает JSON
                # по схеме, аналогичной OpenAI ({"choices": [{...}]})
                # Если у вас другая схема — подстройте парсинг под неё.
                data = json.loads(response.text)

                if "choices" in data and len(data["choices"]) > 0:
                    # Берём сам ответ из поля "content"
                    text = data["choices"][0]["message"]["content"]
                else:
                    text = "No content returned."

                return (text,)

            except requests.exceptions.RequestException as e:
                return (f"Error during local API request: {e}",)
            except json.JSONDecodeError as e:
                # Если локальный сервер вернул не JSON или неправильный JSON
                # вернём «как есть» текст ответа
                if response is not None:
                    return (response.text,)
                else:
                    return (f"Error decoding JSON response: {e}",)
