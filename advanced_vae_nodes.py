import torch
import comfy.model_management as model_management

class VAEDecodeWithPrecision:
    """
    Расширенная нода VAEDecode, позволяющая выбирать точность (precision)
    для операции декодирования.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "samples": ("LATENT",),
                "precision": (["auto", "fp32", "fp16", "bf16"], {"default": "auto"}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "😎 SnJake/VAE" # Помещаем в подкатегорию, чтобы не мешать

    def decode(self, vae, samples, precision):
        """
        Декодирует латент в изображение с указанной точностью.
        """
        # 1. Определяем целевую точность (torch.dtype)
        if precision == "auto":
            # Используем стандартную логику ComfyUI для определения оптимального dtype
            target_dtype = model_management.vae_dtype(device=vae.device)
        elif precision == "fp32":
            target_dtype = torch.float32
        elif precision == "fp16":
            target_dtype = torch.float16
        elif precision == "bf16":
            target_dtype = torch.bfloat16
        else:
            print(f"Warning: Unknown precision '{precision}', falling back to auto.")
            target_dtype = model_management.vae_dtype(device=vae.device)

        # 2. Сохраняем исходное состояние VAE
        original_dtype = vae.dtype
        original_model_dtype = vae.first_stage_model.dtype
        
        print(f"Advanced VAEDecode: Decoding with precision {precision} ({target_dtype}). Original was {original_dtype}.")

        try:
            # 3. Временно изменяем точность VAE для выполнения операции
            # Метод vae.decode() внутренне использует vae.dtype для каста тензоров,
            # поэтому важно изменить оба атрибута.
            vae.first_stage_model.to(target_dtype)
            vae.dtype = target_dtype
            
            # 4. Выполняем декодирование
            images = vae.decode(samples["samples"])
            
            # Стандартная обработка размера батча из оригинальной ноды
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                
        finally:
            # 5. Возвращаем VAE в исходное состояние, чтобы не нарушать
            # работу других нод в воркфлоу.
            vae.first_stage_model.to(original_model_dtype)
            vae.dtype = original_dtype

        return (images,)


class VAEEncodeWithPrecision:
    """
    Расширенная нода VAEEncode, позволяющая выбирать точность (precision)
    для операции кодирования.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "pixels": ("IMAGE",),
                "precision": (["auto", "fp32", "fp16", "bf16"], {"default": "auto"}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "😎 SnJake/VAE"

    def encode(self, vae, pixels, precision):
        """
        Кодирует изображение в латент с указанной точностью.
        """
        # 1. Определяем целевую точность (torch.dtype)
        if precision == "auto":
            target_dtype = model_management.vae_dtype(device=vae.device)
        elif precision == "fp32":
            target_dtype = torch.float32
        elif precision == "fp16":
            target_dtype = torch.float16
        elif precision == "bf16":
            target_dtype = torch.bfloat16
        else:
            print(f"Warning: Unknown precision '{precision}', falling back to auto.")
            target_dtype = model_management.vae_dtype(device=vae.device)

        # 2. Сохраняем исходное состояние VAE
        original_dtype = vae.dtype
        original_model_dtype = vae.first_stage_model.dtype
        
        print(f"Advanced VAEEncode: Encoding with precision {precision} ({target_dtype}). Original was {original_dtype}.")

        try:
            # 3. Временно изменяем точность VAE
            vae.first_stage_model.to(target_dtype)
            vae.dtype = target_dtype
            
            # 4. Выполняем кодирование (только каналы RGB)
            latent = vae.encode(pixels[:,:,:,:3])

        finally:
            # 5. Возвращаем VAE в исходное состояние
            vae.first_stage_model.to(original_model_dtype)
            vae.dtype = original_dtype

        return ({"samples": latent},)
