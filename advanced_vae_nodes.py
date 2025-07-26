import torch
import comfy.model_management as model_management

class VAEDecodeWithPrecision:
    """
    Расширенная нода VAEDecode, позволяющая вручную выбирать точность (precision)
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
    CATEGORY = "😎 SnJake/VAE"

    def decode(self, vae, samples, precision):
        if precision == "auto":
            # Если выбран 'auto', используем стандартный метод декодирования без изменений.
            # Это самый безопасный вариант по умолчанию.
            print("Advanced VAEDecode: Precision 'auto', using default VAE behavior.")
            images = vae.decode(samples["samples"])
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            return (images,)

        # Определяем целевую точность (torch.dtype) из строкового значения
        if precision == "fp32":
            target_dtype = torch.float32
        elif precision == "fp16":
            target_dtype = torch.float16
        elif precision == "bf16":
            target_dtype = torch.bfloat16
        else:
            # На случай непредвиденного значения, возвращаемся к 'auto'
            print(f"Warning: Unknown precision '{precision}', falling back to auto.")
            return self.decode(vae, samples, "auto")

        # Получаем доступ к реальной модели PyTorch внутри обертки VAE
        model_to_modify = vae.first_stage_model
        
        # Сохраняем исходную точность модели, чтобы восстановить её позже
        original_dtype = model_to_modify.dtype
        
        # Если точность уже совпадает, ничего не меняем
        if original_dtype == target_dtype:
            print(f"Advanced VAEDecode: VAE already in target precision ({precision}). No change needed.")
            return self.decode(vae, samples, "auto")

        print(f"Advanced VAEDecode: Temporarily casting VAE from {original_dtype} to {target_dtype} for decoding.")
        
        try:
            # Временно переводим модель в целевую точность
            model_to_modify.to(dtype=target_dtype)
            
            # Выполняем декодирование
            images = vae.decode(samples["samples"])
            
            # Стандартная обработка размера батча
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                
        finally:
            # КРИТИЧЕСКИ ВАЖНО: Возвращаем модель в исходное состояние
            # в блоке finally, чтобы это произошло даже в случае ошибки.
            model_to_modify.to(dtype=original_dtype)
            print(f"Advanced VAEDecode: VAE restored to original precision ({original_dtype}).")

        return (images,)


class VAEEncodeWithPrecision:
    """
    Расширенная нода VAEEncode, позволяющая вручную выбирать точность (precision)
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
        if precision == "auto":
            print("Advanced VAEEncode: Precision 'auto', using default VAE behavior.")
            return ({"samples": vae.encode(pixels[:,:,:,:3])},)

        # Определяем целевую точность (torch.dtype)
        if precision == "fp32":
            target_dtype = torch.float32
        elif precision == "fp16":
            target_dtype = torch.float16
        elif precision == "bf16":
            target_dtype = torch.bfloat16
        else:
            print(f"Warning: Unknown precision '{precision}', falling back to auto.")
            return self.encode(vae, pixels, "auto")

        model_to_modify = vae.first_stage_model
        original_dtype = model_to_modify.dtype

        if original_dtype == target_dtype:
            print(f"Advanced VAEEncode: VAE already in target precision ({precision}). No change needed.")
            return self.encode(vae, pixels, "auto")
        
        print(f"Advanced VAEEncode: Temporarily casting VAE from {original_dtype} to {target_dtype} for encoding.")
        
        try:
            # Временно изменяем точность
            model_to_modify.to(dtype=target_dtype)
            
            # Выполняем кодирование
            latent = vae.encode(pixels[:,:,:,:3])

        finally:
            # Возвращаем VAE в исходное состояние
            model_to_modify.to(dtype=original_dtype)
            print(f"Advanced VAEEncode: VAE restored to original precision ({original_dtype}).")

        return ({"samples": latent},)
