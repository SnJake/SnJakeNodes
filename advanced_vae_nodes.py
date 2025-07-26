import torch

class VAEDecodeWithPrecision:
    """
    Расширенная нода VAEDecode, позволяющая вручную выбирать точность (precision).
    Версия 4: работает с менеджером памяти ComfyUI.
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
        # Если выбран 'auto', используем стандартный метод декодирования без вмешательства.
        if precision == "auto":
            images = vae.decode(samples["samples"])
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            return (images,)

        # Определяем целевую точность (torch.dtype) из строкового значения
        if precision == "fp32": target_dtype = torch.float32
        elif precision == "fp16": target_dtype = torch.float16
        elif precision == "bf16": target_dtype = torch.bfloat16
        else: # На случай непредвиденного значения, возвращаемся к 'auto'
            print(f"Warning: Unknown precision '{precision}', falling back to auto.")
            return self.decode(vae, samples, "auto")
        
        # Сохраняем исходную точность из обертки VAE
        original_vae_dtype = vae.vae_dtype
        
        # Если точность уже совпадает, ничего не меняем
        if original_vae_dtype == target_dtype:
            return self.decode(vae, samples, "auto")

        print(f"Advanced VAEDecode: Temporarily setting VAE precision from {original_vae_dtype} to {target_dtype}.")
        
        try:
            # Подменяем атрибут dtype в обертке VAE.
            # Теперь, когда вызовется vae.decode(), внутренний model_management
            # будет использовать эту точность для загрузки модели на GPU.
            vae.vae_dtype = target_dtype
            
            # Выполняем декодирование. ComfyUI сам позаботится о корректном касте модели.
            images = vae.decode(samples["samples"])
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                
        finally:
            # КРИТИЧЕСКИ ВАЖНО: Возвращаем исходную точность в обертку VAE
            vae.vae_dtype = original_vae_dtype
            print(f"Advanced VAEDecode: VAE precision restored to {original_vae_dtype}.")

        return (images,)


class VAEEncodeWithPrecision:
    """
    Расширенная нода VAEEncode, позволяющая вручную выбирать точность (precision).
    Версия 4: работает с менеджером памяти ComfyUI.
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
            return ({"samples": vae.encode(pixels[:,:,:,:3])},)
            
        if precision == "fp32": target_dtype = torch.float32
        elif precision == "fp16": target_dtype = torch.float16
        elif precision == "bf16": target_dtype = torch.bfloat16
        else:
            print(f"Warning: Unknown precision '{precision}', falling back to auto.")
            return self.encode(vae, pixels, "auto")

        original_vae_dtype = vae.vae_dtype

        if original_vae_dtype == target_dtype:
            return self.encode(vae, pixels, "auto")
        
        print(f"Advanced VAEEncode: Temporarily setting VAE precision from {original_vae_dtype} to {target_dtype}.")
        
        try:
            # Подменяем атрибут, чтобы менеджер памяти ComfyUI использовал нужный тип
            vae.vae_dtype = target_dtype
            
            # Выполняем кодирование
            latent = vae.encode(pixels[:,:,:,:3])
        finally:
            # Возвращаем исходную точность в обертку VAE
            vae.vae_dtype = original_vae_dtype
            print(f"Advanced VAEEncode: VAE precision restored to {original_vae_dtype}.")

        return ({"samples": latent},)
