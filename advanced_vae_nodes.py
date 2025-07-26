import torch

class VAEDecodeWithPrecision:
    """
    Расширенная нода VAEDecode, позволяющая вручную выбирать точность (precision)
    для операции декодирования. Версия 3, исправленная.
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
        # Функция для стандартного декодирования без вмешательства
        def default_decode():
            images = vae.decode(samples["samples"])
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            return (images,)

        if precision == "auto":
            print("Advanced VAEDecode: Precision 'auto', using default VAE behavior.")
            return default_decode()

        # Определяем целевую точность
        if precision == "fp32": target_dtype = torch.float32
        elif precision == "fp16": target_dtype = torch.float16
        elif precision == "bf16": target_dtype = torch.bfloat16
        else:
            print(f"Warning: Unknown precision '{precision}', falling back to auto.")
            return default_decode()

        # Получаем доступ к реальной модели PyTorch
        model_to_modify = vae.first_stage_model
        
        try:
            # Надежный способ получить текущую точность модели
            original_dtype = next(model_to_modify.parameters()).dtype
        except StopIteration:
            # Если у модели нет параметров, ничего не делаем
            print("Warning: VAE model has no parameters. Using default behavior.")
            return default_decode()
        
        if original_dtype == target_dtype:
            print(f"Advanced VAEDecode: VAE already in target precision ({precision}). No change needed.")
            return default_decode()

        print(f"Advanced VAEDecode: Temporarily casting VAE from {original_dtype} to {target_dtype} for decoding.")
        
        try:
            model_to_modify.to(dtype=target_dtype)
            images = vae.decode(samples["samples"])
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        finally:
            # Гарантированно возвращаем модель в исходное состояние
            model_to_modify.to(dtype=original_dtype)
            print(f"Advanced VAEDecode: VAE restored to original precision ({original_dtype}).")

        return (images,)


class VAEEncodeWithPrecision:
    """
    Расширенная нода VAEEncode, позволяющая вручную выбирать точность (precision)
    для операции кодирования. Версия 3, исправленная.
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
        def default_encode():
            return ({"samples": vae.encode(pixels[:,:,:,:3])},)

        if precision == "auto":
            print("Advanced VAEEncode: Precision 'auto', using default VAE behavior.")
            return default_encode()

        if precision == "fp32": target_dtype = torch.float32
        elif precision == "fp16": target_dtype = torch.float16
        elif precision == "bf16": target_dtype = torch.bfloat16
        else:
            print(f"Warning: Unknown precision '{precision}', falling back to auto.")
            return default_encode()

        model_to_modify = vae.first_stage_model
        
        try:
            original_dtype = next(model_to_modify.parameters()).dtype
        except StopIteration:
            print("Warning: VAE model has no parameters. Using default behavior.")
            return default_encode()

        if original_dtype == target_dtype:
            print(f"Advanced VAEEncode: VAE already in target precision ({precision}). No change needed.")
            return default_encode()
        
        print(f"Advanced VAEEncode: Temporarily casting VAE from {original_dtype} to {target_dtype} for encoding.")
        
        try:
            model_to_modify.to(dtype=target_dtype)
            latent = vae.encode(pixels[:,:,:,:3])
        finally:
            model_to_modify.to(dtype=original_dtype)
            print(f"Advanced VAEEncode: VAE restored to original precision ({original_dtype}).")

        return ({"samples": latent},)
