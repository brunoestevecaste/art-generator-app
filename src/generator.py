import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import gc
import random

# Detectar dispositivo (MPS para Mac, CUDA para Nvidia, CPU fallback)
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()
DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32

class ArtGenerator:
    """Clase para manejar la generación de imágenes optimizada para recursos locales."""
    
    def __init__(self):
        self.device = DEVICE
        self.dtype = DTYPE

    def generar_caption(self, image_input):
        """Carga BLIP, genera descripción y libera memoria inmediatamente."""
        print(f"--- Cargando BLIP en {self.device} ---")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large", 
            torch_dtype=self.dtype
        ).to(self.device)

        inputs = processor(image_input, return_tensors="pt").to(self.device, self.dtype)
        
        # Generar
        out = model.generate(**inputs, max_new_tokens=70, min_new_tokens=20)
        descripcion = processor.decode(out[0], skip_special_tokens=True)
        
        # Limpieza agresiva de memoria
        del model
        del processor
        del inputs
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
            
        return descripcion

    def generar_imagen(self, prompt_completo):
        """Carga SDXL y genera la imagen."""
        print(f"--- Cargando SDXL en {self.device} ---")
        
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant="fp16"
        )
        
        # Optimizaciones críticas para Mac/Poca VRAM
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        
        if self.device == "mps":
            # En Mac con 8GB/16GB, esto ayuda mucho
            pipe.to("mps")
            pipe.enable_attention_slicing() 
        elif self.device == "cuda":
            pipe.enable_model_cpu_offload()
        
        print("--- Generando Imagen ---")
        image = pipe(
            prompt=prompt_completo,
            negative_prompt="low quality, bad quality, worst quality, blurry, pixelated, bad anatomy, ugly, watermark, text, signature, deformed",
            num_inference_steps=30, # 30 es suficiente para buena calidad y ahorra tiempo
            guidance_scale=7.5,
            height=1024,
            width=1024
        ).images[0]

        # Limpieza final (opcional si la app se cierra, pero buena práctica)
        del pipe
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()

        return image