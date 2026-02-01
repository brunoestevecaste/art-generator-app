import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import gc
import os

# --- 1. CONFIGURACIÓN DEL DISPOSITIVO ---
def get_device_config():
    """Detecta el hardware y devuelve el dispositivo y el tipo de dato correcto."""
    if torch.backends.mps.is_available():
        return "mps", torch.float32 
    elif torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32

DEVICE, DTYPE = get_device_config()

class ArtGenerator:
    """Clase para manejar la generación de imágenes optimizada."""
    
    def __init__(self):
        self.device = DEVICE
        self.dtype = DTYPE

    def generar_caption(self, image_input):
        """Carga BLIP, genera descripción y libera memoria inmediatamente."""
        print(f"--- Cargando BLIP en {self.device} ({self.dtype}) ---")
        
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large", 
            torch_dtype=self.dtype
        ).to(self.device)

        # Procesar imagen
        inputs = processor(image_input, return_tensors="pt").to(self.device, self.dtype)
        
        # Generar texto
        out = model.generate(**inputs, max_new_tokens=70, min_new_tokens=20)
        descripcion = processor.decode(out[0], skip_special_tokens=True)
        
        # --- LIMPIEZA DE MEMORIA ---
        del model
        del processor
        del inputs
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
            
        return descripcion

    def generar_imagen(self, prompt_completo):
        """Carga SDXL y genera la imagen con corrección para Mac."""
        print(f"--- Cargando SDXL en {self.device} ---")
        
        # Cargar pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant="fp16" if self.dtype == torch.float16 else None 
            # Si es float32 (Mac), no forzamos la variante fp16 para evitar conflictos
        )
        
        # Configurar Scheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        
        # --- OPTIMIZACIONES DE MEMORIA ---
        if self.device == "mps":
            pipe.enable_sequential_cpu_offload()
            
            # Slicing ayuda a gestionar la atención con poca memoria
            pipe.enable_attention_slicing("max") 
            
            # Corrección explícita para el problema de imagen negra en VAE
            pipe.vae.to(dtype=torch.float32)
            
        elif self.device == "cuda":
            pipe.enable_model_cpu_offload()

        print("--- Generando Imagen (Paciencia, puede tardar un poco más en float32) ---")
        
        try:
            image = pipe(
                prompt=prompt_completo,
                negative_prompt="low quality, bad quality, worst quality, blurry, pixelated, bad anatomy, ugly, watermark, text, signature, deformed, nsfw",
                num_inference_steps=25, 
                guidance_scale=7.0,
                height=1024,
                width=1024
            ).images[0]
        except Exception as e:
            print(f"Error durante la inferencia: {e}")
            return None

        # Limpieza final
        del pipe
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()

        return image