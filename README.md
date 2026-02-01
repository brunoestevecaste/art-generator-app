# 🎨 AI Art Transformer (Local SDXL App)

Esta aplicación transforma cualquier imagen subida por el usuario en una pieza artística única utilizando Inteligencia Artificial Generativa.

El flujo de trabajo combina dos modelos potentes:
1.  **BLIP (Salesforce):** Analiza la imagen subida y genera una descripción textual (caption).
2.  **Stable Diffusion XL (SDXL 1.0):** Toma esa descripción + un estilo artístico (seleccionado o aleatorio) y genera una imagen de alta fidelidad (1024x1024).

## 🚀 Características

* **Interfaz Gráfica:** Construida con Streamlit para un uso sencillo.
* **Prompt Engineering Automatizado:** Convierte imágenes en prompts complejos automáticamente.
* **Optimización Local:** Configurado para correr en **Apple Silicon (M1/M2/M3)** usando `MPS` y gestión eficiente de memoria RAM.
* **Estilos Diversos:** Soporta desde Óleo y Acuarela hasta Cyberpunk y Vaporwave.

## 🛠️ Instalación y Uso

### Prerrequisitos
* Python 3.10+
* Se recomienda un entorno virtual (venv o conda).
* Hardware recomendado: Mac M1/M2 con 16GB RAM (funciona en 8GB pero más lento).

### Pasos

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/ai-art-transformer.git](https://github.com/tu-usuario/ai-art-transformer.git)
    cd ai-art-transformer
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Ejecutar la aplicación:**
    ```bash
    streamlit run app.py
    ```

## ⚙️ Notas Técnicas
La primera vez que ejecutes la app, descargará los modelos (aprox. 6-8 GB) de HuggingFace. Esto puede tardar varios minutos dependiendo de tu conexión.

El código incluye `gc.collect()` y limpieza de caché MPS/CUDA para evitar desbordamientos de memoria al cambiar entre el modelo de análisis (BLIP) y el generador (SDXL).

## 📄 Licencia
Este proyecto es para fines educativos y de portfolio.