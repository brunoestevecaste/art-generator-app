import streamlit as st
from PIL import Image
import random
from src.styles import ESTILOS_ARTISTICOS
from src.generator import ArtGenerator

# Configuración de la página
st.set_page_config(
    page_title="AI Art Transformer",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 AI Art Transformer")
st.markdown("Sube una foto y transformala en una obra de arte usando **BLIP + SDXL**.")

# Sidebar
st.sidebar.header("Configuración")
modo_seleccion = st.sidebar.radio("Selección de Estilo", ["Aleatorio", "Manual"])

estilo_seleccionado = None
nombre_estilo_display = "Aleatorio"

if modo_seleccion == "Manual":
    nombre_estilo_display = st.sidebar.selectbox("Elige un estilo", list(ESTILOS_ARTISTICOS.keys()))
    estilo_seleccionado = (nombre_estilo_display, ESTILOS_ARTISTICOS[nombre_estilo_display])

uploaded_file = st.sidebar.file_uploader("Sube tu imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])

# Layout principal
col1, col2 = st.columns(2)

if uploaded_file is not None:
    # Mostrar imagen original
    image_input = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("Imagen Original")
        st.image(image_input, use_container_width=True)

    # Botón de generar
    if st.sidebar.button("✨ Generar Arte"):
        generator = ArtGenerator()
        
        with st.status("Analizando imagen con IA...", expanded=True) as status:
            # Paso 1: Captioning
            st.write("🧠 BLIP: Entendiendo el contenido de la imagen...")
            descripcion_base = generator.generar_caption(image_input)
            st.write(f"📝 Descripción base detectada: *{descripcion_base}*")
            
            # Paso 2: Selección de Estilo
            if modo_seleccion == "Aleatorio":
                nombre_estilo, modificadores = random.choice(list(ESTILOS_ARTISTICOS.items()))
            else:
                nombre_estilo, modificadores = estilo_seleccionado
            
            prompt_final = f"{descripcion_base}{modificadores}"
            st.write(f"🎨 Estilo aplicado: **{nombre_estilo}**")
            
            # Paso 3: Generación
            st.write("🖌️ SDXL: Pintando la nueva imagen (esto puede tardar unos segundos)...")
            imagen_resultado = generator.generar_imagen(prompt_final)
            
            status.update(label="¡Arte Generado!", state="complete", expanded=False)

        # Mostrar Resultado
        with col2:
            st.subheader(f"Resultado: {nombre_estilo}")
            st.image(imagen_resultado, use_container_width=True)
            
            # Botón de descarga
            # Guardar en buffer para descargar
            from io import BytesIO
            buf = BytesIO()
            imagen_resultado.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Descargar Obra de Arte",
                data=byte_im,
                file_name="arte_generado.png",
                mime="image/png"
            )

else:
    with col1:
        st.info("👈 Sube una imagen en el menú lateral para comenzar.")