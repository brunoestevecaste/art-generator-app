import streamlit as st
from PIL import Image
import random
from io import BytesIO # Necesario para la descarga sin guardar en disco
from src.styles import ESTILOS_ARTISTICOS
from src.generator import ArtGenerator

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="IA Art Studio",
    page_icon="🎨",
    layout="wide"
)

# Estilos CSS personalizados para mejorar la apariencia (opcional)
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- TÍTULO ---
st.title("🎨 IA Art Studio: De Foto a Obra de Arte")
st.markdown("Sube una imagen, elige un estilo (o un pintor famoso) y deja que la IA la redibuje.")

# --- SIDEBAR: CONFIGURACIÓN ---
st.sidebar.header("🎛️ Panel de Control")

# 1. Subida de archivo
uploaded_file = st.sidebar.file_uploader("1. Sube tu imagen", type=["jpg", "jpeg", "png"])

# 2. Selección de Estilo
st.sidebar.markdown("---")
modo_seleccion = st.sidebar.radio("2. Modo de Estilo", ["Aleatorio 🎲", "Selección Manual 🖐️"])

estilo_seleccionado = None
nombre_estilo_display = "Aleatorio"

if modo_seleccion == "Selección Manual 🖐️":
    # Ordenamos los estilos alfabéticamente para facilitar la búsqueda
    lista_estilos = sorted(list(ESTILOS_ARTISTICOS.keys()))
    nombre_estilo_display = st.sidebar.selectbox("Elige un estilo o artista:", lista_estilos)
    estilo_seleccionado = (nombre_estilo_display, ESTILOS_ARTISTICOS[nombre_estilo_display])

# --- ÁREA PRINCIPAL ---
col1, col2 = st.columns(2)

# Variable para guardar la imagen en el estado de la sesión (para que no desaparezca al tocar algo)
if "imagen_generada" not in st.session_state:
    st.session_state.imagen_generada = None
if "nombre_estilo_generado" not in st.session_state:
    st.session_state.nombre_estilo_generado = ""

if uploaded_file is not None:
    # Cargar y mostrar imagen original
    image_input = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("📸 Imagen Original")
        st.image(image_input, use_container_width=True)

    # Botón de acción
    st.sidebar.markdown("---")
    if st.sidebar.button("✨ GENERAR ARTE ✨", type="primary"):
        generator = ArtGenerator()
        
        # Barra de progreso y status
        with st.status("👩‍🎨 La IA está trabajando...", expanded=True) as status:
            
            # Paso 1: Análisis (BLIP)
            st.write("👁️ Analizando composición de la imagen...")
            descripcion_base = generator.generar_caption(image_input)
            st.write(f"📝 Descripción detectada: *{descripcion_base}*")
            
            # Paso 2: Configurar Prompt
            if modo_seleccion == "Aleatorio 🎲":
                nombre_estilo, modificadores = random.choice(list(ESTILOS_ARTISTICOS.items()))
            else:
                nombre_estilo, modificadores = estilo_seleccionado
            
            prompt_final = f"{descripcion_base}{modificadores}"
            st.session_state.nombre_estilo_generado = nombre_estilo
            
            # Paso 3: Generación (SDXL)
            st.write(f"🎨 Pintando al estilo: **{nombre_estilo}**...")
            imagen_resultado = generator.generar_imagen(prompt_final)
            
            # Guardar en sesión
            st.session_state.imagen_generada = imagen_resultado
            
            status.update(label="¡Obra terminada!", state="complete", expanded=False)

    # --- MOSTRAR RESULTADO Y DESCARGA ---
    if st.session_state.imagen_generada is not None:
        with col2:
            st.subheader(f"🎨 Resultado: {st.session_state.nombre_estilo_generado}")
            st.image(st.session_state.imagen_generada, use_container_width=True)
            
            # --- LÓGICA DE DESCARGA ---
            # Convertimos la imagen de PIL a Bytes en memoria
            buf = BytesIO()
            st.session_state.imagen_generada.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            # Botón de descarga
            st.download_button(
                label="⬇️ Descargar Obra de Arte (HD)",
                data=byte_im,
                file_name=f"arte_{st.session_state.nombre_estilo_generado.replace(' ', '_')}.png",
                mime="image/png"
            )

else:
    # Mensaje de bienvenida cuando no hay imagen
    with col1:
        st.info("👈 Para empezar, sube una imagen en el menú de la izquierda.")
    with col2:
        st.empty()