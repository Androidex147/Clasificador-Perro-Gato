import os
# Esto obliga a TensorFlow a usar el modo compatibilidad con versiones antiguas
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
from tensorflow import keras  
from PIL import Image, ImageOps  
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Reconocimiento Perros vs Gatos", page_icon="🐾")

st.title("🐶 Detector de Mascotas 🐱")
st.write("Usa la cámara para saber si es un perro o un gato.")

# DEFINIMOS UNA FUNCIÓN PARA CARGAR EL MODELO Y GUARDARLO EN CACHE
@st.cache_resource
def carga_modelo():
    modelo = keras.models.load_model("app/keras_model.h5", compile=False)
    with open("app/labels.txt", "r") as f:
        clases = [line.strip() for line in f]  # Eliminar saltos de línea
    return modelo, clases

# 1. CARGAMOS EL MODELO Y LAS ETIQUETAS
try:
    mi_modelo, nombre_clases = carga_modelo()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# 2. TOMAMOS IMAGEN DESDE LA CÁMARA
imagen_camara = st.camera_input("Haz una foto")

# 3. PREDICCIÓN
if imagen_camara is not None:
    imagen = Image.open(imagen_camara).convert("RGB")
    imagen = ImageOps.fit(imagen, (224, 224), Image.Resampling.LANCZOS)
    imagen_array = np.asarray(imagen)
    normalizada_imagen_array = (imagen_array.astype(np.float32) / 127.5) - 1

    # Crear un batch con una sola imagen
    lote_imagenes = np.expand_dims(normalizada_imagen_array, axis=0)

    # Predicción
    resultados = mi_modelo.predict(lote_imagenes)
    indice = np.argmax(resultados[0])
    etiqueta = nombre_clases[indice]
    probabilidad = resultados[0][indice]

    st.divider()  # Línea separadora visual

    if "Perro" in etiqueta:
        st.success(f"¡Es un **PERRO**! 🐶")
        st.balloons()
    else:
        st.success(f"¡Es un **GATO**! 🐱")
        st.snow()

    st.write(f"Estoy un {probabilidad:.2%} seguro.")
