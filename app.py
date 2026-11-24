import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(page_title="Recicjale Computacion en la nube", layout="centered")
st.title("Modelo predictivo Reciclaje clase de IA-UTH -2026-Marlon Saenz 202230010131")
st.write("Tome una foto con la cámara o suba una imagen para clasificar con el modelo MobileNetV2 pre entrenado")

IMG_SIZE=(224,224)
MODEL_DIR=Path("modelo_reciclaje_mobilenet")
CLASS_PATH=MODEL_DIR/"class_names.json"
MODEL_PATHS=[MODEL_DIR/"waste_mobilenet.keras", MODEL_DIR/"waste_mobilenet.h5"]

LABELS_ES= {
  "cardboard": "Cartón",
    "glass": "Vidrio",
    "metal": "Metal",
    "paper": "Papel",
    "plastic": "Plástico",
    "trash": "Basura",  
}
@st.cache_resource
def cargar_modelo():
    for path in MODEL_PATHS:
        if path.exists():
            return tf.keras.models.load_model(path, compile=False)
    st.error("No se encontró el modelo. Coloque la carpeta modelo_reciclaje_mobilenet junto a app.py.")
    st.stop()

@st.cache_data
def cargar_clases():
    if CLASS_PATH.exists():
        with open(CLASS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

def preparar_imagen(img):
    # Igual que en el entrenamiento: RGB, 224x224, pixeles 0-255.
    # NO se aplica preprocess_input aqui porque el modelo ya lo trae adentro.
    img = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

def predecir(img):
    preds = modelo.predict(preparar_imagen(img), verbose=0)[0]
    top3 = np.argsort(preds)[-3:][::-1]
    return [
        (LABELS_ES.get(clases[i], clases[i]), float(preds[i]) * 100)
        for i in top3
    ]
modelo = cargar_modelo()
clases = cargar_clases()

tab_camara, tab_archivo = st.tabs(["Cámara", "Subir imagen"])
with tab_camara:
    foto = st.camera_input("Tome una fotografía del residuo")
with tab_archivo:
    archivo_subido = st.file_uploader("Seleccione una imagen", type=["jpg", "jpeg", "png"])

archivo = foto if foto is not None else archivo_subido

if archivo:
    imagen = ImageOps.exif_transpose(Image.open(archivo))
    st.image(imagen, caption="Imagen capturada")

    resultados = predecir(imagen)
    st.subheader("Resultado")
    st.success(f"Predicción principal: {resultados[0][0]} ({resultados[0][1]:.2f}%)")

    st.write("Top 3 probabilidades:")
    for clase, prob in resultados:
        st.write(f"{clase}: {prob:.2f}%")
else:
    st.info("Tome una foto o cargue una imagen para iniciar la clasificación.")
