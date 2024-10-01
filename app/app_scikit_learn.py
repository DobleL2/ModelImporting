import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import joblib
from PIL import Image

# Cargar el modelo entrenado desde el archivo .joblib
model = joblib.load('random_forest_mnist_model.joblib')

# Función para preprocesar la imagen dibujada
def preprocess_image(img):
    # Redimensionar la imagen a 28x28 píxeles
    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Invertir los colores si la imagen es negra sobre fondo blanco
    img_resized = 255 - img_resized
    
    # Aplanar la imagen a un array 1D y normalizarla
    img_resized = img_resized.flatten().astype('float32') / 255.0
    
    # Redimensionar a la forma esperada por el modelo (1, 784)
    img_resized = np.expand_dims(img_resized, axis=0)
    
    return img_resized

# Título de la aplicación
st.title("Reconocimiento de Dígitos Dibujados a Mano")

# Crear un lienzo interactivo para que el usuario dibuje un número
canvas_result = st_canvas(
    fill_color="white",  # Color de fondo del lienzo
    stroke_width=10,     # Grosor del trazo
    stroke_color="black", # Color del trazo
    background_color="white", # Color de fondo
    height=350,          # Altura del lienzo
    width=350,           # Ancho del lienzo
    drawing_mode="freedraw", # Modo de dibujo libre
    key="canvas",
)

# Si el usuario dibuja en la pizarra y presiona el botón de predicción
if st.button("Predecir"):
    if canvas_result.image_data is not None:
        # Convertir la imagen del lienzo a escala de grises
        img = canvas_result.image_data.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Preprocesar la imagen
        img_preprocessed = preprocess_image(img)
        
        # Hacer la predicción
        prediction = model.predict(img_preprocessed)
        
        # Mostrar el resultado de la predicción
        st.write(f"El modelo predice que el número es: {prediction[0]}")

# Mensaje para guiar al usuario
st.write("Dibuja un número (0-9) en la pizarra y haz clic en el botón 'Predecir'.")
