import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Deshabilita GPU para TensorFlow

import tensorflow as tf
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import requests
import pickle

# 📌 URL del modelo en tu repositorio de GitHub (cambia esto con tu enlace real)
MODEL_URL = "https://raw.githubusercontent.com/TU_USUARIO/TU_REPOSITORIO/main/modelo_baloto.h5"
SCALER_URL = "https://raw.githubusercontent.com/TU_USUARIO/TU_REPOSITORIO/main/scaler_baloto.pkl"

@st.cache_resource
def cargar_modelo():
    # Descargar el modelo si no está en la carpeta
    if not os.path.exists("modelo_baloto.h5"):
        st.write("Descargando modelo...")
        r = requests.get(MODEL_URL)
        with open("modelo_baloto.h5", "wb") as f:
            f.write(r.content)

    # Descargar el scaler si no está en la carpeta
    if not os.path.exists("scaler_baloto.pkl"):
        st.write("Descargando scaler...")
        r = requests.get(SCALER_URL)
        with open("scaler_baloto.pkl", "wb") as f:
            f.write(r.content)

    modelo = load_model("modelo_baloto.h5", compile=False)  # Carga sin compilar
    
    # Cargar el scaler con Pickle y manejar posibles errores
    try:
        with open("scaler_baloto.pkl", "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error(f"Error al cargar el scaler: {e}")
        st.stop()

    return modelo, scaler

st.title("Predicción del Baloto")

# 📌 Subir archivo de datos históricos
file_path = st.file_uploader("Sube el archivo de resultados históricos", type=["txt", "csv"])

if file_path:
    data = pd.read_csv(file_path, header=None, names=["Fecha", "N1", "N2", "N3", "N4", "N5", "N6"])
    data["Fecha"] = pd.to_datetime(data["Fecha"], format="%d/%m/%Y")
    data["Dia_Semana"] = data["Fecha"].dt.day_name()

    # 📌 Filtrar sorteos de miércoles
    data = data[data["Dia_Semana"] == "Wednesday"].drop(columns=["Dia_Semana"])
    data = data.sort_values(by="Fecha").reset_index(drop=True)

    # 📌 Cargar el modelo y scaler
    modelo, scaler = cargar_modelo()

    # 📌 Última combinación ganadora para predecir la siguiente
    ultima_entrada = scaler.transform(data.iloc[-1, 1:].values.reshape(1, -1))
    prediccion_scaled = modelo.predict(ultima_entrada)
    prediccion_final = scaler.inverse_transform(prediccion_scaled).astype(int)

    # 📌 Mostrar la predicción
    st.write("Números Predichos para el Próximo Sorteo:")
    st.write(prediccion_final)

    # 📌 Guardar en un archivo Excel
    prediccion_df = pd.DataFrame(prediccion_final, columns=["N1", "N2", "N3", "N4", "N5", "N6"])
    prediccion_df.insert(0, "Fecha_Predicha", "Próximo Sorteo")
    output_path = "Prediccion_Baloto.xlsx"
    prediccion_df.to_excel(output_path, index=False)

    st.download_button("Descargar Predicción", data=open(output_path, "rb").read(), file_name="Prediccion_Baloto.xlsx")
    output_path = "Prediccion_Baloto.xlsx"
    prediccion_df.to_excel(output_path, index=False)

    st.download_button("Descargar Predicción", data=open(output_path, "rb").read(), file_name="Prediccion_Baloto.xlsx")
