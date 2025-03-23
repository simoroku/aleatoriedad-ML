import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model

# 📌 Cargar el modelo guardado
@st.cache_resource
def cargar_modelo():
    modelo = load_model("modelo_baloto.h5")
    scaler = joblib.load("scaler_baloto.pkl")
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
