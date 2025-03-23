import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 📌 PASO 1: Cargar y Preprocesar los Datos
file_path = "Baloto (COLOMBIA).txt"  # Asegúrate de subir este archivo en Colab

# Leer el archivo TXT y convertirlo en DataFrame
data = pd.read_csv(file_path, header=None, names=["Fecha", "N1", "N2", "N3", "N4", "N5", "N6"])

# Convertir la columna Fecha a tipo datetime
data["Fecha"] = pd.to_datetime(data["Fecha"], format="%d/%m/%Y")

# Filtrar solo los sorteos de los miércoles
data["Dia_Semana"] = data["Fecha"].dt.day_name()
data_miercoles = data[data["Dia_Semana"] == "Wednesday"].drop(columns=["Dia_Semana"])

# Ordenar los datos por fecha
data_miercoles = data_miercoles.sort_values(by="Fecha").reset_index(drop=True)

# 📌 PASO 2: Preparar los datos para la red neuronal
# Definir las entradas (X) y salidas (Y)
X = data_miercoles[["N1", "N2", "N3", "N4", "N5", "N6"]].values[:-1]  # Todos menos el último sorteo
Y = data_miercoles[["N1", "N2", "N3", "N4", "N5", "N6"]].values[1:]   # La siguiente combinación ganadora

# Normalizar los datos entre 0 y 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.transform(Y)

# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42, shuffle=False)

# 📌 PASO 3: Construcción de la Red Neuronal
model = Sequential([
    Dense(64, activation="relu", input_shape=(6,)),  # Capa oculta con 64 neuronas y ReLU
    Dense(64, activation="relu"),  # Otra capa oculta
    Dense(6, activation="linear")  # Capa de salida con 6 neuronas (valores numéricos)
])

# Compilar el modelo
model.compile(optimizer="adam", loss="mse")

# Entrenar el modelo con 5000 épocas
model.fit(X_train, Y_train, epochs=5000, batch_size=16, verbose=1, validation_data=(X_test, Y_test))

# 📌 PASO 4: Predicción para el Miércoles 26/03/2025
# Tomamos el último sorteo para predecir el siguiente
ultima_entrada = X_scaled[-1].reshape(1, -1)
prediccion_scaled = model.predict(ultima_entrada)

# Desnormalizar la predicción
prediccion_final = scaler.inverse_transform(prediccion_scaled).astype(int)

# Guardar los resultados en un archivo Excel
prediccion_df = pd.DataFrame(prediccion_final, columns=["N1", "N2", "N3", "N4", "N5", "N6"])
prediccion_df.insert(0, "Fecha_Predicha", "26/03/2025")

# Guardar en Excel
output_path = "Prediccion_Baloto_26_03_2025.xlsx"
prediccion_df.to_excel(output_path, index=False)

print(f"Predicción guardada en: {output_path}")
