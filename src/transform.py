# src/transform.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def transform_data(df):
    print("Transformando datos...")

    # Eliminar columnas vacías (NASA trae columnas basura)
    df = df.loc[:, df.notna().all(axis=0)]

    # Definir nombres oficiales NASA
    column_names = (
        ["engine_id", "cycle"] +
        [f"op_setting_{i}" for i in range(1, 4)] +
        [f"sensor_{i}" for i in range(1, 22)]
    )

    df.columns = column_names + ["dataset_id"]

    # Eliminar sensores constantes (sin información)
    constant_sensors = [
        col for col in df.columns
        if col.startswith("sensor_") and df[col].std() == 0
    ]
    df.drop(columns=constant_sensors, inplace=True)

    # Normalización (solo sensores)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    scaler = StandardScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])

    return df
