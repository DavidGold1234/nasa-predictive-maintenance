import pandas as pd


def transform_data(df):

    print("Transformando datos...")

    # eliminar columnas vacías
    df = df.loc[:, df.notna().all(axis=0)]

    # nombres oficiales NASA
    column_names = (
        ["engine_id", "cycle"] +
        [f"op_setting_{i}" for i in range(1, 4)] +
        [f"sensor_{i}" for i in range(1, 22)]
    )

    df.columns = column_names + ["dataset_id"]

    # eliminar sensores constantes
    constant_sensors = [
        col for col in df.columns
        if col.startswith("sensor_") and df[col].std() == 0
    ]

    df.drop(columns=constant_sensors, inplace=True)

    print("Sensores eliminados:", constant_sensors)

    # crear número de motor real
    df["engine_number"] = df["engine_id"].str.split("_").str[1].astype(int)

    # ordenar correctamente
    df = df.sort_values(["dataset_id", "engine_number", "cycle"])
    df.reset_index(drop=True, inplace=True)

    # eliminar columna auxiliar
    df.drop(columns="engine_number", inplace=True)

    print("Motores únicos:", df["engine_id"].nunique())

    return df



