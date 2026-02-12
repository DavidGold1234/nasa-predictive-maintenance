import pandas as pd
from pathlib import Path

DATA_RAW_PATH = Path("data/raw")

def extract_data():

    print("Extrayendo datos NASA...")
    print("Ruta buscada:", DATA_RAW_PATH.resolve())

    files = [
        "train_FD001.txt",
        "train_FD002.txt",
        "train_FD003.txt",
        "train_FD004.txt"
    ]

    dataframes = []

    for file in files:

        dataset_name = file.replace("train_", "").replace(".txt", "")

        df = pd.read_csv(
            DATA_RAW_PATH / file,
            sep=r"\s+",
            header=None
        )

        # agregar dataset
        df["dataset_id"] = dataset_name

        # CREAR ENGINE_ID ÚNICO 
        df[0] = dataset_name + "_" + df[0].astype(str)

        dataframes.append(df)

    full_df = pd.concat(dataframes, ignore_index=True)

    print("Datasets unificados correctamente")

    return full_df
