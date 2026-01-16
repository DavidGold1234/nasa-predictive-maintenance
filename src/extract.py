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
        df = pd.read_csv(
            DATA_RAW_PATH / file,
            sep=r"\s+",
            header=None
        )
        df["dataset_id"] = file.replace("train_", "").replace(".txt", "")
        dataframes.append(df)

    full_df = pd.concat(dataframes, ignore_index=True)
    return full_df