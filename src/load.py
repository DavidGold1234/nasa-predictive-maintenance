# src/load.py
from pathlib import Path

DATA_PROCESSED_PATH = Path("data/processed")

def load_data(df):
    print("Cargando datos procesados...")

    DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    output_file = DATA_PROCESSED_PATH / "nasa_train_all_fd_clean.csv"

    df.to_csv(output_file, index=False)
    print(f"Datos guardados en {output_file}")
