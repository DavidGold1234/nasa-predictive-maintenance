import pandas as pd
from utils import load_features
from windowing import create_sliding_windows

print("Cargando dataset procesado...")

df = pd.read_csv("data/processed/nasa_train_all_fd_clean.csv")

features = load_features()

print("Features usadas:", features)

# Ordenar correctamente
df = df.sort_values(["engine_id", "cycle"])

# Crear ventanas
X = create_sliding_windows(df, features, window_size=30)

print("Shape ventanas:", X.shape)
