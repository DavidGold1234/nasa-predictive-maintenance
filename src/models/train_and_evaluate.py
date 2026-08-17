from __future__ import annotations

import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import IsolationForest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# =========================================================
# CONFIG
# =========================================================
SEED = int(os.environ.get("SEED", "42"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HDFS_GOLD_TRAIN = os.environ.get(
    "HDFS_GOLD_TRAIN",
    "hdfs://namenode:9000/user/root/nasa/gold/nasa_train_gold",
)
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "/data/models/artifacts")
HDFS_ARTIFACTS_ACTIVE = os.environ.get(
    "HDFS_ARTIFACTS_ACTIVE",
    "hdfs://namenode:9000/user/root/nasa/artifacts/active",
)
HDFS_ARTIFACTS_VERSIONS = os.environ.get(
    "HDFS_ARTIFACTS_VERSIONS",
    "hdfs://namenode:9000/user/root/nasa/artifacts/versions",
)

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "tesis")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
POSTGRES_SCHEMA = os.environ.get("POSTGRES_SCHEMA", "public")
POSTGRES_IF_EXISTS = os.environ.get("POSTGRES_IF_EXISTS", "append")

RUN_ID = os.environ.get("RUN_ID", datetime.now(timezone.utc).strftime("nasa_train_%Y%m%d_%H%M%S"))

WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "30"))
# Feature de dinámica transitoria calculada en run_preprocessing.py: cuántos
# ciclos lleva el motor en el régimen operativo actual. Se agrega como columna
# de entrada extra a los modelos (no es un sensor físico), para que puedan
# distinguir "acabo de cambiar de régimen" de "degradación real".
DURATION_COL = "cycles_since_regime_change"
EPOCHS = int(os.environ.get("EPOCHS", "30"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
PATIENCE = int(os.environ.get("PATIENCE", "6"))
VAL_ENGINE_RATIO = float(os.environ.get("VAL_ENGINE_RATIO", "0.20"))

LSTM_MAX_WINDOWS = int(os.environ.get("LSTM_MAX_WINDOWS", "50000"))
TCN_MAX_WINDOWS = int(os.environ.get("TCN_MAX_WINDOWS", "50000"))
TRANSFORMER_MAX_WINDOWS = int(os.environ.get("TRANSFORMER_MAX_WINDOWS", "50000"))
# "auto" es el valor correcto cuando se entrena SOLO sobre datos sanos confirmados.
# Con contamination="auto" el IF usa el umbral teórico (-0.5) sin asumir anomalías en los datos sanos.
_if_cont = os.environ.get("IF_CONTAMINATION", "auto")
IF_CONTAMINATION: float | str = float(_if_cont) if _if_cont != "auto" else "auto"

EARLY_RUL_MIN = int(os.environ.get("EARLY_RUL_MIN", "90"))
CRITICAL_RUL_MAX = int(os.environ.get("CRITICAL_RUL_MAX", "30"))

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# =========================================================
# OPTIONAL WRITERS
# =========================================================
def write_to_postgres(df: pd.DataFrame, table_name: str) -> None:
    if df.empty:
        print(f"⚠️ {table_name}: dataframe vacío, no se escribe")
        return

    full_table_name = f"{POSTGRES_SCHEMA}.{table_name}" if POSTGRES_SCHEMA else table_name

    try:
        from sqlalchemy import create_engine  # type: ignore

        engine = create_engine(
            f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
        )
        df.to_sql(
            name=table_name,
            con=engine,
            schema=POSTGRES_SCHEMA or None,
            if_exists=POSTGRES_IF_EXISTS,
            index=False,
            method="multi",
            chunksize=1000,
        )
        print(f"✅ PostgreSQL vía SQLAlchemy -> {full_table_name} | rows={len(df)}")
        return
    except Exception as exc:
        # Python borra la variable de "except ... as X" al salir del bloque, así que
        # se copia a una variable normal para poder referenciarla en el mensaje de abajo.
        first_error = str(exc)
        print(f"ℹ️ SQLAlchemy no disponible o falló para {table_name}: {first_error}")

    try:
        from pyspark.sql import SparkSession  # type: ignore

        spark = (
            SparkSession.builder.appName(f"Write{table_name}")
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        jdbc_url = f"jdbc:postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
        jdbc_props = {
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "driver": "org.postgresql.Driver",
        }
        spark.createDataFrame(df).write.mode(POSTGRES_IF_EXISTS).jdbc(
            jdbc_url, full_table_name, properties=jdbc_props
        )
        print(f"✅ PostgreSQL vía Spark JDBC -> {full_table_name} | rows={len(df)}")
        return
    except Exception as second_error:
        raise RuntimeError(
            f"No pude escribir {table_name} a PostgreSQL ni por SQLAlchemy ni por Spark JDBC. "
            f"Errores: [{first_error}] | [{second_error}]"
        )

# =========================================================
# HELPERS
# =========================================================
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_sensor_cols(pdf: pd.DataFrame, artifact_dir: str) -> list[str]:
    cfg_path = os.path.join(artifact_dir, "selected_features.json")
    if os.path.exists(cfg_path):
        cfg = load_json(cfg_path)
        cols = [c for c in cfg.get("features", []) if c.startswith("sensor_") and c in pdf.columns]
        if cols:
            return cols
    return sorted([c for c in pdf.columns if c.startswith("sensor_")])


def load_gold_dataframe(path: str) -> pd.DataFrame:
    from pyspark.sql import SparkSession  # type: ignore

    spark = (
        SparkSession.builder.appName("TrainAndEvaluateFromGold")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    pdf = spark.read.parquet(path).toPandas()
    spark.stop()
    return pdf


def run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Comando falló: {' '.join(cmd)}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )


def build_spark(app_name: str):
    from pyspark.sql import SparkSession  # type: ignore

    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def hdfs_put_via_spark(spark, local_path: str, hdfs_dir: str, filename: str) -> str:
    jvm = spark._jvm
    conf = spark._jsc.hadoopConfiguration()
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(jvm.java.net.URI.create(hdfs_dir), conf)
    fs.mkdirs(jvm.org.apache.hadoop.fs.Path(hdfs_dir))
    hdfs_target = f"{hdfs_dir.rstrip('/')}/{filename}"
    src = jvm.org.apache.hadoop.fs.Path(f"file://{os.path.abspath(local_path)}")
    dst = jvm.org.apache.hadoop.fs.Path(hdfs_target)
    fs.copyFromLocalFile(False, True, src, dst)
    return hdfs_target


def backup_artifacts_to_hdfs(
    spark,
    local_paths: Iterable[str],
    run_id: str,
    category: str = "models",
) -> dict[str, dict[str, str]]:
    active_dir = f"{HDFS_ARTIFACTS_ACTIVE.rstrip('/')}/{category}"
    version_dir = f"{HDFS_ARTIFACTS_VERSIONS.rstrip('/')}/{run_id}/{category}"
    manifest: dict[str, dict[str, str]] = {}

    for local_path in local_paths:
        filename = Path(local_path).name
        active_target = hdfs_put_via_spark(spark, local_path, active_dir, filename)
        version_target = hdfs_put_via_spark(spark, local_path, version_dir, filename)
        manifest[filename] = {
            "local": str(local_path),
            "hdfs_active": active_target,
            "hdfs_version": version_target,
        }
    return manifest


def sample_windows(X: np.ndarray, max_windows: int) -> np.ndarray:
    if max_windows <= 0 or len(X) <= max_windows:
        return X
    idx = np.random.choice(len(X), size=max_windows, replace=False)
    idx.sort()
    return X[idx]


def split_train_val_by_engine(meta: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    engines = np.array(sorted(meta["engine_id"].astype(str).unique()))
    if len(engines) <= 1:
        idx = np.arange(len(meta))
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * (1.0 - val_ratio)))
        return idx[:cut], idx[cut:]

    rng = np.random.default_rng(seed)
    n_val_eng = max(1, int(np.ceil(len(engines) * val_ratio)))
    val_engines = set(rng.choice(engines, size=n_val_eng, replace=False).tolist())
    mask_val = meta["engine_id"].astype(str).isin(val_engines).to_numpy()
    val_idx = np.where(mask_val)[0]
    train_idx = np.where(~mask_val)[0]

    if len(train_idx) == 0 or len(val_idx) == 0:
        idx = np.arange(len(meta))
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * (1.0 - val_ratio)))
        return idx[:cut], idx[cut:]

    return train_idx, val_idx


def scale_with_reference(ref_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    arr_ref = np.asarray(ref_scores, dtype=float)
    arr_scores = np.asarray(scores, dtype=float)
    lo = float(np.min(arr_ref))
    hi = float(np.max(arr_ref))
    if hi - lo < 1e-12:
        return np.zeros_like(arr_scores, dtype=float)
    return (arr_scores - lo) / (hi - lo)


def calibration_dict(ref_scores: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(ref_scores, dtype=float)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "p997": float(np.percentile(arr, 99.7)),
    }


def monotonicity_series(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    diffs = np.diff(values)
    pos = np.sum(diffs > 0)
    neg = np.sum(diffs < 0)
    den = max(1, len(diffs))
    return float((pos - neg) / den)


def build_windows(
    df: pd.DataFrame,
    sensor_cols: list[str],
    split_filter: str | None = None,
    require_healthy: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    pdf = df.copy()
    if split_filter is not None:
        pdf = pdf[pdf["split"] == split_filter].copy()

    windows = []
    meta_rows = []

    for (dataset_id, engine_id), g in pdf.groupby(["dataset_id", "engine_id"], sort=True):
        gg = g.sort_values(["cycle"]).reset_index(drop=True)
        if len(gg) < WINDOW_SIZE:
            continue

        engine_num = int(gg["engine_num"].iloc[0])
        max_cycle = int(gg["cycle"].max())
        for end_idx in range(WINDOW_SIZE - 1, len(gg)):
            w = gg.iloc[end_idx - WINDOW_SIZE + 1 : end_idx + 1]
            if require_healthy and int(w["is_healthy_train"].min()) != 1:
                continue

            end_row = w.iloc[-1]
            windows.append(w[sensor_cols].to_numpy(dtype=np.float32))
            meta_rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "engine_id": str(engine_id),
                    "engine_num": engine_num,
                    "cycle": int(end_row["cycle"]),
                    "split": str(end_row["split"]),
                    "is_healthy_train": int(end_row.get("is_healthy_train", 0)),
                    "operating_cluster": int(end_row.get("operating_cluster", -1)),
                    "true_rul": float(max_cycle - int(end_row["cycle"])),
                }
            )

    return np.asarray(windows, dtype=np.float32), pd.DataFrame(meta_rows)


def infer_ae_scores(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(DEVICE)
            recon = model(xb).cpu().numpy()
            mse = np.mean((X[i : i + batch_size] - recon) ** 2, axis=(1, 2))
            scores.append(mse)
    return np.concatenate(scores, axis=0) if scores else np.array([])


def group_internal_metrics(df: pd.DataFrame, score_col: str) -> Dict[str, float]:
    """
    Calcula 4 métricas clave por motor y devuelve el promedio.

    Métricas elegidas:
    - spearman: correlación del score con la degradación (-RUL). Indica si el modelo detecta la tendencia.
    - early_late_delta: diferencia de score entre zona sana (RUL alto) y zona crítica (RUL bajo).
                        Positivo = el modelo asigna scores más altos cerca del fallo. Valor clave.
    - monotonicity: fracción de ciclos donde el score sube. Mide consistencia de la señal.
    - n_windows: tamaño muestral total (para contexto).

    Fallback percentil: si un motor es demasiado corto para tener ventanas en la región
    EARLY_RUL_MIN o CRITICAL_RUL_MAX, se usan los percentiles 70% y 20% del RUL del propio motor.
    """
    frames = []
    for _, g in df.groupby("engine_id"):
        gg = g.sort_values("cycle").copy()
        if len(gg) < 3:
            continue

        # --- early region: RUL alto (motor sano) ---
        early = gg.loc[gg["true_rul"] >= EARLY_RUL_MIN, score_col]
        if len(early) == 0:
            # Motor corto: fallback al 30% superior de RUL de este motor
            rul_70pct = float(gg["true_rul"].quantile(0.70))
            early = gg.loc[gg["true_rul"] >= rul_70pct, score_col]

        # --- late region: RUL bajo (motor cerca del fallo) ---
        late = gg.loc[gg["true_rul"] <= CRITICAL_RUL_MAX, score_col]
        if len(late) == 0:
            # Motor corto: fallback al 20% inferior de RUL de este motor
            rul_20pct = float(gg["true_rul"].quantile(0.20))
            late = gg.loc[gg["true_rul"] <= rul_20pct, score_col]

        rho = spearmanr(-gg["true_rul"], gg[score_col]).correlation

        frames.append(
            {
                "spearman": 0.0 if pd.isna(rho) else float(rho),
                "early_late_delta": float(late.mean() - early.mean()) if len(early) and len(late) else np.nan,
                "monotonicity": monotonicity_series(gg[score_col].to_numpy()),
                "n_windows": int(len(gg)),
            }
        )

    if not frames:
        return {"spearman": np.nan, "early_late_delta": np.nan, "monotonicity": np.nan, "n_windows": 0}

    metrics_df = pd.DataFrame(frames)
    out = {c: float(metrics_df[c].mean()) for c in ["spearman", "early_late_delta", "monotonicity"]}
    out["n_windows"] = int(metrics_df["n_windows"].sum())
    return out

# =========================================================
# MODELS
# =========================================================
class LSTMAutoencoder(nn.Module):
    """
    Encoder bidireccional con mean-pooling + decoder LSTM.
    Mejoras respecto a la versión anterior:
    - Encoder bidireccional (2 capas): captura patrones temporales en ambas direcciones.
    - Mean-pooling sobre todos los timesteps: más robusto que solo el último estado oculto.
    - Proyección latente con capa oculta + Tanh: representación más expresiva.
    """
    def __init__(self, n_features: int, latent_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        # hidden_dim * 2 porque el encoder es bidireccional
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=n_features,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out, _ = self.encoder(x)                                          # [B, T, hidden*2]
        pooled = enc_out.mean(dim=1)                                           # [B, hidden*2]
        z = self.to_latent(pooled)                                             # [B, latent_dim]
        dec_in = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)     # [B, T, hidden]
        out, _ = self.decoder(dec_in)                                          # [B, T, n_features]
        return out


class Chomp1d(nn.Module):
    """Recorta el padding 'futuro' que Conv1d agrega simétrico, dejando solo
    la parte causal (pasado) — truco estándar de implementación de TCN."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TCNBlock(nn.Module):
    """Bloque TCN estándar (Bai et al. 2018): 2 convoluciones causales dilatadas
    + conexión residual."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNAutoencoder(nn.Module):
    """
    Autoencoder con Temporal Convolutional Network: convoluciones CAUSALES
    (cada paso solo puede ver el pasado, nunca el futuro) y DILATADAS
    (dilatación 1,2,4,8 con kernel=3 -> campo receptivo de 31 pasos, cubre
    toda la ventana de 30 ciclos de forma causal).

    Reemplaza al CNNAutoencoder anterior (aplanar + Linear): ese diseño ya
    tenía acceso a toda la ventana vía el aplanado, pero sin ninguna noción
    estructural de orden temporal — la capa Linear combina los 128 valores
    como un conjunto sin orden. El TCN respeta el orden y la causalidad por
    construcción, dando un sesgo inductivo secuencial real sin recurrencia
    (a diferencia del LSTM), lo que en teoría debería ayudar a capturar
    tendencias de degradación monótonas mejor que el CNN plano.
    """
    def __init__(self, n_features: int, hidden_channels: int = 16, latent_dim: int = 32, kernel_size: int = 3):
        super().__init__()
        dilations = [1, 2, 4, 8]

        enc_layers = []
        in_ch = n_features
        for d in dilations:
            enc_layers.append(TCNBlock(in_ch, hidden_channels, kernel_size, dilation=d))
            in_ch = hidden_channels
        self.encoder = nn.Sequential(*enc_layers)
        self.to_latent = nn.Linear(hidden_channels, latent_dim)

        self.from_latent = nn.Linear(latent_dim, hidden_channels)
        dec_layers = []
        in_ch = hidden_channels
        for d in dilations:
            dec_layers.append(TCNBlock(in_ch, hidden_channels, kernel_size, dilation=d))
            in_ch = hidden_channels
        self.decoder = nn.Sequential(*dec_layers)
        self.out_proj = nn.Conv1d(hidden_channels, n_features, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)                        # [B, n_features, T]
        enc_out = self.encoder(x)                     # [B, hidden, T]
        pooled = enc_out.mean(dim=2)                   # [B, hidden]
        latent = self.to_latent(pooled)                 # [B, latent_dim]  <- bottleneck real

        h = torch.relu(self.from_latent(latent))         # [B, hidden]
        h = h.unsqueeze(2).repeat(1, 1, x.size(2))         # [B, hidden, T]
        dec_out = self.decoder(h)                          # [B, hidden, T]
        out = self.out_proj(dec_out)                        # [B, n_features, T]
        return out.transpose(1, 2)


class PositionalEncoding(nn.Module):
    """Codificación posicional senoidal estándar (Vaswani et al. 2017)."""
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerAutoencoder(nn.Module):
    """
    Autoencoder basado en self-attention: en vez de recurrencia (LSTM) o
    convoluciones causales (TCN), usa atención global con codificación
    posicional explícita para relacionar cualquier par de ciclos de la
    ventana directamente, sin importar la distancia entre ellos. Es el
    cuarto modelo del ensemble (además de IsolationForest, LSTM_AE, TCN_AE),
    con un sesgo inductivo distinto a los otros tres.
    """
    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.to_latent = nn.Linear(d_model, latent_dim)

        self.from_latent = nn.Linear(latent_dim, d_model)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos_enc(self.input_proj(x))            # [B, T, d_model]
        enc_out = self.encoder(h)                         # [B, T, d_model]
        pooled = enc_out.mean(dim=1)                        # [B, d_model]
        latent = self.to_latent(pooled)                       # [B, latent_dim]  <- bottleneck real

        dec_in = self.from_latent(latent).unsqueeze(1).repeat(1, x.size(1), 1)  # [B, T, d_model]
        dec_in = self.pos_enc(dec_in)
        dec_out = self.decoder(dec_in)                          # [B, T, d_model]
        return self.out_proj(dec_out)                             # [B, T, n_features]


def train_autoencoder(
    model: nn.Module,
    X_train: np.ndarray,
    X_val: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    max_windows: int,
) -> Tuple[nn.Module, Dict[str, float]]:
    X_train_sub = sample_windows(X_train, max_windows)
    X_val_sub = sample_windows(X_val, max_windows)

    dl_train = DataLoader(TensorDataset(torch.tensor(X_train_sub, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    dl_val = None
    if len(X_val_sub) > 0:
        dl_val = DataLoader(TensorDataset(torch.tensor(X_val_sub, dtype=torch.float32)), batch_size=batch_size, shuffle=False)

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for (xb,) in dl_train:
            xb = xb.to(DEVICE)
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            # Gradient clipping: evita explosión de gradientes en LSTMs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        if dl_val is not None and len(X_val_sub) > 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for (xb,) in dl_val:
                    xb = xb.to(DEVICE)
                    recon = model(xb)
                    val_losses.append(criterion(recon, xb).item())
            val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        else:
            val_loss = train_loss

        print(f"Epoch {epoch + 1:02d}/{epochs} - train_loss={train_loss:.6f} - val_loss={val_loss:.6f}")
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch + 1
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"⏹️ Early stopping en epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val_loss": float(best_val), "best_epoch": int(best_epoch)}

# =========================================================
# MAIN
# =========================================================
def main() -> None:
    print("📥 Leyendo gold NASA ya transformado...")
    gold = load_gold_dataframe(HDFS_GOLD_TRAIN)
    if gold.empty:
        raise RuntimeError("nasa_train_gold está vacío")

    required = {"dataset_id", "engine_id", "engine_num", "cycle", "split", "is_healthy_train"}
    missing = [c for c in required if c not in gold.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en gold: {missing}")

    sensor_cols = choose_sensor_cols(gold, ARTIFACTS_DIR)
    if not sensor_cols:
        raise ValueError("No encontré sensores válidos en gold ni en selected_features.json")
    if DURATION_COL not in gold.columns:
        raise ValueError(f"Falta '{DURATION_COL}' en gold (generado por run_preprocessing.py)")
    model_input_cols = sensor_cols + [DURATION_COL]

    gold = gold.copy()
    gold["dataset_id"] = gold["dataset_id"].astype(str)
    gold["engine_id"] = gold["engine_id"].astype(str)
    gold["engine_num"] = pd.to_numeric(gold["engine_num"], errors="coerce")
    gold["cycle"] = pd.to_numeric(gold["cycle"], errors="coerce")
    gold["is_healthy_train"] = pd.to_numeric(gold["is_healthy_train"], errors="coerce").fillna(0).astype(int)
    if "operating_cluster" in gold.columns:
        gold["operating_cluster"] = pd.to_numeric(gold["operating_cluster"], errors="coerce").fillna(-1).astype(int)

    for c in model_input_cols:
        gold[c] = pd.to_numeric(gold[c], errors="coerce")

    gold = gold.dropna(subset=["engine_num", "cycle"] + model_input_cols).copy()
    gold["engine_num"] = gold["engine_num"].astype(int)
    gold["cycle"] = gold["cycle"].astype(int)
    gold = gold.sort_values(["dataset_id", "engine_num", "cycle"]).reset_index(drop=True)

    X_train, meta_train = build_windows(gold, model_input_cols, split_filter="train", require_healthy=False)
    X_test, meta_test = build_windows(gold, model_input_cols, split_filter="test", require_healthy=False)
    X_train_healthy, meta_train_healthy = build_windows(gold, model_input_cols, split_filter="train", require_healthy=True)
    if len(X_train) == 0 or len(X_test) == 0 or len(X_train_healthy) == 0:
        raise RuntimeError(f"No hay suficientes ventanas. train={len(X_train)} test={len(X_test)} healthy={len(X_train_healthy)}")

    print(f"✅ Ventanas train: {X_train.shape}")
    print(f"✅ Ventanas test: {X_test.shape}")
    print(f"✅ Ventanas train healthy: {X_train_healthy.shape}")

    healthy_train_idx, healthy_val_idx = split_train_val_by_engine(meta_train_healthy, VAL_ENGINE_RATIO, SEED)
    X_healthy_fit = X_train_healthy[healthy_train_idx]
    X_healthy_val = X_train_healthy[healthy_val_idx] if len(healthy_val_idx) else X_train_healthy[healthy_train_idx[: min(len(healthy_train_idx), 1024)]]

    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    X_healthy_fit_flat = X_healthy_fit.reshape(len(X_healthy_fit), -1)
    X_healthy_ref_flat = X_train_healthy.reshape(len(X_train_healthy), -1)

    print("\n=== Training Isolation Forest (healthy train only) ===")
    if_model = IsolationForest(n_estimators=300, contamination=IF_CONTAMINATION, random_state=SEED, n_jobs=-1)
    if_model.fit(X_healthy_fit_flat)

    print("\n=== Training LSTM Autoencoder (healthy train only) ===")
    lstm_ae = LSTMAutoencoder(n_features=len(model_input_cols))
    lstm_ae, lstm_hist = train_autoencoder(lstm_ae, X_healthy_fit, X_healthy_val, EPOCHS, BATCH_SIZE, LEARNING_RATE, PATIENCE, LSTM_MAX_WINDOWS)

    print("\n=== Training TCN Autoencoder (healthy train only) ===")
    tcn_ae = TCNAutoencoder(n_features=len(model_input_cols))
    tcn_ae, tcn_hist = train_autoencoder(tcn_ae, X_healthy_fit, X_healthy_val, EPOCHS, BATCH_SIZE, LEARNING_RATE, PATIENCE, TCN_MAX_WINDOWS)

    print("\n=== Training Transformer Autoencoder (healthy train only) ===")
    transformer_ae = TransformerAutoencoder(n_features=len(model_input_cols))
    transformer_ae, transformer_hist = train_autoencoder(
        transformer_ae, X_healthy_fit, X_healthy_val, EPOCHS, BATCH_SIZE, LEARNING_RATE, PATIENCE, TRANSFORMER_MAX_WINDOWS
    )

    print("\n=== Scoring internal ===")
    if_train_raw = -if_model.decision_function(X_train_flat)
    if_test_raw = -if_model.decision_function(X_test_flat)
    if_healthy_ref_raw = -if_model.decision_function(X_healthy_ref_flat)

    lstm_train_raw = infer_ae_scores(lstm_ae, X_train)
    lstm_test_raw = infer_ae_scores(lstm_ae, X_test)
    lstm_healthy_ref_raw = infer_ae_scores(lstm_ae, X_train_healthy)

    tcn_train_raw = infer_ae_scores(tcn_ae, X_train)
    tcn_test_raw = infer_ae_scores(tcn_ae, X_test)
    tcn_healthy_ref_raw = infer_ae_scores(tcn_ae, X_train_healthy)

    transformer_train_raw = infer_ae_scores(transformer_ae, X_train)
    transformer_test_raw = infer_ae_scores(transformer_ae, X_test)
    transformer_healthy_ref_raw = infer_ae_scores(transformer_ae, X_train_healthy)

    if_train_score = scale_with_reference(if_healthy_ref_raw, if_train_raw)
    if_test_score = scale_with_reference(if_healthy_ref_raw, if_test_raw)
    lstm_train_score = scale_with_reference(lstm_healthy_ref_raw, lstm_train_raw)
    lstm_test_score = scale_with_reference(lstm_healthy_ref_raw, lstm_test_raw)
    tcn_train_score = scale_with_reference(tcn_healthy_ref_raw, tcn_train_raw)
    tcn_test_score = scale_with_reference(tcn_healthy_ref_raw, tcn_test_raw)
    transformer_train_score = scale_with_reference(transformer_healthy_ref_raw, transformer_train_raw)
    transformer_test_score = scale_with_reference(transformer_healthy_ref_raw, transformer_test_raw)

    print("\n=== Guardando artifacts activos locales ===")
    local_if_path = os.path.join(ARTIFACTS_DIR, "isolation_forest.pkl")
    local_lstm_path = os.path.join(ARTIFACTS_DIR, "lstm_autoencoder.pt")
    local_tcn_path = os.path.join(ARTIFACTS_DIR, "tcn_autoencoder.pt")
    local_transformer_path = os.path.join(ARTIFACTS_DIR, "transformer_autoencoder.pt")
    local_calibration_path = os.path.join(ARTIFACTS_DIR, "score_calibration.json")
    local_model_meta_path = os.path.join(ARTIFACTS_DIR, "model_metadata.json")

    joblib.dump(if_model, local_if_path)
    torch.save(lstm_ae.state_dict(), local_lstm_path)
    torch.save(tcn_ae.state_dict(), local_tcn_path)
    torch.save(transformer_ae.state_dict(), local_transformer_path)

    score_calibration = {
        "run_id": RUN_ID,
        "reference": "healthy_train_windows",
        "IsolationForest": calibration_dict(if_healthy_ref_raw),
        "LSTM_AE": calibration_dict(lstm_healthy_ref_raw),
        "TCN_AE": calibration_dict(tcn_healthy_ref_raw),
        "Transformer_AE": calibration_dict(transformer_healthy_ref_raw),
    }
    with open(local_calibration_path, "w", encoding="utf-8") as f:
        json.dump(score_calibration, f, indent=2)

    model_metadata = {
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gold_input_path": HDFS_GOLD_TRAIN,
        "window_size": WINDOW_SIZE,
        "n_features": len(model_input_cols),
        "sensor_cols": sensor_cols,
        "duration_col": DURATION_COL,
        "model_input_cols": model_input_cols,
        "models": ["IsolationForest", "LSTM_AE", "TCN_AE", "Transformer_AE"],
        "device": DEVICE,
        "if_contamination": IF_CONTAMINATION,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        "val_engine_ratio": VAL_ENGINE_RATIO,
        "seed": SEED,
        "train_windows_full": int(len(X_train)),
        "test_windows_full": int(len(X_test)),
        "train_windows_healthy": int(len(X_train_healthy)),
        "healthy_fit_windows": int(len(X_healthy_fit)),
        "healthy_val_windows": int(len(X_healthy_val)),
        "lstm_best_val_loss": float(lstm_hist["best_val_loss"]),
        "tcn_best_val_loss": float(tcn_hist["best_val_loss"]),
        "transformer_best_val_loss": float(transformer_hist["best_val_loss"]),
        "lstm_best_epoch": int(lstm_hist["best_epoch"]),
        "tcn_best_epoch": int(tcn_hist["best_epoch"]),
        "transformer_best_epoch": int(transformer_hist["best_epoch"]),
        "hdfs_artifacts_active_root": HDFS_ARTIFACTS_ACTIVE,
        "hdfs_artifacts_versions_root": HDFS_ARTIFACTS_VERSIONS,
    }
    with open(local_model_meta_path, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    local_model_artifacts = [
        local_if_path, local_lstm_path, local_tcn_path, local_transformer_path,
        local_calibration_path, local_model_meta_path,
    ]
    print("☁️ Respaldando modelos/calibración en HDFS...")
    hdfs_spark = build_spark("BackupTrainArtifactsToHDFS")
    backup_manifest = backup_artifacts_to_hdfs(hdfs_spark, local_model_artifacts, RUN_ID, category="models")

    manifest_path = os.path.join(ARTIFACTS_DIR, "model_backup_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"run_id": RUN_ID, "files": backup_manifest}, f, indent=2)
    hdfs_put_via_spark(hdfs_spark, manifest_path, f"{HDFS_ARTIFACTS_VERSIONS.rstrip('/')}/{RUN_ID}/models", Path(manifest_path).name)
    hdfs_spark.stop()

    # Solo scores calibrados (0-1). Los raw_scores se guardan en score_calibration.json como referencia.
    train_pred = meta_train.copy()
    train_pred["run_id"] = RUN_ID
    train_pred["if_score"] = if_train_score
    train_pred["lstm_score"] = lstm_train_score
    train_pred["tcn_score"] = tcn_train_score
    train_pred["transformer_score"] = transformer_train_score

    test_pred = meta_test.copy()
    test_pred["run_id"] = RUN_ID
    test_pred["if_score"] = if_test_score
    test_pred["lstm_score"] = lstm_test_score
    test_pred["tcn_score"] = tcn_test_score
    test_pred["transformer_score"] = transformer_test_score

    all_pred = pd.concat([train_pred, test_pred], ignore_index=True).sort_values(["dataset_id", "engine_num", "cycle", "split"]).reset_index(drop=True)

    metrics_rows = []
    test_eval = test_pred.copy()
    for model_name, score_col in [
        ("IsolationForest", "if_score"),
        ("LSTM_AE", "lstm_score"),
        ("TCN_AE", "tcn_score"),
        ("Transformer_AE", "transformer_score"),
    ]:
        row = {"run_id": RUN_ID, "model": model_name, "score_reference": "healthy_train_windows"}
        row.update(group_internal_metrics(test_eval[["engine_id", "cycle", "true_rul", score_col]].copy(), score_col))
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    run_summary_df = pd.DataFrame([
        {
            "run_id": RUN_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "gold_input_path": HDFS_GOLD_TRAIN,
            "window_size": WINDOW_SIZE,
            "n_train_windows": int(len(X_train)),
            "n_test_windows": int(len(X_test)),
            "n_healthy_train_windows": int(len(X_train_healthy)),
            "n_train_engines": int(meta_train["engine_id"].nunique()),
            "n_test_engines": int(meta_test["engine_id"].nunique()),
            "sensor_count": len(sensor_cols),
            "score_reference": "healthy_train_windows",
        }
    ])

    print("\n=== Escribiendo resultados a PostgreSQL ===")
    write_to_postgres(all_pred, "eval_interna_predictions")
    write_to_postgres(metrics_df, "eval_interna_metrics")
    write_to_postgres(run_summary_df, "eval_interna_runs")

    print("\n✅ train_and_evaluate terminado")
    print(f"Artifacts activos locales: {ARTIFACTS_DIR}")
    print(f"Respaldo HDFS run_id: {RUN_ID}")
    print("\nResumen métricas internas:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
