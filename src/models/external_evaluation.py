from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import torch
import torch.nn as nn


# =========================================================
# CONFIG
# =========================================================
HDFS_SILVER = os.environ.get("HDFS_SILVER", "hdfs://namenode:9000/user/root/nasa/silver")
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "/data/models/artifacts")

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "tesis")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
POSTGRES_SCHEMA = os.environ.get("POSTGRES_SCHEMA", "public")
POSTGRES_IF_EXISTS = os.environ.get("POSTGRES_IF_EXISTS", "append")

RUN_ID = os.environ.get("RUN_ID", datetime.now(timezone.utc).strftime("nasa_external_%Y%m%d_%H%M%S"))

WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "30"))
DATASETS = [d.strip() for d in os.environ.get("DATASETS", "FD001,FD002,FD003,FD004").split(",") if d.strip()]
DURATION_COL = "cycles_since_regime_change"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STABLE_RUL_MIN = int(os.environ.get("STABLE_RUL_MIN", "125"))
CRITICAL_RUL_MAX = int(os.environ.get("CRITICAL_RUL_MAX", "30"))
DETECTION_PERCENTILE = float(os.environ.get("DETECTION_PERCENTILE", "99.0"))


# =========================================================
# OPTIONAL WRITERS
# =========================================================
def write_to_postgres(df: pd.DataFrame, table_name: str) -> None:
    if df.empty:
        print(f"⚠️ {table_name}: dataframe vacío, no se escribe")
        return

    full_table_name = f"{POSTGRES_SCHEMA}.{table_name}" if POSTGRES_SCHEMA else table_name
    first_error_msg = None

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
        first_error_msg = str(exc)
        print(f"ℹ️ SQLAlchemy no disponible o falló para {table_name}: {first_error_msg}")

    try:
        from pyspark.sql import SparkSession  # type: ignore

        spark = (
            SparkSession.builder
            .appName(f"Write{table_name}")
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
            jdbc_url,
            full_table_name,
            properties=jdbc_props,
        )
        print(f"✅ PostgreSQL vía Spark JDBC -> {full_table_name} | rows={len(df)}")
        return
    except Exception as second_error:
        raise RuntimeError(
            f"No pude escribir {table_name} a PostgreSQL ni por SQLAlchemy ni por Spark JDBC. "
            f"Errores: [{first_error_msg}] | [{second_error}]"
        )


# =========================================================
# MODELS
# =========================================================
class LSTMAutoencoder(nn.Module):
    """Debe coincidir exactamente con la arquitectura de train_and_evaluate.py."""
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
        enc_out, _ = self.encoder(x)
        pooled = enc_out.mean(dim=1)
        z = self.to_latent(pooled)
        dec_in = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(dec_in)
        return out


class Chomp1d(nn.Module):
    """Debe coincidir exactamente con la arquitectura de train_and_evaluate.py."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TCNBlock(nn.Module):
    """Debe coincidir exactamente con la arquitectura de train_and_evaluate.py."""
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
    """Debe coincidir exactamente con la arquitectura de train_and_evaluate.py."""
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
        x = x.transpose(1, 2)
        enc_out = self.encoder(x)
        pooled = enc_out.mean(dim=2)
        latent = self.to_latent(pooled)

        h = torch.relu(self.from_latent(latent))
        h = h.unsqueeze(2).repeat(1, 1, x.size(2))
        dec_out = self.decoder(h)
        out = self.out_proj(dec_out)
        return out.transpose(1, 2)


class PositionalEncoding(nn.Module):
    """Debe coincidir exactamente con la arquitectura de train_and_evaluate.py."""
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerAutoencoder(nn.Module):
    """Debe coincidir exactamente con la arquitectura de train_and_evaluate.py."""
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
        h = self.pos_enc(self.input_proj(x))
        enc_out = self.encoder(h)
        pooled = enc_out.mean(dim=1)
        latent = self.to_latent(pooled)

        dec_in = self.from_latent(latent).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_in = self.pos_enc(dec_in)
        dec_out = self.decoder(dec_in)
        return self.out_proj(dec_out)


# =========================================================
# HELPERS
# =========================================================
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_torch_state_dict(path: str):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def load_spark_dataframe(path: str) -> pd.DataFrame:
    try:
        from pyspark.sql import SparkSession  # type: ignore
    except Exception as exc:
        raise RuntimeError("external_evaluation.py necesita pyspark para leer silver desde HDFS.") from exc

    spark = (
        SparkSession.builder.appName("ExternalEvaluationNASA")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark.read.parquet(path).toPandas()


def apply_regime_baseline(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    """Debe coincidir con la lógica de run_preprocessing.py / build_ros_gold.py."""
    sensor_cols = baseline["sensor_cols"]
    if baseline["regressor"] is not None:
        X = baseline["poly"].transform(df[baseline["op_cols"]].to_numpy())
        pred = baseline["regressor"].predict(X)
    else:
        pred = np.tile(baseline["flat_mean"], (len(df), 1))

    out = df.copy()
    out.loc[:, sensor_cols] = df[sensor_cols].to_numpy() - pred
    return out


def compute_cycles_since_regime_change(
    pdf: pd.DataFrame,
    op_cols: list[str],
    detector,
    duration_col: str,
) -> pd.DataFrame:
    """Debe coincidir con la lógica de run_preprocessing.py / build_ros_gold.py."""
    out = pdf.reset_index(drop=True).copy()
    out["_regime_label"] = detector.predict(out[op_cols])

    counts = np.zeros(len(out), dtype=np.float64)
    for _, g in out.groupby(["dataset_id", "engine_id"]):
        g_sorted = g.sort_values("cycle")
        labels = g_sorted["_regime_label"].to_numpy()
        idx = g_sorted.index.to_numpy()
        counter = 0
        g_counts = np.empty(len(labels), dtype=np.float64)
        for i in range(len(labels)):
            counter = 0 if (i == 0 or labels[i] != labels[i - 1]) else counter + 1
            g_counts[i] = counter
        counts[idx] = g_counts

    out[duration_col] = counts
    return out.drop(columns=["_regime_label"])


def preprocess_pdf(
    pdf: pd.DataFrame,
    sensor_cols: list[str],
    op_cols: list[str],
    regime_baseline: dict,
    healthy_global_scaler,
    regime_detector,
    duration_scaler,
    duration_col: str,
) -> pd.DataFrame:
    out = pdf.copy()

    needed = ["dataset_id", "engine_id", "engine_num", "cycle"] + op_cols + sensor_cols
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en externa: {missing}")

    out = out.dropna(subset=needed).copy()
    out["engine_num"] = pd.to_numeric(out["engine_num"], errors="coerce")
    out["cycle"] = pd.to_numeric(out["cycle"], errors="coerce")
    out = out.dropna(subset=["engine_num", "cycle"]).copy()
    out["engine_num"] = out["engine_num"].astype(int)
    out["cycle"] = out["cycle"].astype(int)

    # Datos externos NASA: mismo dominio que el detector/línea base activos (no hace
    # falta adaptación cruzada como en ROS), se aplican directo.
    out = compute_cycles_since_regime_change(out, op_cols, regime_detector, duration_col)
    out[duration_col] = out[duration_col].clip(upper=WINDOW_SIZE).astype(float)

    out = apply_regime_baseline(out, regime_baseline)
    out = out.sort_values(["dataset_id", "engine_num", "cycle"]).reset_index(drop=True)
    out.loc[:, sensor_cols] = healthy_global_scaler.transform(out[sensor_cols])
    out.loc[:, [duration_col]] = duration_scaler.transform(out[[duration_col]])
    return out


def build_windows(
    df: pd.DataFrame,
    sensor_cols: list[str],
    rul_by_engine: Dict[int, float],
) -> Tuple[np.ndarray, pd.DataFrame]:
    windows = []
    rows = []

    for (dataset_id, engine_id), g in df.groupby(["dataset_id", "engine_id"], sort=True):
        engine_num = int(g["engine_num"].iloc[0])

        if engine_num not in rul_by_engine:
            raise KeyError(f"No encontré RUL para engine_num={engine_num} ({engine_id})")

        gg = g.sort_values("cycle").reset_index(drop=True)
        if len(gg) < WINDOW_SIZE:
            continue

        end_rul = float(rul_by_engine[engine_num])
        max_cycle = int(gg["cycle"].max())
        gg["true_rul"] = (max_cycle + end_rul - gg["cycle"]).astype(float)

        vals = gg[sensor_cols].to_numpy(dtype=np.float32)
        cycles = gg["cycle"].to_numpy()

        for end_idx in range(WINDOW_SIZE - 1, len(gg)):
            windows.append(vals[end_idx - WINDOW_SIZE + 1 : end_idx + 1])
            rows.append(
                {
                    "run_id": RUN_ID,
                    "dataset_id": str(dataset_id),
                    "engine_id": str(engine_id),
                    "engine_num": engine_num,
                    "cycle": int(cycles[end_idx]),
                    "true_rul": float(gg.iloc[end_idx]["true_rul"]),
                }
            )

    return np.asarray(windows, dtype=np.float32), pd.DataFrame(rows)


def infer_ae_scores(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(DEVICE)
            recon = model(xb).cpu().numpy()
            mse = np.mean((X[i : i + batch_size] - recon) ** 2, axis=(1, 2))
            outs.append(mse)
    return np.concatenate(outs, axis=0) if outs else np.array([])


def monotonicity_series(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    diffs = np.diff(values)
    pos = np.sum(diffs > 0)
    neg = np.sum(diffs < 0)
    den = max(1, len(diffs))
    return float((pos - neg) / den)


def aggregate_dataset_metrics(df: pd.DataFrame, score_col: str, threshold: float) -> Dict[str, float]:
    """
    Calcula 4 métricas clave de evaluación externa con RUL como referencia:

    - spearman: correlación del score con -RUL. Mide si el modelo detecta la tendencia de degradación.
    - stable_alert_delta: diferencia de score entre zona estable (RUL>=125) y zona de alerta (30<RUL<125).
                          Positivo = el modelo distingue bien entre motor sano y motor degradándose.
    - detection_rate: % de motores donde se disparó alerta antes del fallo. Mide utilidad práctica.
    - earliest_detection_rul_mean: RUL promedio en que se disparó la primera alerta. Mayor es mejor
                                   (detección más anticipada).
    """
    stable = df.loc[df["true_rul"] >= STABLE_RUL_MIN, score_col]
    alert = df.loc[(df["true_rul"] < STABLE_RUL_MIN) & (df["true_rul"] > CRITICAL_RUL_MAX), score_col]

    rho = spearmanr(-df["true_rul"], df[score_col]).correlation

    detected_ruls = []
    for _, g in df.groupby("engine_id"):
        gg = g.sort_values("cycle")
        detected = gg[gg[score_col] >= threshold]
        if len(detected) > 0:
            detected_ruls.append(float(detected.iloc[0]["true_rul"]))

    return {
        "spearman": 0.0 if pd.isna(rho) else float(rho),
        "stable_alert_delta": float(alert.mean() - stable.mean()) if len(stable) and len(alert) else np.nan,
        "detection_rate": float(len(detected_ruls) / max(1, df["engine_id"].nunique())),
        "earliest_detection_rul_mean": float(np.mean(detected_ruls)) if detected_ruls else np.nan,
        "n_windows": int(len(df)),
        "n_engines": int(df["engine_id"].nunique()),
    }


def weighted_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Promedio ponderado de las 4 métricas clave a través de los 4 datasets."""
    rows = []

    for model, g in df.groupby("model"):
        n_windows = g["n_windows"].sum()
        n_engines = g["n_engines"].sum()
        det_weight = np.maximum(1e-9, g["detection_rate"] * g["n_engines"])

        row = {"run_id": RUN_ID, "model": model}
        for col in ["spearman", "stable_alert_delta"]:
            row[col] = float(np.average(g[col].fillna(0), weights=g["n_windows"])) if n_windows > 0 else np.nan

        row["detection_rate"] = (
            float(np.average(g["detection_rate"], weights=g["n_engines"])) if n_engines > 0 else np.nan
        )
        row["earliest_detection_rul_mean"] = (
            float(np.average(g["earliest_detection_rul_mean"].fillna(0), weights=det_weight))
            if det_weight.sum() > 0
            else np.nan
        )
        row["n_windows"] = int(n_windows)
        row["n_engines"] = int(n_engines)
        rows.append(row)

    return pd.DataFrame(rows)


def calibration_scale(stats: Optional[dict], scores: np.ndarray) -> np.ndarray:
    if not stats:
        return np.asarray(scores, dtype=float)

    lo = float(stats.get("min", np.nan))
    hi = float(stats.get("max", np.nan))
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        return np.zeros_like(scores, dtype=float)

    return (np.asarray(scores, dtype=float) - lo) / (hi - lo)


def percentile_key(percentile: float) -> str:
    rounded = round(percentile, 3)
    if abs(rounded - 95.0) < 1e-9:
        return "p95"
    if abs(rounded - 99.0) < 1e-9:
        return "p99"
    if abs(rounded - 99.7) < 1e-9:
        return "p997"
    return ""


def threshold_from_calibration(stats: Optional[dict], percentile: float) -> Optional[float]:
    if not stats:
        return None

    key = percentile_key(percentile)
    lo = float(stats.get("min", np.nan))
    hi = float(stats.get("max", np.nan))

    if key and key in stats and np.isfinite(lo) and np.isfinite(hi) and (hi - lo) >= 1e-12:
        raw_thr = float(stats[key])
        return (raw_thr - lo) / (hi - lo)

    return None


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    cfg = load_json(os.path.join(ARTIFACTS_DIR, "selected_features.json"))
    feature_cols = cfg["features"]
    sensor_cols = [c for c in feature_cols if c.startswith("sensor_")]
    op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
    model_input_cols = sensor_cols + [DURATION_COL]

    regime_baseline = joblib.load(os.path.join(ARTIFACTS_DIR, "regime_baseline.pkl"))
    healthy_global_scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "healthy_global_scaler.pkl"))
    regime_detector = joblib.load(os.path.join(ARTIFACTS_DIR, "regime_detector_kmeans.pkl"))
    duration_scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "regime_duration_scaler.pkl"))

    if_model = joblib.load(os.path.join(ARTIFACTS_DIR, "isolation_forest.pkl"))

    lstm_ae = LSTMAutoencoder(n_features=len(model_input_cols)).to(DEVICE)
    lstm_ae.load_state_dict(load_torch_state_dict(os.path.join(ARTIFACTS_DIR, "lstm_autoencoder.pt")))
    lstm_ae.eval()

    tcn_ae = TCNAutoencoder(n_features=len(model_input_cols)).to(DEVICE)
    tcn_ae.load_state_dict(load_torch_state_dict(os.path.join(ARTIFACTS_DIR, "tcn_autoencoder.pt")))
    tcn_ae.eval()

    transformer_ae = TransformerAutoencoder(n_features=len(model_input_cols)).to(DEVICE)
    transformer_ae.load_state_dict(load_torch_state_dict(os.path.join(ARTIFACTS_DIR, "transformer_autoencoder.pt")))
    transformer_ae.eval()

    score_calibration_path = os.path.join(ARTIFACTS_DIR, "score_calibration.json")
    score_calibration = load_json(score_calibration_path) if os.path.exists(score_calibration_path) else {}

    thresholds = {}
    for model_name in ["IsolationForest", "LSTM_AE", "TCN_AE", "Transformer_AE"]:
        thr = threshold_from_calibration(score_calibration.get(model_name), DETECTION_PERCENTILE)
        if thr is not None:
            thresholds[model_name] = float(thr)

    pred_frames = []
    summary_rows = []

    for dataset_id in DATASETS:
        print(f"\n=== {dataset_id} ===")

        silver_test = load_spark_dataframe(f"{HDFS_SILVER}/test_{dataset_id}")
        silver_rul = load_spark_dataframe(f"{HDFS_SILVER}/rul_{dataset_id}")

        if silver_test.empty:
            raise RuntimeError(f"{dataset_id}: silver/test está vacío")
        if silver_rul.empty:
            raise RuntimeError(f"{dataset_id}: silver/rul está vacío")

        silver_test = silver_test.copy()
        silver_test["dataset_id"] = silver_test["dataset_id"].astype(str)
        silver_test["engine_id"] = silver_test["engine_id"].astype(str)
        silver_test["engine_num"] = pd.to_numeric(silver_test["engine_num"], errors="coerce")
        silver_test["cycle"] = pd.to_numeric(silver_test["cycle"], errors="coerce")
        silver_test = silver_test.dropna(subset=["engine_num", "cycle"] + op_cols + sensor_cols).copy()
        silver_test["engine_num"] = silver_test["engine_num"].astype(int)
        silver_test["cycle"] = silver_test["cycle"].astype(int)
        silver_test = silver_test.sort_values(["dataset_id", "engine_num", "cycle"]).reset_index(drop=True)

        silver_rul = silver_rul.copy()
        silver_rul["dataset_id"] = silver_rul["dataset_id"].astype(str)
        silver_rul["engine_id"] = silver_rul["engine_id"].astype(str)
        silver_rul["engine_num"] = pd.to_numeric(silver_rul["engine_num"], errors="coerce")
        silver_rul["rul"] = pd.to_numeric(silver_rul["rul"], errors="coerce")
        silver_rul = silver_rul.dropna(subset=["engine_num", "rul"]).copy()
        silver_rul["engine_num"] = silver_rul["engine_num"].astype(int)
        silver_rul["rul"] = silver_rul["rul"].astype(int)
        silver_rul = silver_rul.sort_values(["dataset_id", "engine_num"]).reset_index(drop=True)

        print(
            f"{dataset_id} | silver_test engines={silver_test['engine_id'].nunique()} "
            f"| engine_num={silver_test['engine_num'].nunique()} | rows={len(silver_test)}"
        )
        print(
            f"{dataset_id} | silver_rul engines={silver_rul['engine_num'].nunique()} | rows={len(silver_rul)}"
        )

        pdf = preprocess_pdf(
            silver_test[["dataset_id", "engine_id", "engine_num", "cycle"] + op_cols + sensor_cols].copy(),
            sensor_cols=sensor_cols,
            op_cols=op_cols,
            regime_baseline=regime_baseline,
            healthy_global_scaler=healthy_global_scaler,
            regime_detector=regime_detector,
            duration_scaler=duration_scaler,
            duration_col=DURATION_COL,
        )

        rul_by_engine = silver_rul.set_index("engine_num")["rul"].astype(float).to_dict()

        X, meta = build_windows(pdf, model_input_cols, rul_by_engine)
        if len(X) == 0:
            print(f"⚠️ {dataset_id}: no se generaron ventanas")
            continue

        print(
            f"{dataset_id} | meta engines={meta['engine_id'].nunique()} "
            f"| windows={len(meta)}"
        )

        X_flat = X.reshape(len(X), -1)

        meta["if_raw_score"] = -if_model.decision_function(X_flat)
        meta["lstm_raw_score"] = infer_ae_scores(lstm_ae, X)
        meta["tcn_raw_score"] = infer_ae_scores(tcn_ae, X)
        meta["transformer_raw_score"] = infer_ae_scores(transformer_ae, X)

        meta["if_score"] = calibration_scale(score_calibration.get("IsolationForest"), meta["if_raw_score"].to_numpy())
        meta["lstm_score"] = calibration_scale(score_calibration.get("LSTM_AE"), meta["lstm_raw_score"].to_numpy())
        meta["tcn_score"] = calibration_scale(score_calibration.get("TCN_AE"), meta["tcn_raw_score"].to_numpy())
        meta["transformer_score"] = calibration_scale(score_calibration.get("Transformer_AE"), meta["transformer_raw_score"].to_numpy())
        meta["score_reference"] = score_calibration.get("reference", "healthy_train_windows")
        meta = meta.sort_values(["dataset_id", "engine_num", "cycle"]).reset_index(drop=True)

        pred_frames.append(meta.copy())

        for model_name, score_col in [
            ("IsolationForest", "if_score"),
            ("LSTM_AE", "lstm_score"),
            ("TCN_AE", "tcn_score"),
            ("Transformer_AE", "transformer_score"),
        ]:
            threshold = thresholds.get(model_name)
            if threshold is None:
                threshold = float(np.percentile(meta[score_col], DETECTION_PERCENTILE))

            row = {
                "run_id": RUN_ID,
                "dataset_id": dataset_id,
                "model": model_name,
                "threshold": float(threshold),
                "score_reference": score_calibration.get("reference", "healthy_train_windows"),
            }
            row.update(
                aggregate_dataset_metrics(
                    meta[["engine_id", "cycle", "true_rul", score_col]].copy(),
                    score_col,
                    float(threshold),
                )
            )
            summary_rows.append(row)

    if not pred_frames:
        raise RuntimeError("No se generaron predicciones externas para ningún dataset")

    pred_df = pd.concat(pred_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    weighted_df = weighted_summary(summary_df)
    runs_df = pd.DataFrame(
        [
            {
                "run_id": RUN_ID,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "datasets": ",".join(DATASETS),
                "window_size": WINDOW_SIZE,
                "detection_percentile": DETECTION_PERCENTILE,
                "score_reference": score_calibration.get("reference", "healthy_train_windows"),
            }
        ]
    )

    # Asegurar columnas mínimas antes de escribir
    pred_df["run_id"] = RUN_ID
    pred_df["score_reference"] = pred_df.get("score_reference", score_calibration.get("reference", "healthy_train_windows"))

    # Solo scores calibrados (0-1). Los raw scores están en score_calibration.json.
    pred_cols = [
        "run_id",
        "dataset_id",
        "engine_id",
        "engine_num",
        "cycle",
        "true_rul",
        "if_score",
        "lstm_score",
        "tcn_score",
        "transformer_score",
    ]

    pred_df = pred_df[[c for c in pred_cols if c in pred_df.columns]].copy()

    summary_df["run_id"] = RUN_ID
    weighted_df["run_id"] = RUN_ID

    print("\n=== Escribiendo externa a PostgreSQL ===")
    write_to_postgres(pred_df, "eval_externa_predictions")
    write_to_postgres(summary_df, "eval_externa_summary")
    write_to_postgres(weighted_df, "eval_externa_weighted")
    write_to_postgres(runs_df, "eval_externa_runs")

    print("\n✅ external_evaluation terminada")
    print(weighted_df.to_string(index=False))


if __name__ == "__main__":
    main()