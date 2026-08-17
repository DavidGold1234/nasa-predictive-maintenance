from __future__ import annotations

import json
import math
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pyspark.sql import SparkSession


HDFS_ROS_GOLD = os.environ.get(
    "HDFS_ROS_GOLD",
    "hdfs://namenode:9000/user/root/nasa/gold/ros_gold",
)
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "/data/models/artifacts")

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "tesis")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
POSTGRES_SCHEMA = os.environ.get("POSTGRES_SCHEMA", "public")
POSTGRES_IF_EXISTS = os.environ.get("POSTGRES_IF_EXISTS", "append")

WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "30"))
DURATION_COL = "cycles_since_regime_change"
BASELINE_SIZE = int(os.environ.get("BASELINE_SIZE", "200"))
BASELINE_LOG_EVERY = int(os.environ.get("BASELINE_LOG_EVERY", "20"))
RUN_ID = os.environ.get("RUN_ID", datetime.now(timezone.utc).strftime("ros_eval_%Y%m%d_%H%M%S"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def write_to_postgres(df: pd.DataFrame, table_name: str) -> None:
    if df.empty:
        print(f"⚠️ {table_name}: dataframe vacío, no se escribe")
        return

    full_table = f"{POSTGRES_SCHEMA}.{table_name}" if POSTGRES_SCHEMA else table_name
    try:
        from sqlalchemy import create_engine  # type: ignore

        engine = create_engine(
            f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
        )
        df.to_sql(
            table_name,
            con=engine,
            schema=POSTGRES_SCHEMA or None,
            if_exists=POSTGRES_IF_EXISTS,
            index=False,
            method="multi",
            chunksize=1000,
        )
        print(f"✅ PostgreSQL vía SQLAlchemy -> {full_table} | rows={len(df)}")
        return
    except Exception as err:
        print(f"ℹ️ SQLAlchemy no disponible o falló para {table_name}: {err}")

    spark = (
        SparkSession.builder
        .appName(f"Write{table_name}")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .getOrCreate()
    )
    jdbc_url = f"jdbc:postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    props = {"user": POSTGRES_USER, "password": POSTGRES_PASSWORD, "driver": "org.postgresql.Driver"}
    spark.createDataFrame(df).write.mode(POSTGRES_IF_EXISTS).jdbc(jdbc_url, full_table, properties=props)
    print(f"✅ PostgreSQL vía Spark JDBC -> {full_table} | rows={len(df)}")
    spark.stop()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_sensor_cols(pdf: pd.DataFrame, artifact_dir: str) -> list[str]:
    cfg = load_json(os.path.join(artifact_dir, "selected_features.json"))
    cols = [c for c in cfg.get("features", []) if c.startswith("sensor_") and c in pdf.columns]
    if cols:
        return cols
    return sorted([c for c in pdf.columns if c.startswith("sensor_")])


def safe_scale(raw_score: float, name: str, calibration: dict[str, Any] | None) -> float:
    if not calibration or name not in calibration:
        return float(raw_score)
    info = calibration[name]
    lo = float(info.get("min", raw_score))
    hi = float(info.get("max", raw_score))
    if hi - lo < 1e-12:
        return 0.0
    return float((raw_score - lo) / (hi - lo))


def compute_status(score: float, history: list[float], baseline: list[float]) -> tuple[str, float, float, float, float, float, float]:
    prev = history[-1] if history else score
    delta = float(score - prev)

    if len(baseline) < max(10, BASELINE_SIZE):
        return "BASELINE", delta, 0.0, float(np.mean(baseline)) if baseline else float(score), 0.0, 0.0

    arr = np.asarray(baseline, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr)) + 1e-9
    z = float((score - mean) / std)
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))

    if score >= p99 or z >= 3.0:
        status = "CRITICAL"
    elif score >= p95 or z >= 2.0:
        status = "WARNING"
    else:
        status = "HEALTHY"

    trend = 0.0
    if len(history) >= 5:
        recent = np.asarray(history[-5:] + [score], dtype=float)
        trend = float(np.mean(np.diff(recent)))

    return status, delta, z, mean, p95, p99, trend


def fuse_status(lstm_status: str, tcn_status: str, transformer_status: str, if_status: str) -> str:
    """
    Lógica de fusión del ensemble (misma que antes con CNN, extendida a 3 autoencoders):
    - CRITICAL si al menos 1 AE (LSTM, TCN o Transformer) es CRITICAL.
    - WARNING  si al menos 1 AE es WARNING.
    - WARNING  si IF es CRITICAL pero los 3 AEs son HEALTHY (señal secundaria fuerte).
    - HEALTHY  en caso contrario.
    El IF actúa como confirmación secundaria; los AEs son el criterio primario.
    Pendiente (no resuelto en esta sesión): si conviene ponderar distinto a cada AE
    en vez de tratarlos como gatillos equivalentes.
    """
    ae_statuses = (lstm_status, tcn_status, transformer_status)
    primary_critical = "CRITICAL" in ae_statuses
    primary_warning = "WARNING" in ae_statuses

    if primary_critical:
        return "CRITICAL"
    if primary_warning:
        return "WARNING"
    if if_status == "CRITICAL":
        return "WARNING"
    return "HEALTHY"


def main() -> None:
    required = [
        "selected_features.json",
        "isolation_forest.pkl",
        "lstm_autoencoder.pt",
        "tcn_autoencoder.pt",
        "transformer_autoencoder.pt",
    ]
    missing = [name for name in required if not (Path(ARTIFACTS_DIR) / name).exists()]
    if missing:
        raise FileNotFoundError(f"Faltan artifacts para evaluación ROS: {missing}")

    spark = (
        SparkSession.builder
        .appName("EvaluateROSFromGold")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    pdf = spark.read.parquet(HDFS_ROS_GOLD).toPandas()
    spark.stop()

    if pdf.empty:
        raise RuntimeError("ros_gold está vacío")
    if DURATION_COL not in pdf.columns:
        raise ValueError(f"Falta '{DURATION_COL}' en ros_gold (generado por build_ros_gold.py)")

    sensor_cols = choose_sensor_cols(pdf, ARTIFACTS_DIR)
    model_input_cols = sensor_cols + [DURATION_COL]
    pdf = pdf.sort_values(["dataset_id", "engine_id", "cycle"]).reset_index(drop=True)

    calibration_path = Path(ARTIFACTS_DIR) / "score_calibration.json"
    calibration = load_json(str(calibration_path)) if calibration_path.exists() else None

    if_model = joblib.load(os.path.join(ARTIFACTS_DIR, "isolation_forest.pkl"))

    n_features = len(model_input_cols)
    lstm = LSTMAutoencoder(n_features=n_features).to(DEVICE)
    lstm.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "lstm_autoencoder.pt"), map_location=DEVICE))
    lstm.eval()

    tcn = TCNAutoencoder(n_features=n_features).to(DEVICE)
    tcn.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "tcn_autoencoder.pt"), map_location=DEVICE))
    tcn.eval()

    transformer = TransformerAutoencoder(n_features=n_features).to(DEVICE)
    transformer.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "transformer_autoencoder.pt"), map_location=DEVICE))
    transformer.eval()

    predictions: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []

    for engine_id, g in pdf.groupby("engine_id", sort=True):
        g = g.sort_values("cycle").reset_index(drop=True)
        dataset_id = str(g["dataset_id"].iloc[0])
        engine_num = int(g["engine_num"].iloc[0])

        window = deque(maxlen=WINDOW_SIZE)
        baseline_if: list[float] = []
        baseline_lstm: list[float] = []
        baseline_tcn: list[float] = []
        baseline_transformer: list[float] = []

        hist_if: list[float] = []
        hist_lstm: list[float] = []
        hist_tcn: list[float] = []
        hist_transformer: list[float] = []

        warmup_rows = 0
        baseline_rows = 0
        eval_rows = 0
        first_warning_cycle = None
        first_critical_cycle = None

        for _, row in g.iterrows():
            feat = row[model_input_cols].to_numpy(dtype=np.float32)
            window.append(feat)

            if len(window) < WINDOW_SIZE:
                warmup_rows += 1
                continue

            x = np.asarray(window, dtype=np.float32)
            x_batch = x.reshape(1, WINDOW_SIZE, n_features)
            x_tensor = torch.tensor(x_batch, dtype=torch.float32, device=DEVICE)

            if_raw = float(-if_model.decision_function(x.reshape(1, -1))[0])
            with torch.no_grad():
                lstm_rec = lstm(x_tensor)
                tcn_rec = tcn(x_tensor)
                transformer_rec = transformer(x_tensor)
                lstm_raw = float(torch.mean((x_tensor - lstm_rec) ** 2).item())
                tcn_raw = float(torch.mean((x_tensor - tcn_rec) ** 2).item())
                transformer_raw = float(torch.mean((x_tensor - transformer_rec) ** 2).item())

            if_score = safe_scale(if_raw, "IsolationForest", calibration)
            lstm_score = safe_scale(lstm_raw, "LSTM_AE", calibration)
            tcn_score = safe_scale(tcn_raw, "TCN_AE", calibration)
            transformer_score = safe_scale(transformer_raw, "Transformer_AE", calibration)

            if len(baseline_if) < BASELINE_SIZE:
                baseline_if.append(if_score)
                baseline_lstm.append(lstm_score)
                baseline_tcn.append(tcn_score)
                baseline_transformer.append(transformer_score)
                baseline_rows += 1
                if baseline_rows % max(1, BASELINE_LOG_EVERY) == 0:
                    print(f"ℹ️ {engine_id} baseline {baseline_rows}/{BASELINE_SIZE}")
                continue

            lstm_status, lstm_delta, lstm_z, lstm_mean, lstm_warn, lstm_crit, lstm_trend = compute_status(
                lstm_score, hist_lstm, baseline_lstm
            )
            tcn_status, tcn_delta, tcn_z, tcn_mean, tcn_warn, tcn_crit, tcn_trend = compute_status(
                tcn_score, hist_tcn, baseline_tcn
            )
            transformer_status, transformer_delta, transformer_z, transformer_mean, transformer_warn, transformer_crit, transformer_trend = compute_status(
                transformer_score, hist_transformer, baseline_transformer
            )
            if_status, if_delta, if_z, if_mean, if_warn, if_crit, if_trend = compute_status(
                if_score, hist_if, baseline_if
            )

            final_status = fuse_status(lstm_status, tcn_status, transformer_status, if_status)
            eval_rows += 1

            cycle = int(row["cycle"])
            if first_warning_cycle is None and final_status in {"WARNING", "CRITICAL"}:
                first_warning_cycle = cycle
            if first_critical_cycle is None and final_status == "CRITICAL":
                first_critical_cycle = cycle

            # Solo campos esenciales: scores calibrados, status individual y status final.
            # Los thresholds, deltas y z-scores son derivables de los scores si se necesitan.
            predictions.append(
                {
                    "run_id": RUN_ID,
                    "dataset_id": dataset_id,
                    "engine_id": str(engine_id),
                    "engine_num": engine_num,
                    "cycle": cycle,
                    "if_score": if_score,
                    "lstm_score": lstm_score,
                    "tcn_score": tcn_score,
                    "transformer_score": transformer_score,
                    "if_status": if_status,
                    "lstm_status": lstm_status,
                    "tcn_status": tcn_status,
                    "transformer_status": transformer_status,
                    "final_status": final_status,
                }
            )

            hist_if.append(if_score)
            hist_lstm.append(lstm_score)
            hist_tcn.append(tcn_score)
            hist_transformer.append(transformer_score)

        metrics.append(
            {
                "run_id": RUN_ID,
                "dataset_id": dataset_id,
                "engine_id": str(engine_id),
                "engine_num": engine_num,
                "total_rows_gold": int(len(g)),
                "warmup_rows": int(warmup_rows),
                "baseline_rows": int(baseline_rows),
                "evaluated_rows": int(eval_rows),
                "first_warning_cycle": None if first_warning_cycle is None else int(first_warning_cycle),
                "first_critical_cycle": None if first_critical_cycle is None else int(first_critical_cycle),
                "last_cycle": int(g["cycle"].max()),
                "final_status_last": predictions[-1]["final_status"] if predictions else "BASELINE",
                "artifact_version": g["artifact_version"].iloc[0] if "artifact_version" in g.columns else None,
            }
        )

    pred_df = pd.DataFrame(predictions)
    metrics_df = pd.DataFrame(metrics)

    write_to_postgres(pred_df, "eval_ros_predictions")
    write_to_postgres(metrics_df, "eval_ros_metrics")

    print("✅ Evaluación ROS desde gold terminada")
    print(f"✅ eval_ros_predictions rows={len(pred_df)}")
    print(f"✅ eval_ros_metrics rows={len(metrics_df)}")


if __name__ == "__main__":
    main()
