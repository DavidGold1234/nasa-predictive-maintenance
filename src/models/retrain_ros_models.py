from __future__ import annotations

import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pyspark.sql import SparkSession
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader, TensorDataset

# =========================================================
# CONFIG
# =========================================================
SEED = int(os.environ.get("SEED", "42"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HDFS_ROS_GOLD = os.environ.get(
    "HDFS_ROS_GOLD",
    "hdfs://namenode:9000/user/root/nasa/gold/ros_gold",
)
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "/data/models/artifacts")
VERSIONS_DIR = os.environ.get("VERSIONS_DIR", "/data/models/versions")
HDFS_ARTIFACTS_ACTIVE = os.environ.get(
    "HDFS_ARTIFACTS_ACTIVE",
    "hdfs://namenode:9000/user/root/nasa/artifacts/active",
)
HDFS_ARTIFACTS_VERSIONS = os.environ.get(
    "HDFS_ARTIFACTS_VERSIONS",
    "hdfs://namenode:9000/user/root/nasa/artifacts/versions",
)

RUN_ID = os.environ.get("RUN_ID", datetime.now(timezone.utc).strftime("ros_retrain_%Y%m%d_%H%M%S"))
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "30"))
DURATION_COL = "cycles_since_regime_change"
HEALTHY_RATIO = float(os.environ.get("HEALTHY_RATIO", "0.30"))
# Reducido a 50 para permitir reentrenamiento con datos ROS de simulación limitada
MIN_HEALTHY_WINDOWS = int(os.environ.get("MIN_HEALTHY_WINDOWS", "50"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
# Fine-tuning: LR reducido 10× para no destruir los pesos NASA aprendidos
FINETUNE_LR = float(os.environ.get("FINETUNE_LR", "0.0001"))
# 1 = congelar encoder (preserva conocimiento NASA), 0 = entrenar todo desde cero
FREEZE_ENCODER = int(os.environ.get("FREEZE_ENCODER", "1"))
PATIENCE = int(os.environ.get("PATIENCE", "5"))
VAL_ENGINE_RATIO = float(os.environ.get("VAL_ENGINE_RATIO", "0.20"))
LSTM_MAX_WINDOWS = int(os.environ.get("LSTM_MAX_WINDOWS", "30000"))
TCN_MAX_WINDOWS = int(os.environ.get("TCN_MAX_WINDOWS", "30000"))
TRANSFORMER_MAX_WINDOWS = int(os.environ.get("TRANSFORMER_MAX_WINDOWS", "30000"))
# "auto" = umbral teórico de IF; correcto cuando se entrena SOLO sobre ventanas sanas confirmadas
_if_cont = os.environ.get("IF_CONTAMINATION", "auto")
IF_CONTAMINATION: float | str = float(_if_cont) if _if_cont != "auto" else "auto"

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# HELPERS
# =========================================================
def build_spark(app_name: str) -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def choose_sensor_cols(pdf: pd.DataFrame, artifact_dir: str) -> list[str]:
    cfg_path = os.path.join(artifact_dir, "selected_features.json")
    if os.path.exists(cfg_path):
        cfg = load_json(cfg_path)
        cols = [c for c in cfg.get("features", []) if c.startswith("sensor_") and c in pdf.columns]
        if cols:
            return cols
    return sorted([c for c in pdf.columns if c.startswith("sensor_")])


def _get_fs_and_path(spark: SparkSession, uri_str: str):
    jvm = spark._jvm
    conf = spark._jsc.hadoopConfiguration()
    uri = jvm.java.net.URI(uri_str)
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(uri, conf)
    path = jvm.org.apache.hadoop.fs.Path(uri_str)
    return fs, path


def hdfs_put(spark: SparkSession, local_path: str | Path, hdfs_dir: str, filename: str | None = None) -> str:
    local_path = Path(local_path).resolve()
    name = filename or local_path.name

    fs_dir, dir_path = _get_fs_and_path(spark, hdfs_dir)
    fs_dir.mkdirs(dir_path)

    target_uri = f"{hdfs_dir.rstrip('/')}/{name}"
    fs_target, target_path = _get_fs_and_path(spark, target_uri)

    jvm = spark._jvm
    local_uri = jvm.java.net.URI(local_path.as_uri())
    local_hdfs_path = jvm.org.apache.hadoop.fs.Path(local_uri)

    fs_target.copyFromLocalFile(False, True, local_hdfs_path, target_path)
    return target_uri


def copy_to_version(local_source: str | Path, version_dir: str | Path) -> Path:
    src = Path(local_source)
    dst = Path(version_dir) / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def backup_files_to_hdfs(
    spark: SparkSession,
    files: Iterable[str | Path],
    hdfs_dir: str,
) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for file_path in files:
        file_path = Path(file_path)
        manifest[file_path.name] = hdfs_put(spark, file_path, hdfs_dir)
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


def build_ros_windows(
    df: pd.DataFrame,
    sensor_cols: list[str],
    healthy_only: bool,
    healthy_ratio: float,
) -> Tuple[np.ndarray, pd.DataFrame]:
    windows = []
    meta_rows = []

    group_cols = [c for c in ["dataset_id", "engine_id"] if c in df.columns]
    if not group_cols:
        raise ValueError("ros_gold debe tener al menos engine_id")

    for keys, g in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)

        gg = g.sort_values("cycle").reset_index(drop=True)
        if len(gg) < WINDOW_SIZE:
            continue

        min_cycle = int(gg["cycle"].min())
        max_cycle = int(gg["cycle"].max())
        span = max_cycle - min_cycle + 1
        healthy_cycles = max(WINDOW_SIZE, int(np.ceil(span * healthy_ratio)))
        healthy_cutoff = min_cycle + healthy_cycles - 1

        dataset_id = str(gg["dataset_id"].iloc[0]) if "dataset_id" in gg.columns else "ros"
        engine_id = str(gg["engine_id"].iloc[0])
        engine_num = int(gg["engine_num"].iloc[0]) if "engine_num" in gg.columns else -1

        for end_idx in range(WINDOW_SIZE - 1, len(gg)):
            w = gg.iloc[end_idx - WINDOW_SIZE + 1 : end_idx + 1]
            end_row = w.iloc[-1]
            is_healthy_window = int(w["cycle"].max()) <= healthy_cutoff
            if healthy_only and not is_healthy_window:
                continue

            windows.append(w[sensor_cols].to_numpy(dtype=np.float32))
            meta_rows.append(
                {
                    "dataset_id": dataset_id,
                    "engine_id": engine_id,
                    "engine_num": engine_num,
                    "cycle": int(end_row["cycle"]),
                    "is_healthy_window": int(is_healthy_window),
                }
            )

    return np.asarray(windows, dtype=np.float32), pd.DataFrame(meta_rows)


# =========================================================
# FINE-TUNING HELPERS
# =========================================================
def load_pretrained_weights(model: nn.Module, weights_path: str) -> bool:
    """
    Carga pesos pre-entrenados (NASA) en el modelo.
    Retorna True si los pesos se cargaron, False si no existían (train desde cero).
    """
    path = Path(weights_path)
    if not path.exists():
        print(f"⚠️  No se encontraron pesos pre-entrenados en {weights_path}. Entrenando desde cero.")
        return False
    try:
        state = torch.load(str(path), map_location=DEVICE)
        model.load_state_dict(state, strict=True)
        print(f"✅ Pesos NASA cargados desde {weights_path}")
        return True
    except Exception as exc:
        print(f"⚠️  Error cargando pesos pre-entrenados: {exc}. Entrenando desde cero.")
        return False


def freeze_encoder(model: nn.Module, model_type: str) -> int:
    """
    Congela las capas del encoder del modelo para preservar el conocimiento NASA.

    Estrategia de transfer learning para la tesis:
    - El ENCODER aprende la firma de degradación en espacio latente (conocimiento NASA).
    - El DECODER aprende a reconstruir los patrones de sensores ROS específicos.
    - Congelar encoder + fine-tune decoder = adaptar ROS sin olvidar NASA.

    El conocimiento que se transfiere NO es la comparación directa de motores físicos,
    sino el PROCESO de degradación: ambos dominios comparten la estructura latente de
    cómo un sistema se degrada (señales que se desvían de su baseline sano), aunque
    los sensores físicos sean distintos (temperatura/presión en turbofan vs
    torque/pendiente en robot ROS).

    Retorna el número de parámetros entrenables después de congelar.
    """
    frozen_keywords = {
        "LSTM": ["encoder", "to_latent"],          # encoder LSTM bidireccional + proyección latente
        "TCN": ["encoder", "to_latent"],           # bloques TCN causales dilatados + proyección latente
        "Transformer": ["encoder", "to_latent", "input_proj"],  # self-attention + proyección latente
    }
    keywords = frozen_keywords.get(model_type, [])

    for name, param in model.named_parameters():
        if any(kw in name for kw in keywords):
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"   Fine-tuning {model_type}: {trainable:,}/{total:,} params entrenables "
        f"(encoder congelado, decoder libre)"
    )
    return trainable


def prepare_for_finetune(model: nn.Module, model_type: str, weights_path: str) -> tuple[nn.Module, float, bool]:
    """
    Conecta load_pretrained_weights()/freeze_encoder() al loop de entrenamiento.
    Antes estaban definidas pero nunca se llamaban desde main(): el "reentrenamiento"
    ROS en realidad entrenaba todo desde cero, sin transferir nada de NASA pese a que
    el diseño (FREEZE_ENCODER, FINETUNE_LR) ya estaba pensado para hacerlo.

    Retorna (modelo, learning_rate a usar, si quedó en modo fine-tuning).
    """
    loaded = load_pretrained_weights(model, weights_path)
    if loaded and FREEZE_ENCODER:
        freeze_encoder(model, model_type)
        return model, FINETUNE_LR, True

    if loaded and not FREEZE_ENCODER:
        print(f"   {model_type}: pesos NASA cargados pero FREEZE_ENCODER=0 -> fine-tuning completo (todo entrenable).")
        return model, FINETUNE_LR, True

    return model, LEARNING_RATE, False


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

    dl_train = DataLoader(
        TensorDataset(torch.tensor(X_train_sub, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    dl_val = None
    if len(X_val_sub) > 0:
        dl_val = DataLoader(
            TensorDataset(torch.tensor(X_val_sub, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=False,
        )

    model = model.to(DEVICE)
    # Solo optimizar parámetros entrenables (los no congelados del encoder en fine-tuning)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No hay parámetros entrenables. Verifica la configuración de freeze_encoder.")
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
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
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(VERSIONS_DIR, exist_ok=True)

    spark = build_spark("RetrainROSModels")

    try:
        print("📥 Leyendo ros_gold...")
        gold = spark.read.parquet(HDFS_ROS_GOLD).toPandas()

        if gold.empty:
            raise RuntimeError("ros_gold está vacío")

        required_base = {"engine_id", "cycle"}
        missing_base = [c for c in required_base if c not in gold.columns]
        if missing_base:
            raise ValueError(f"Faltan columnas base en ros_gold: {missing_base}")

        sensor_cols = choose_sensor_cols(gold, ARTIFACTS_DIR)
        if not sensor_cols:
            raise ValueError("No encontré sensores válidos en ros_gold ni en selected_features.json")
        if DURATION_COL not in gold.columns:
            raise ValueError(f"Falta '{DURATION_COL}' en ros_gold (generado por build_ros_gold.py)")
        model_input_cols = sensor_cols + [DURATION_COL]

        if "dataset_id" not in gold.columns:
            gold["dataset_id"] = "ros"
        if "engine_num" not in gold.columns:
            gold["engine_num"] = pd.to_numeric(gold["engine_id"], errors="coerce").fillna(-1).astype(int)

        gold = gold.copy()
        gold["dataset_id"] = gold["dataset_id"].astype(str)
        gold["engine_id"] = gold["engine_id"].astype(str)
        gold["engine_num"] = pd.to_numeric(gold["engine_num"], errors="coerce")
        gold["cycle"] = pd.to_numeric(gold["cycle"], errors="coerce")

        for c in model_input_cols:
            gold[c] = pd.to_numeric(gold[c], errors="coerce")

        gold = gold.dropna(subset=["engine_num", "cycle"] + model_input_cols).copy()
        gold["engine_num"] = gold["engine_num"].astype(int)
        gold["cycle"] = gold["cycle"].astype(int)
        gold = gold.sort_values(
            [c for c in ["dataset_id", "engine_num", "cycle"] if c in gold.columns]
        ).reset_index(drop=True)

        X_all, meta_all = build_ros_windows(
            gold, model_input_cols, healthy_only=False, healthy_ratio=HEALTHY_RATIO
        )
        X_healthy, meta_healthy = build_ros_windows(
            gold, model_input_cols, healthy_only=True, healthy_ratio=HEALTHY_RATIO
        )

        if len(X_all) == 0 or len(X_healthy) == 0:
            raise RuntimeError(
                f"No hay suficientes ventanas en ros_gold. total={len(X_all)} healthy={len(X_healthy)}"
            )
        if len(X_healthy) < MIN_HEALTHY_WINDOWS:
            raise RuntimeError(
                f"Muy pocas ventanas healthy para retrain: {len(X_healthy)} < {MIN_HEALTHY_WINDOWS}. "
                "Ajusta HEALTHY_RATIO o espera más datos ROS."
            )

        print(f"✅ Ventanas ROS totales: {X_all.shape}")
        print(f"✅ Ventanas ROS healthy: {X_healthy.shape}")

        train_idx, val_idx = split_train_val_by_engine(meta_healthy, VAL_ENGINE_RATIO, SEED)
        X_fit = X_healthy[train_idx]
        X_val = X_healthy[val_idx] if len(val_idx) else X_healthy[train_idx[: min(len(train_idx), 1024)]]

        X_fit_flat = X_fit.reshape(len(X_fit), -1)
        X_ref_flat = X_healthy.reshape(len(X_healthy), -1)

        print("\n=== Retraining Isolation Forest on ROS healthy windows ===")
        if_model = IsolationForest(
            n_estimators=300,
            contamination=IF_CONTAMINATION,
            random_state=SEED,
            n_jobs=-1,
        )
        if_model.fit(X_fit_flat)

        print("\n=== Retraining LSTM Autoencoder on ROS healthy windows ===")
        lstm_ae = LSTMAutoencoder(n_features=len(model_input_cols))
        lstm_ae, lstm_lr, lstm_finetuned = prepare_for_finetune(
            lstm_ae, "LSTM", os.path.join(ARTIFACTS_DIR, "lstm_autoencoder.pt")
        )
        lstm_ae, lstm_hist = train_autoencoder(
            lstm_ae,
            X_fit,
            X_val,
            EPOCHS,
            BATCH_SIZE,
            lstm_lr,
            PATIENCE,
            LSTM_MAX_WINDOWS,
        )

        print("\n=== Retraining TCN Autoencoder on ROS healthy windows ===")
        tcn_ae = TCNAutoencoder(n_features=len(model_input_cols))
        tcn_ae, tcn_lr, tcn_finetuned = prepare_for_finetune(
            tcn_ae, "TCN", os.path.join(ARTIFACTS_DIR, "tcn_autoencoder.pt")
        )
        tcn_ae, tcn_hist = train_autoencoder(
            tcn_ae,
            X_fit,
            X_val,
            EPOCHS,
            BATCH_SIZE,
            tcn_lr,
            PATIENCE,
            TCN_MAX_WINDOWS,
        )

        print("\n=== Retraining Transformer Autoencoder on ROS healthy windows ===")
        transformer_ae = TransformerAutoencoder(n_features=len(model_input_cols))
        transformer_ae, transformer_lr, transformer_finetuned = prepare_for_finetune(
            transformer_ae, "Transformer", os.path.join(ARTIFACTS_DIR, "transformer_autoencoder.pt")
        )
        transformer_ae, transformer_hist = train_autoencoder(
            transformer_ae,
            X_fit,
            X_val,
            EPOCHS,
            BATCH_SIZE,
            transformer_lr,
            PATIENCE,
            TRANSFORMER_MAX_WINDOWS,
        )

        print("\n=== Building calibration from ROS healthy reference ===")
        if_ref_raw = -if_model.decision_function(X_ref_flat)
        lstm_ref_raw = infer_ae_scores(lstm_ae, X_healthy)
        tcn_ref_raw = infer_ae_scores(tcn_ae, X_healthy)
        transformer_ref_raw = infer_ae_scores(transformer_ae, X_healthy)

        version_root_local = Path(VERSIONS_DIR) / RUN_ID
        version_models_dir = version_root_local / "models"
        version_pre_dir = version_root_local / "preprocessing"
        version_meta_dir = version_root_local / "metadata"
        version_models_dir.mkdir(parents=True, exist_ok=True)
        version_pre_dir.mkdir(parents=True, exist_ok=True)
        version_meta_dir.mkdir(parents=True, exist_ok=True)

        print("\n=== Guardando artifacts activos locales ===")
        local_if_path = Path(ARTIFACTS_DIR) / "isolation_forest.pkl"
        local_lstm_path = Path(ARTIFACTS_DIR) / "lstm_autoencoder.pt"
        local_tcn_path = Path(ARTIFACTS_DIR) / "tcn_autoencoder.pt"
        local_transformer_path = Path(ARTIFACTS_DIR) / "transformer_autoencoder.pt"
        local_calibration_path = Path(ARTIFACTS_DIR) / "score_calibration.json"
        local_retrain_meta_path = Path(ARTIFACTS_DIR) / "retrain_metadata.json"
        local_active_version_path = Path(ARTIFACTS_DIR) / "active_version.json"

        joblib.dump(if_model, local_if_path)
        torch.save(lstm_ae.state_dict(), local_lstm_path)
        torch.save(tcn_ae.state_dict(), local_tcn_path)
        torch.save(transformer_ae.state_dict(), local_transformer_path)

        score_calibration = {
            "run_id": RUN_ID,
            "reference": "ros_gold_healthy_windows",
            "IsolationForest": calibration_dict(if_ref_raw),
            "LSTM_AE": calibration_dict(lstm_ref_raw),
            "TCN_AE": calibration_dict(tcn_ref_raw),
            "Transformer_AE": calibration_dict(transformer_ref_raw),
        }
        write_json(local_calibration_path, score_calibration)

        retrain_metadata = {
            "run_id": RUN_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "ros_gold",
            "gold_input_path": HDFS_ROS_GOLD,
            "window_size": WINDOW_SIZE,
            "healthy_ratio": HEALTHY_RATIO,
            "min_healthy_windows": MIN_HEALTHY_WINDOWS,
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
            "finetune_lr": FINETUNE_LR,
            "freeze_encoder_requested": bool(FREEZE_ENCODER),
            "patience": PATIENCE,
            "val_engine_ratio": VAL_ENGINE_RATIO,
            "seed": SEED,
            "total_windows": int(len(X_all)),
            "healthy_windows": int(len(X_healthy)),
            "fit_windows": int(len(X_fit)),
            "val_windows": int(len(X_val)),
            "lstm_finetuned_from_nasa": lstm_finetuned,
            "tcn_finetuned_from_nasa": tcn_finetuned,
            "transformer_finetuned_from_nasa": transformer_finetuned,
            "lstm_best_val_loss": float(lstm_hist["best_val_loss"]),
            "tcn_best_val_loss": float(tcn_hist["best_val_loss"]),
            "transformer_best_val_loss": float(transformer_hist["best_val_loss"]),
            "lstm_best_epoch": int(lstm_hist["best_epoch"]),
            "tcn_best_epoch": int(tcn_hist["best_epoch"]),
            "transformer_best_epoch": int(transformer_hist["best_epoch"]),
            "hdfs_artifacts_active_root": HDFS_ARTIFACTS_ACTIVE,
            "hdfs_artifacts_versions_root": HDFS_ARTIFACTS_VERSIONS,
            "versions_dir": str(version_root_local),
        }
        write_json(local_retrain_meta_path, retrain_metadata)

        active_version_payload = {
            "version": RUN_ID,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "updated_by": "retrain_ros_models.py",
            "source": "ros_gold",
        }
        write_json(local_active_version_path, active_version_payload)

        # Snapshot local version folder
        version_model_files = [
            copy_to_version(local_if_path, version_models_dir),
            copy_to_version(local_lstm_path, version_models_dir),
            copy_to_version(local_tcn_path, version_models_dir),
            copy_to_version(local_transformer_path, version_models_dir),
            copy_to_version(local_calibration_path, version_models_dir),
        ]

        preprocessing_snapshot_names = [
            "selected_features.json",
            "regime_baseline.pkl",
            "healthy_global_scaler.pkl",
            "regime_detector_kmeans.pkl",
            "regime_duration_scaler.pkl",
            "preprocessing_metadata.json",
        ]
        version_pre_files = []
        for name in preprocessing_snapshot_names:
            src = Path(ARTIFACTS_DIR) / name
            if src.exists():
                version_pre_files.append(copy_to_version(src, version_pre_dir))

        version_meta_files = [
            copy_to_version(local_retrain_meta_path, version_meta_dir),
            copy_to_version(local_active_version_path, version_meta_dir),
        ]

        print("☁️ Respaldando artifacts/modelos en HDFS...")
        hdfs_active_models_dir = f"{HDFS_ARTIFACTS_ACTIVE.rstrip('/')}/models"
        hdfs_active_meta_dir = f"{HDFS_ARTIFACTS_ACTIVE.rstrip('/')}/metadata"
        hdfs_version_models_dir = f"{HDFS_ARTIFACTS_VERSIONS.rstrip('/')}/{RUN_ID}/models"
        hdfs_version_pre_dir = f"{HDFS_ARTIFACTS_VERSIONS.rstrip('/')}/{RUN_ID}/preprocessing"
        hdfs_version_meta_dir = f"{HDFS_ARTIFACTS_VERSIONS.rstrip('/')}/{RUN_ID}/metadata"

        active_manifest = {}
        for f in [local_if_path, local_lstm_path, local_tcn_path, local_transformer_path, local_calibration_path]:
            active_manifest[f.name] = hdfs_put(spark, f, hdfs_active_models_dir)
        active_manifest[local_active_version_path.name] = hdfs_put(
            spark, local_active_version_path, hdfs_active_meta_dir
        )
        active_manifest[local_retrain_meta_path.name] = hdfs_put(
            spark, local_retrain_meta_path, hdfs_active_meta_dir
        )

        version_manifest = {
            "models": backup_files_to_hdfs(spark, version_model_files, hdfs_version_models_dir),
            "preprocessing": (
                backup_files_to_hdfs(spark, version_pre_files, hdfs_version_pre_dir)
                if version_pre_files
                else {}
            ),
            "metadata": backup_files_to_hdfs(spark, version_meta_files, hdfs_version_meta_dir),
        }

        version_manifest_path = version_meta_dir / "retrain_backup_manifest.json"
        write_json(
            version_manifest_path,
            {
                "run_id": RUN_ID,
                "active": active_manifest,
                "version": version_manifest,
            },
        )
        hdfs_put(spark, version_manifest_path, hdfs_version_meta_dir)

        print("✅ Retrain ROS terminado")
        print(f"✅ Artifacts activos locales: {ARTIFACTS_DIR}")
        print(f"✅ Version local creada: {version_root_local}")
        print(f"✅ Respaldo HDFS versión: {HDFS_ARTIFACTS_VERSIONS.rstrip('/')}/{RUN_ID}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()