from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from kafka import KafkaConsumer, KafkaProducer


ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "/data/models/artifacts")
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC_IN = os.environ.get("TOPIC_IN", "ros_motor_telemetry")
TOPIC_OUT = os.environ.get("TOPIC_OUT", "ros_motor_alerts")

WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "30"))
DURATION_COL = "cycles_since_regime_change"
BASELINE_SIZE = int(os.environ.get("BASELINE_SIZE", "100"))
BASELINE_LOG_EVERY = int(os.environ.get("BASELINE_LOG_EVERY", "10"))
LOG_COMPACT = int(os.environ.get("LOG_COMPACT", "1"))
DEBUG_METRICS = int(os.environ.get("DEBUG_METRICS", "0"))
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


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_artifact(name: str) -> str:
    path = Path(ARTIFACTS_DIR) / name
    if path.exists():
        return str(path)
    raise FileNotFoundError(f"No se encontró artifact activo: {path}")


def choose_sensor_cols() -> list[str]:
    cfg = load_json(resolve_artifact("selected_features.json"))
    cols = [c for c in cfg.get("features", []) if c.startswith("sensor_")]
    if not cols:
        raise ValueError("selected_features.json no contiene sensors válidos")
    return cols


def load_torch_state_dict(path: str):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def get_calibration_entry(calibration: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if not calibration:
        return None

    alias_map = {
        "IsolationForest": ["IsolationForest", "isolation_forest", "IF", "if"],
        "LSTM_AE": ["LSTM_AE", "LSTM", "lstm"],
        "TCN_AE": ["TCN_AE", "TCN", "tcn"],
        "Transformer_AE": ["Transformer_AE", "Transformer", "transformer"],
    }

    for alias in alias_map.get(name, [name]):
        entry = calibration.get(alias)
        if isinstance(entry, dict):
            return entry

    return None


def safe_scale(raw_score: float, name: str, calibration: dict[str, Any] | None) -> float:
    info = get_calibration_entry(calibration, name)
    if not info:
        return float(raw_score)

    lo = float(info.get("min", raw_score))
    hi = float(info.get("max", raw_score))

    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        return float(raw_score)

    return float((raw_score - lo) / (hi - lo))


def compute_status(
    score: float,
    history: list[float],
    baseline: list[float],
) -> tuple[str, float, float, float, float, float, float]:
    prev = history[-1] if history else score
    delta = float(score - prev)

    if len(baseline) < BASELINE_SIZE:
        baseline_mean = float(np.mean(baseline)) if baseline else float(score)
        return "BASELINE", delta, 0.0, baseline_mean, 0.0, 0.0, 0.0

    arr = np.asarray(baseline, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    std_safe = max(std, 1e-9)
    z = float((score - mean) / std_safe)
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))

    if std < 1e-9 and abs(score - mean) < 1e-9:
        status = "HEALTHY"
    elif score > p99 or z >= 3.0:
        status = "CRITICAL"
    elif score > p95 or z >= 2.0:
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
    Lógica de fusión del ensemble (misma que evaluate_ros_from_gold.py, extendida a 3 AEs):
    - CRITICAL si al menos 1 AE (LSTM, TCN o Transformer) es CRITICAL.
    - WARNING  si al menos 1 AE es WARNING.
    - WARNING  si IF es CRITICAL pero los 3 AEs están sanos (señal secundaria fuerte).
    - HEALTHY  en caso contrario.
    """
    ae_statuses = (lstm_status, tcn_status, transformer_status)
    if "CRITICAL" in ae_statuses:
        return "CRITICAL"
    if "WARNING" in ae_statuses:
        return "WARNING"
    if if_status == "CRITICAL":
        return "WARNING"
    return "HEALTHY"


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


def compute_live_duration(
    msg: dict[str, Any],
    op_cols: list[str],
    engine_id: str,
    regime_detector,
    last_regime_label: dict[str, int],
    cycles_since_change: dict[str, int],
    window_size: int,
) -> float:
    """
    Mantiene estado por motor (régimen detectado en el mensaje anterior + cuántos
    ciclos lleva ahí) para poder calcular 'cycles_since_regime_change' fila por fila,
    igual que run_preprocessing.py/build_ros_gold.py lo calculan en batch. Si no hay
    detector ROS activo (detección deshabilitada por falta de datos sanos al construir
    ros_gold), la feature queda neutral (0.0) en vez de usar un detector NASA cruzado.
    """
    if regime_detector is None:
        return 0.0

    op_df = pd.DataFrame([{c: float(msg[c]) for c in op_cols}], columns=op_cols)
    label = int(regime_detector.predict(op_df)[0])

    prev_label = last_regime_label.get(engine_id)
    if prev_label is None or label != prev_label:
        cycles_since_change[engine_id] = 0
    else:
        cycles_since_change[engine_id] = min(cycles_since_change.get(engine_id, 0) + 1, window_size)
    last_regime_label[engine_id] = label

    return float(cycles_since_change[engine_id])


def preprocess_row(
    msg: dict[str, Any],
    sensor_cols: list[str],
    op_cols: list[str],
    regime_baseline: dict,
    global_scaler,
    engine_id: str,
    regime_detector,
    duration_scaler,
    last_regime_label: dict[str, int],
    cycles_since_change: dict[str, int],
    window_size: int,
) -> np.ndarray:
    missing = [c for c in sensor_cols + op_cols if c not in msg or msg[c] is None]
    if missing:
        raise KeyError(f"Mensaje ROS no tiene columnas requeridas: {missing}")

    row_df = pd.DataFrame(
        [{c: float(msg[c]) for c in op_cols + sensor_cols}],
        columns=op_cols + sensor_cols,
    )

    residual_df = apply_regime_baseline(row_df, regime_baseline)
    sensor_feat = global_scaler.transform(residual_df[sensor_cols])[0]

    duration_raw = compute_live_duration(
        msg=msg,
        op_cols=op_cols,
        engine_id=engine_id,
        regime_detector=regime_detector,
        last_regime_label=last_regime_label,
        cycles_since_change=cycles_since_change,
        window_size=window_size,
    )
    duration_scaled = float(duration_scaler.transform([[duration_raw]])[0, 0])

    feat = np.concatenate([sensor_feat, [duration_scaled]]).astype(np.float32)
    return feat


def log_event(text: str) -> None:
    print(text, flush=True)


def main() -> None:
    sensor_cols = choose_sensor_cols()
    op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]

    regime_baseline = joblib.load(resolve_artifact("regime_baseline.pkl"))
    global_scaler = joblib.load(resolve_artifact("healthy_global_scaler.pkl"))
    duration_scaler = joblib.load(resolve_artifact("regime_duration_scaler.pkl"))
    try:
        regime_detector = joblib.load(resolve_artifact("regime_detector_kmeans.pkl"))
    except FileNotFoundError:
        regime_detector = None
        log_event(
            "⚠️ No hay regime_detector_kmeans.pkl activo (detección deshabilitada al "
            f"construir ros_gold): '{DURATION_COL}' quedará neutral (0.0) para todos los motores."
        )

    if_model = joblib.load(resolve_artifact("isolation_forest.pkl"))

    calibration_path = Path(ARTIFACTS_DIR) / "score_calibration.json"
    calibration = load_json(str(calibration_path)) if calibration_path.exists() else None

    n_features = len(sensor_cols) + 1

    lstm = LSTMAutoencoder(n_features=n_features).to(DEVICE)
    lstm.load_state_dict(load_torch_state_dict(resolve_artifact("lstm_autoencoder.pt")))
    lstm.eval()

    tcn = TCNAutoencoder(n_features=n_features).to(DEVICE)
    tcn.load_state_dict(load_torch_state_dict(resolve_artifact("tcn_autoencoder.pt")))
    tcn.eval()

    transformer = TransformerAutoencoder(n_features=n_features).to(DEVICE)
    transformer.load_state_dict(load_torch_state_dict(resolve_artifact("transformer_autoencoder.pt")))
    transformer.eval()

    consumer = KafkaConsumer(
        TOPIC_IN,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        consumer_timeout_ms=1000,
    )

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda d: json.dumps(d).encode("utf-8"),
    )

    last_cycle_seen: dict[str, int] = {}
    windows: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))

    # Estado por motor para 'cycles_since_regime_change' en vivo.
    last_regime_label: dict[str, int] = {}
    cycles_since_change: dict[str, int] = {}

    baseline_if: dict[str, list[float]] = defaultdict(list)
    baseline_lstm: dict[str, list[float]] = defaultdict(list)
    baseline_tcn: dict[str, list[float]] = defaultdict(list)
    baseline_transformer: dict[str, list[float]] = defaultdict(list)

    hist_if: dict[str, list[float]] = defaultdict(list)
    hist_lstm: dict[str, list[float]] = defaultdict(list)
    hist_tcn: dict[str, list[float]] = defaultdict(list)
    hist_transformer: dict[str, list[float]] = defaultdict(list)

    log_event("📦 Cargando artifacts NASA/ROS...")
    log_event(f"✅ Inference lista | topic_in={TOPIC_IN} | topic_out={TOPIC_OUT} | sensors={sensor_cols}")

    while True:
        any_msg = False

        for message in consumer:
            any_msg = True
            msg = message.value

            engine_id = str(msg.get("engine_id"))
            cycle = int(msg.get("cycle"))

            # Inference solo toma 1 fila efectiva por ciclo.
            if last_cycle_seen.get(engine_id) == cycle:
                continue
            last_cycle_seen[engine_id] = cycle

            try:
                feat = preprocess_row(
                    msg=msg,
                    sensor_cols=sensor_cols,
                    op_cols=op_cols,
                    regime_baseline=regime_baseline,
                    global_scaler=global_scaler,
                    engine_id=engine_id,
                    regime_detector=regime_detector,
                    duration_scaler=duration_scaler,
                    last_regime_label=last_regime_label,
                    cycles_since_change=cycles_since_change,
                    window_size=WINDOW_SIZE,
                )
            except Exception as exc:
                log_event(f"⚠️ engine={engine_id} cycle={cycle} preprocessing_error={exc}")
                continue

            windows[engine_id].append(feat)

            if len(windows[engine_id]) < WINDOW_SIZE:
                if DEBUG_METRICS:
                    log_event(f"ℹ️ engine={engine_id} cycle={cycle} warmup={len(windows[engine_id])}/{WINDOW_SIZE}")
                continue

            x = np.asarray(windows[engine_id], dtype=np.float32)
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

            if len(baseline_if[engine_id]) < BASELINE_SIZE:
                baseline_if[engine_id].append(if_score)
                baseline_lstm[engine_id].append(lstm_score)
                baseline_tcn[engine_id].append(tcn_score)
                baseline_transformer[engine_id].append(transformer_score)

                n = len(baseline_if[engine_id])
                if n % max(1, BASELINE_LOG_EVERY) == 0:
                    log_event(f"ℹ️ engine={engine_id} cycle={cycle} baseline={n}/{BASELINE_SIZE}")
                continue

            lstm_status, lstm_delta, lstm_z, lstm_mean, lstm_warn, lstm_crit, lstm_trend = compute_status(
                lstm_score, hist_lstm[engine_id], baseline_lstm[engine_id]
            )
            tcn_status, tcn_delta, tcn_z, tcn_mean, tcn_warn, tcn_crit, tcn_trend = compute_status(
                tcn_score, hist_tcn[engine_id], baseline_tcn[engine_id]
            )
            transformer_status, transformer_delta, transformer_z, transformer_mean, transformer_warn, transformer_crit, transformer_trend = compute_status(
                transformer_score, hist_transformer[engine_id], baseline_transformer[engine_id]
            )
            if_status, if_delta, if_z, if_mean, if_warn, if_crit, if_trend = compute_status(
                if_score, hist_if[engine_id], baseline_if[engine_id]
            )

            final_status = fuse_status(lstm_status, tcn_status, transformer_status, if_status)

            hist_if[engine_id].append(if_score)
            hist_lstm[engine_id].append(lstm_score)
            hist_tcn[engine_id].append(tcn_score)
            hist_transformer[engine_id].append(transformer_score)

            # Payload compacto: solo lo que necesita el consumidor downstream (Grafana, alertas).
            # Los scores detallados por modelo permiten debugging sin saturar el topic.
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "engine_id": engine_id,
                "cycle": cycle,
                "if_score": round(float(if_score), 4),
                "lstm_score": round(float(lstm_score), 4),
                "tcn_score": round(float(tcn_score), 4),
                "transformer_score": round(float(transformer_score), 4),
                "if_status": if_status,
                "lstm_status": lstm_status,
                "tcn_status": tcn_status,
                "transformer_status": transformer_status,
                "final_status": final_status,
            }

            producer.send(TOPIC_OUT, payload)

            if LOG_COMPACT:
                log_event(
                    f"engine={engine_id} cycle={cycle} | "
                    f"LSTM={lstm_status}({lstm_score:.4f}) "
                    f"TCN={tcn_status}({tcn_score:.4f}) "
                    f"Transformer={transformer_status}({transformer_score:.4f}) "
                    f"IF={if_status}({if_score:.4f}) -> {final_status}"
                )
            elif DEBUG_METRICS:
                log_event(json.dumps(payload, ensure_ascii=False))

        if not any_msg:
            time.sleep(0.5)


if __name__ == "__main__":
    main()