from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# NOTA: fit_regime_baseline/apply_regime_baseline duplican la lógica de
# src/preprocessing/run_preprocessing.py (misma convención del proyecto de
# no compartir un módulo común entre etapas Spark independientes).


HDFS_ROS_SILVER = os.environ.get(
    "HDFS_ROS_SILVER",
    "hdfs://namenode:9000/user/root/nasa/silver/ros_clean",
)
HDFS_ROS_GOLD = os.environ.get(
    "HDFS_ROS_GOLD",
    "hdfs://namenode:9000/user/root/nasa/gold/ros_gold",
)
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "/data/models/artifacts")

# HEALTHY_RATIO: fracción inicial de ciclos de cada motor que se considera sana.
# Se usa para ajustar la línea base ROS a partir del baseline propio del motor.
# Debe coincidir con el valor usado en run_preprocessing.py y retrain_ros_models.py.
HEALTHY_RATIO = float(os.environ.get("HEALTHY_RATIO", "0.30"))

REGRESSION_DEGREE = int(os.environ.get("REGRESSION_DEGREE", "2"))

# MIN_HEALTHY_ROWS_FOR_REGRESSION: mínimo de filas sanas ROS necesarias para ajustar
# una línea base por regresión propia. Si hay menos, se usa una línea base plana
# (media sana ROS, sin dependencia de op_settings) en vez de extrapolar la regresión
# NASA fuera de su rango de entrenamiento (los op_settings de ROS son magnitudes
# físicas distintas —torque/pendiente/temperatura— a las de NASA).
MIN_HEALTHY_ROWS_FOR_REGRESSION = int(os.environ.get("MIN_HEALTHY_ROWS_FOR_REGRESSION", "30"))

# MIN_HEALTHY_ROWS_FOR_GLOBAL_SCALER: mínimo de filas sanas ROS (ya con el residuo
# aplicado) necesarias para ajustar el global scaler ROS. Si hay menos, se usa el
# global scaler NASA como fallback (aquí sí es seguro: StandardScaler no extrapola).
MIN_HEALTHY_ROWS_FOR_GLOBAL_SCALER = int(os.environ.get("MIN_HEALTHY_ROWS_FOR_GLOBAL_SCALER", "10"))

SEED = int(os.environ.get("SEED", "42"))
REGIME_DETECT_CLUSTERS = int(os.environ.get("REGIME_DETECT_CLUSTERS", "6"))
DURATION_COL = "cycles_since_regime_change"
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "30"))

# MIN_HEALTHY_ROWS_FOR_DETECTOR: mínimo de filas sanas ROS para ajustar un detector
# de régimen (KMeans) propio. A diferencia del global scaler, NO se usa el detector
# NASA como fallback (vive en el espacio numérico de los op_settings NASA — altitud/
# Mach/acelerador —, muy distinto al de ROS —torque/pendiente/temperatura—; casi
# todas las filas ROS caerían en el mismo cluster NASA, dejando la detección inútil).
# Con pocos datos, se deshabilita la feature (cycles_since_regime_change=0 siempre)
# en vez de dar una señal engañosa.
MIN_HEALTHY_ROWS_FOR_DETECTOR = int(os.environ.get("MIN_HEALTHY_ROWS_FOR_DETECTOR", "30"))


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_sensor_cols(pdf: pd.DataFrame, artifact_dir: str) -> list[str]:
    """Usa exactamente los mismos sensores definidos en selected_features.json (resultado del EDA)."""
    cfg = load_json(os.path.join(artifact_dir, "selected_features.json"))
    cols = [c for c in cfg.get("features", []) if c.startswith("sensor_") and c in pdf.columns]
    if cols:
        return cols
    return sorted([c for c in pdf.columns if c.startswith("sensor_")])


def get_healthy_index(
    pdf: pd.DataFrame,
    healthy_ratio: float,
    op_cols: list[str] | None = None,
    n_regime_clusters: int = REGIME_DETECT_CLUSTERS,
    seed: int = SEED,
) -> list[int]:
    """
    Retorna los índices del DataFrame que representan el baseline sano de cada motor.

    Si se pasa `op_cols`, la selección es consciente de régimen: en vez de tomar solo
    los primeros HEALTHY_RATIO ciclos cronológicos del motor, se agrupa primero por
    régimen operativo (KMeans liviano sobre op_cols, propio de cada motor) y se toman
    los primeros HEALTHY_RATIO ciclos DENTRO de cada régimen. Esto evita que el
    baseline sano quede compuesto solo por el primer régimen que visita el motor
    (real para simuladores/operaciones con saltos de régimen discretos y poco
    frecuentes, donde "los primeros N% de ciclos" puede caer enteramente antes del
    primer cambio de régimen, dejando la línea base y el detector de régimen sin
    ejemplos sanos de los regímenes posteriores — ver Etapa 8 en
    docs/ExperimentosNASA_2026-08-16.md). Sin `op_cols`, mantiene el comportamiento
    original (prefijo cronológico simple).
    """
    healthy_idx: list[int] = []
    for _, g in pdf.groupby("engine_id"):
        if op_cols is None or len(g) < 2:
            min_c = int(g["cycle"].min())
            max_c = int(g["cycle"].max())
            span = max_c - min_c + 1
            cutoff = min_c + max(1, int(np.ceil(span * healthy_ratio))) - 1
            idx = g.index[g["cycle"] <= cutoff].tolist()
            healthy_idx.extend(idx)
            continue

        unique_op_rows = g[op_cols].drop_duplicates()
        k = max(1, min(n_regime_clusters, len(unique_op_rows), len(g)))
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(g[op_cols])
        g_labeled = g.assign(_regime_label=labels)

        for _, gg in g_labeled.groupby("_regime_label"):
            gg_sorted = gg.sort_values("cycle")
            n_take = max(1, int(np.ceil(len(gg_sorted) * healthy_ratio)))
            healthy_idx.extend(gg_sorted.index[:n_take].tolist())

    return healthy_idx


def fit_regime_baseline(
    healthy_df: pd.DataFrame,
    op_cols: list[str],
    sensor_cols: list[str],
    degree: int = 2,
) -> dict:
    """Debe coincidir con la lógica de run_preprocessing.py."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X = poly.fit_transform(healthy_df[op_cols].to_numpy())
    regressor = LinearRegression()
    regressor.fit(X, healthy_df[sensor_cols].to_numpy())
    return {
        "poly": poly,
        "regressor": regressor,
        "flat_mean": None,
        "op_cols": op_cols,
        "sensor_cols": sensor_cols,
        "raw_mean": healthy_df[sensor_cols].mean().to_numpy(),
        "raw_std": healthy_df[sensor_cols].std().to_numpy(),
    }


def fit_flat_baseline(healthy_df: pd.DataFrame, op_cols: list[str], sensor_cols: list[str]) -> dict:
    """Línea base sin dependencia de op_settings (media sana), para cuando no hay
    suficientes filas ROS sanas para ajustar una regresión propia con confianza."""
    mean_vec = healthy_df[sensor_cols].mean().to_numpy()
    return {
        "poly": None,
        "regressor": None,
        "flat_mean": mean_vec,
        "op_cols": op_cols,
        "sensor_cols": sensor_cols,
        "raw_mean": mean_vec,
        "raw_std": healthy_df[sensor_cols].std().to_numpy(),
    }


def apply_regime_baseline(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    sensor_cols = baseline["sensor_cols"]
    if baseline["regressor"] is not None:
        X = baseline["poly"].transform(df[baseline["op_cols"]].to_numpy())
        pred = baseline["regressor"].predict(X)
    else:
        pred = np.tile(baseline["flat_mean"], (len(df), 1))

    out = df.copy()
    out.loc[:, sensor_cols] = df[sensor_cols].to_numpy() - pred
    return out


def fit_ros_regime_baseline(
    pdf: pd.DataFrame,
    op_cols: list[str],
    sensor_cols: list[str],
    healthy_idx: list[int],
    nasa_global_scaler: StandardScaler,
    degree: int,
) -> tuple[dict, StandardScaler, pd.DataFrame]:
    """
    Ajusta una línea base de régimen operativo específica de ROS usando el baseline
    sano del propio motor (en vez de reusar la línea base NASA: los op_settings de
    ROS son magnitudes físicas distintas —torque/pendiente/temperatura— a las de
    NASA, y una regresión extrapola mal fuera de su rango de entrenamiento; por eso
    el fallback con pocos datos es una línea base PLANA, no la regresión NASA).
    """
    pdf_healthy = pdf.loc[healthy_idx].copy()

    if len(pdf_healthy) >= MIN_HEALTHY_ROWS_FOR_REGRESSION:
        ros_baseline = fit_regime_baseline(pdf_healthy, op_cols, sensor_cols, degree=degree)
        print(f"  ✅ Línea base ROS (regresión) ajustada con {len(pdf_healthy)} filas sanas.")
    elif len(pdf_healthy) > 0:
        ros_baseline = fit_flat_baseline(pdf_healthy, op_cols, sensor_cols)
        print(
            f"  ⚠️  Solo {len(pdf_healthy)} filas sanas ROS (mínimo={MIN_HEALTHY_ROWS_FOR_REGRESSION} "
            f"para regresión). Usando línea base plana (media sana ROS, sin dependencia de op_settings)."
        )
    else:
        raise RuntimeError("Sin filas ROS sanas para ajustar ninguna línea base.")

    pdf_residual = apply_regime_baseline(pdf, ros_baseline)

    healthy_residual = pdf_residual.loc[healthy_idx]
    if len(healthy_residual) < MIN_HEALTHY_ROWS_FOR_GLOBAL_SCALER:
        print("  ⚠️  Muy pocos datos sanos para global scaler ROS. Usando global scaler NASA.")
        ros_global_scaler = nasa_global_scaler
    else:
        ros_global_scaler = StandardScaler()
        ros_global_scaler.fit(healthy_residual[sensor_cols])
        print(f"  ✅ Global scaler ROS ajustado con {len(healthy_residual)} filas sanas (residuo).")

    return ros_baseline, ros_global_scaler, pdf_residual


def fit_regime_detector(healthy_df: pd.DataFrame, op_cols: list[str], n_clusters: int, seed: int) -> KMeans:
    """Debe coincidir con la lógica de run_preprocessing.py."""
    unique_op_rows = healthy_df[op_cols].drop_duplicates()
    k = max(1, min(n_clusters, len(unique_op_rows), len(healthy_df)))
    detector = KMeans(n_clusters=k, random_state=seed, n_init=10)
    detector.fit(healthy_df[op_cols])
    return detector


def compute_cycles_since_regime_change(
    pdf: pd.DataFrame,
    op_cols: list[str],
    detector: KMeans,
    duration_col: str,
) -> pd.DataFrame:
    """Debe coincidir con la lógica de run_preprocessing.py."""
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


def fit_ros_regime_duration(
    pdf: pd.DataFrame,
    op_cols: list[str],
    healthy_idx: list[int],
    nasa_duration_scaler: StandardScaler,
    n_clusters: int,
    seed: int,
    duration_col: str,
    window_size: int,
) -> tuple[pd.DataFrame, StandardScaler, "KMeans | None"]:
    """
    Calcula 'cycles_since_regime_change' para ROS con un detector propio (nunca el
    de NASA, ver comentario de MIN_HEALTHY_ROWS_FOR_DETECTOR). Si no hay datos
    suficientes para un detector confiable, la feature queda deshabilitada (0
    constante) en vez de dar una señal cruzada entre dominios que no tiene sentido.
    """
    pdf_healthy = pdf.loc[healthy_idx]

    if len(pdf_healthy) >= MIN_HEALTHY_ROWS_FOR_DETECTOR:
        detector = fit_regime_detector(pdf_healthy, op_cols, n_clusters, seed)
        pdf_out = compute_cycles_since_regime_change(pdf, op_cols, detector, duration_col)
        print(f"  ✅ Detector de régimen ROS ajustado con {len(pdf_healthy)} filas sanas (k={detector.n_clusters}).")
    else:
        detector = None
        pdf_out = pdf.reset_index(drop=True).copy()
        pdf_out[duration_col] = 0.0
        print(
            f"  ⚠️  Solo {len(pdf_healthy)} filas sanas ROS (mínimo={MIN_HEALTHY_ROWS_FOR_DETECTOR} para "
            f"detector). Deshabilitando '{duration_col}' (0 constante) en vez de usar el detector NASA "
            "(vive en un espacio numérico distinto al de ROS)."
        )

    pdf_out[duration_col] = pdf_out[duration_col].clip(upper=window_size).astype(float)

    healthy_duration = pdf_out.loc[healthy_idx, [duration_col]]
    if len(healthy_duration) < MIN_HEALTHY_ROWS_FOR_GLOBAL_SCALER:
        print("  ⚠️  Muy pocos datos sanos para duration scaler ROS. Usando duration scaler NASA.")
        duration_scaler = nasa_duration_scaler
    else:
        duration_scaler = StandardScaler()
        duration_scaler.fit(healthy_duration)
        print(f"  ✅ Duration scaler ROS ajustado con {len(healthy_duration)} filas sanas.")

    return pdf_out, duration_scaler, detector


def check_distribution_compatibility(
    pdf: pd.DataFrame,
    sensor_cols: list[str],
    nasa_baseline: dict,
) -> None:
    """
    Compara la distribución RAW de los sensores ROS contra la referencia sana NASA
    (raw_mean/raw_std guardados en el artifact al ajustar la línea base).
    Emite advertencia si algún sensor difiere más de 3σ en media respecto a NASA.
    Sirve para detectar cuándo los datos ROS requieren adaptación de dominio.
    """
    nasa_mean = nasa_baseline["raw_mean"]
    nasa_std = nasa_baseline["raw_std"]

    ros_stats = pdf[sensor_cols].agg(["mean", "std"])
    warnings_found = []

    for i, col in enumerate(sensor_cols):
        ros_mean = float(ros_stats.loc["mean", col])
        std_ref = float(nasa_std[i]) if nasa_std[i] > 1e-9 else 1.0
        mean_diff_sigma = abs(ros_mean - float(nasa_mean[i])) / std_ref
        if mean_diff_sigma > 3.0:
            warnings_found.append(
                f"  {col}: media_ROS={ros_mean:.3f}  media_NASA={nasa_mean[i]:.3f}  "
                f"diferencia={mean_diff_sigma:.1f}σ → se usará línea base ROS-específica"
            )

    if warnings_found:
        print(
            "\n⚠️  Distribución ROS difiere de NASA en los siguientes sensores:\n"
            + "\n".join(warnings_found)
            + "\n   → La línea base ROS-específica ajustada en este paso resolverá el desfase.\n"
        )
    else:
        print("✅ Distribución ROS compatible con artefactos NASA (diferencias < 3σ en todos los sensores).")


def latest_active_version(artifact_dir: str) -> str:
    """
    Nunca retorna None: si la columna resultante queda con None en todas las filas
    (p. ej. primera corrida, antes de que exista active_version.json), pandas la deja
    como dtype 'object' y Spark no puede inferirle un tipo al crear el DataFrame
    (CANNOT_DETERMINE_TYPE). "unknown" mantiene la columna como string siempre.
    """
    path = Path(artifact_dir) / "active_version.json"
    if not path.exists():
        return "unknown"
    try:
        data = load_json(str(path))
        return str(data.get("version") or data.get("active_version") or "unknown")
    except Exception:
        return "unknown"


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("BuildROSGold")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    pdf = spark.read.parquet(HDFS_ROS_SILVER).toPandas()
    if pdf.empty:
        raise RuntimeError("ROS silver está vacío")

    required_artifacts = [
        "selected_features.json",
        "regime_baseline.pkl",
        "healthy_global_scaler.pkl",
        "regime_duration_scaler.pkl",
    ]
    missing_artifacts = [name for name in required_artifacts if not (Path(ARTIFACTS_DIR) / name).exists()]
    if missing_artifacts:
        raise FileNotFoundError(f"Faltan artifacts activos para ROS gold: {missing_artifacts}")

    op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
    # Usa exactamente los mismos sensores del EDA (selected_features.json), sin reducir.
    sensor_cols = choose_sensor_cols(pdf, ARTIFACTS_DIR)

    keep_cols = [
        c for c in [
            "timestamp", "event_ts", "ingest_ts",
            "dataset_id", "engine_id", "engine_num", "cycle",
            *op_cols, *sensor_cols,
        ] if c in pdf.columns
    ]
    pdf = pdf[keep_cols].copy()

    required_cols = ["dataset_id", "engine_id", "engine_num", "cycle", *op_cols, *sensor_cols]
    missing = [c for c in required_cols if c not in pdf.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para ROS gold: {missing}")

    # Silver conserva el raw; gold colapsa a 1 fila efectiva por ciclo.
    order_cols = [c for c in ["event_ts", "ingest_ts", "timestamp"] if c in pdf.columns]
    if order_cols:
        pdf = pdf.sort_values(["dataset_id", "engine_id", "cycle", *order_cols])
    else:
        pdf = pdf.sort_values(["dataset_id", "engine_id", "cycle"])
    pdf = pdf.drop_duplicates(subset=["dataset_id", "engine_id", "cycle"], keep="last").reset_index(drop=True)

    for c in ["engine_num", "cycle", *op_cols, *sensor_cols]:
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce")
    pdf = pdf.dropna(subset=required_cols).copy()
    pdf["engine_num"] = pdf["engine_num"].astype(int)
    pdf["cycle"] = pdf["cycle"].astype(int)

    # --- Cargar artefactos NASA ---
    nasa_baseline: dict = joblib.load(os.path.join(ARTIFACTS_DIR, "regime_baseline.pkl"))
    nasa_global_scaler: StandardScaler = joblib.load(
        os.path.join(ARTIFACTS_DIR, "healthy_global_scaler.pkl")
    )
    nasa_duration_scaler: StandardScaler = joblib.load(
        os.path.join(ARTIFACTS_DIR, "regime_duration_scaler.pkl")
    )

    # Verificar desfase de distribución antes de adaptar
    check_distribution_compatibility(pdf, sensor_cols, nasa_baseline)

    # --- Identificar baseline sano ROS (primeros HEALTHY_RATIO de ciclos de cada motor,
    # calculado por régimen para no dejar fuera regímenes que el motor visita más tarde) ---
    healthy_idx = get_healthy_index(pdf, HEALTHY_RATIO, op_cols=op_cols)
    print(f"\n📐 Ajustando línea base ROS-específica sobre {len(healthy_idx)} filas sanas ({HEALTHY_RATIO*100:.0f}% inicial por régimen)...")

    # --- Ajustar línea base ROS-específica y aplicarla ---
    # Esto resuelve el desfase de distribución entre NASA y ROS sin perder la estructura
    # de degradación aprendida. El modelo aprende "qué es normal en ROS" y detecta desviaciones.
    ros_baseline, ros_global_scaler, pdf_residual = fit_ros_regime_baseline(
        pdf=pdf,
        op_cols=op_cols,
        sensor_cols=sensor_cols,
        healthy_idx=healthy_idx,
        nasa_global_scaler=nasa_global_scaler,
        degree=REGRESSION_DEGREE,
    )

    out = pdf_residual.copy()
    out.loc[:, sensor_cols] = ros_global_scaler.transform(out[sensor_cols])

    # --- Feature de dinámica transitoria: cycles_since_regime_change ---
    print(f"\n📐 Calculando '{DURATION_COL}' con detector de régimen ROS-específico...")
    out, ros_duration_scaler, ros_regime_detector = fit_ros_regime_duration(
        pdf=out,
        op_cols=op_cols,
        healthy_idx=healthy_idx,
        nasa_duration_scaler=nasa_duration_scaler,
        n_clusters=REGIME_DETECT_CLUSTERS,
        seed=SEED,
        duration_col=DURATION_COL,
        window_size=WINDOW_SIZE,
    )
    out.loc[:, [DURATION_COL]] = ros_duration_scaler.transform(out[[DURATION_COL]])

    # --- Guardar línea base y detector ROS-específicos como artifacts activos ---
    # Sobrescriben los NASA para que retrain_ros_models.py, evaluate_ros_from_gold.py
    # y ros_kafka_inference.py usen automáticamente la normalización/feature adaptadas.
    # Los originales NASA están respaldados en HDFS (artifacts/versions/).
    ros_baseline_path = os.path.join(ARTIFACTS_DIR, "regime_baseline.pkl")
    ros_global_path = os.path.join(ARTIFACTS_DIR, "healthy_global_scaler.pkl")
    ros_duration_scaler_path = os.path.join(ARTIFACTS_DIR, "regime_duration_scaler.pkl")
    ros_detector_path = os.path.join(ARTIFACTS_DIR, "regime_detector_kmeans.pkl")
    joblib.dump(ros_baseline, ros_baseline_path)
    joblib.dump(ros_global_scaler, ros_global_path)
    joblib.dump(ros_duration_scaler, ros_duration_scaler_path)
    if ros_regime_detector is not None:
        joblib.dump(ros_regime_detector, ros_detector_path)
    else:
        # Sin detector ROS confiable: se elimina el activo (si existe) para que
        # ros_kafka_inference.py sepa que debe deshabilitar la feature (0 constante)
        # en vez de usar por error el detector NASA (espacio numérico incompatible).
        Path(ros_detector_path).unlink(missing_ok=True)
    print("✅ Línea base / detector de régimen ROS-específicos guardados en artifacts activos:")
    print(f"   {ros_baseline_path}")
    print(f"   {ros_global_path}")
    print(f"   {ros_duration_scaler_path}")
    print(f"   {ros_detector_path} {'(sin cambios: detección deshabilitada)' if ros_regime_detector is None else ''}")

    # --- Construir gold ROS ---
    out["gold_ready_ts_utc"] = datetime.now(timezone.utc).isoformat()
    out["artifact_version"] = latest_active_version(ARTIFACTS_DIR)

    gold_cols = [
        "timestamp", "event_ts", "ingest_ts",
        "gold_ready_ts_utc", "artifact_version",
        "dataset_id", "engine_id", "engine_num", "cycle",
        *sensor_cols,
        DURATION_COL,
    ]
    gold_cols = [c for c in gold_cols if c in out.columns]
    gold_pdf = out[gold_cols].copy()

    spark.createDataFrame(gold_pdf).write.mode("overwrite").parquet(HDFS_ROS_GOLD)

    print(f"\n✅ ROS gold generado en: {HDFS_ROS_GOLD}")
    print(f"✅ Filas gold ROS: {len(gold_pdf)}")
    print(f"✅ Sensores gold: {sensor_cols}")
    print(
        "\n📌 Siguiente paso recomendado: ejecutar retrain_ros_models.py para que los modelos\n"
        "   (LSTM-AE, TCN-AE, Transformer-AE, Isolation Forest) aprendan también los patrones ROS sanos\n"
        "   usando esta normalización ROS-específica.\n"
    )

    spark.stop()


if __name__ == "__main__":
    main()
