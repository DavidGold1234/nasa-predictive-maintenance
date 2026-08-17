from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# =========================================================
# CONFIG
# =========================================================
HDFS_SILVER_TRAIN = os.environ.get(
    "HDFS_SILVER_TRAIN",
    "hdfs://namenode:9000/user/root/nasa/silver/train_all",
)
HDFS_GOLD_TRAIN = os.environ.get(
    "HDFS_GOLD_TRAIN",
    "hdfs://namenode:9000/user/root/nasa/gold/nasa_train_gold",
)

FEATURES_JSON = os.environ.get("FEATURES_JSON", "/src/features/selected_features.json")
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "/data/models/artifacts")
HDFS_ARTIFACTS_ACTIVE = os.environ.get(
    "HDFS_ARTIFACTS_ACTIVE",
    "hdfs://namenode:9000/user/root/nasa/artifacts/active",
)
HDFS_ARTIFACTS_VERSIONS = os.environ.get(
    "HDFS_ARTIFACTS_VERSIONS",
    "hdfs://namenode:9000/user/root/nasa/artifacts/versions",
)
PREPROCESSING_RUN_ID = os.environ.get(
    "PREPROCESSING_RUN_ID",
    datetime.utcnow().strftime("preprocessing_%Y%m%d_%H%M%S"),
)

TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", "0.80"))
HEALTHY_RATIO = float(os.environ.get("HEALTHY_RATIO", "0.30"))
REGRESSION_DEGREE = int(os.environ.get("REGRESSION_DEGREE", "2"))
REGIME_DETECT_CLUSTERS = int(os.environ.get("REGIME_DETECT_CLUSTERS", "6"))
SEED = int(os.environ.get("SEED", "42"))
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "30"))
DURATION_COL = "cycles_since_regime_change"

# =========================================================
# HELPERS
# =========================================================
def fit_regime_baseline(
    healthy_df: pd.DataFrame,
    op_cols: list[str],
    sensor_cols: list[str],
    degree: int = 2,
) -> dict:
    """
    Ajusta una línea base CONTINUA sensor ~ f(op_settings) por regresión polinómica,
    en vez de una línea base discreta por cluster (KMeans + z-score por cluster).

    Motivo: en FD002/FD004 la condición operativa cambia ~82% de los ciclos
    (ver diagnóstico empírico), así que casi el 100% de las ventanas de 30 ciclos
    cruzan varios clusters. Con normalización discreta por cluster, cada ciclo de
    la ventana puede quedar centrado con una media/std distinta, metiendo saltos
    artificiales que no tienen relación con degradación real. La regresión da una
    línea base que varía suave y continuamente con la condición operativa, sin
    saltos discretos entre ciclos consecutivos.
    """
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


def apply_regime_baseline(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    """Resta la línea base (predicha por regresión o, en fallback, la media plana) => residuo."""
    sensor_cols = baseline["sensor_cols"]
    if baseline["regressor"] is not None:
        X = baseline["poly"].transform(df[baseline["op_cols"]].to_numpy())
        pred = baseline["regressor"].predict(X)
    else:
        pred = np.tile(baseline["flat_mean"], (len(df), 1))

    out = df.copy()
    out.loc[:, sensor_cols] = df[sensor_cols].to_numpy() - pred
    return out


def fit_regime_detector(healthy_df: pd.DataFrame, op_cols: list[str], n_clusters: int, seed: int) -> KMeans:
    """
    KMeans liviano SOLO para detectar cambios de régimen operativo (no para
    normalizar sensores — eso lo hace fit_regime_baseline). Sirve para derivar
    la feature 'cycles_since_regime_change'.

    Motivación (hallazgo empírico): el salto de sensores en FD002/FD004 al
    cambiar de régimen no se explica por el método de normalización (una
    regresión evaluada en los mismos ~6 puntos discretos de régimen da
    prácticamente el mismo salto que un lookup por cluster). Es más probable
    que sea una dinámica transitoria real: el sensor tarda unos ciclos en
    estabilizarse tras el cambio. Dársela explícitamente al modelo como
    feature le permite distinguir "acabo de cambiar de régimen" de
    "degradación real", en vez de tener que inferirlo solo de los sensores.
    """
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
    """Cuenta, por motor y en orden de ciclo, cuántos ciclos lleva el motor en el
    mismo régimen detectado (0 = el régimen acaba de cambiar en este ciclo)."""
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


def build_spark(app_name: str) -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_feature_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_sensor_cols(df: pd.DataFrame, cfg: dict) -> list[str]:
    feature_cols = cfg.get("features", [])
    sensors = [c for c in feature_cols if c.startswith("sensor_") and c in df.columns]
    if sensors:
        return sensors
    return [c for c in df.columns if c.startswith("sensor_")]


def stratified_engine_split(
    df: pd.DataFrame,
    train_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    train_engines: set[str] = set()
    test_engines: set[str] = set()

    for _, part in df.groupby("dataset_id"):
        engines = np.array(sorted(part["engine_id"].astype(str).unique()))
        if len(engines) == 1:
            train_engines.add(str(engines[0]))
            continue

        train_size = int(np.floor(len(engines) * train_ratio))
        train_size = max(1, min(len(engines) - 1, train_size))

        chosen = rng.choice(engines, size=train_size, replace=False)
        chosen_set = set(chosen.astype(str).tolist())
        train_engines |= chosen_set
        test_engines |= (set(engines.astype(str).tolist()) - chosen_set)

    return train_engines, test_engines


def add_healthy_flag(df: pd.DataFrame, split_col: str, healthy_ratio: float) -> pd.DataFrame:
    out = df.copy()
    out["is_healthy_train"] = 0

    train_mask = out[split_col] == "train"
    train_only = out.loc[train_mask].copy()
    if train_only.empty:
        return out

    healthy_idx: list[int] = []
    for _, g in train_only.groupby("engine_id"):
        max_cycle = float(g["cycle"].max())
        cutoff = max_cycle * healthy_ratio
        idx = g.index[g["cycle"] <= cutoff]
        healthy_idx.extend(idx.tolist())

    out.loc[healthy_idx, "is_healthy_train"] = 1
    return out


def hdfs_put_via_spark(spark: SparkSession, local_path: str, hdfs_dir: str, filename: str) -> str:
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
    spark: SparkSession,
    local_paths: Iterable[str],
    run_id: str,
    category: str = "preprocessing",
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


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    spark = build_spark("RunNASAPreprocessingToGold")
    print("📥 Leyendo NASA silver train_all...")
    pdf = spark.read.parquet(HDFS_SILVER_TRAIN).toPandas()

    if pdf.empty:
        raise RuntimeError("silver/train_all está vacío")

    pdf = pdf.sort_values(["dataset_id", "engine_num", "cycle"]).reset_index(drop=True)

    op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
    cfg = load_feature_config(FEATURES_JSON)
    sensor_cols = choose_sensor_cols(pdf, cfg)

    required_cols = ["dataset_id", "engine_id", "engine_num", "cycle"] + op_cols + sensor_cols
    missing = [c for c in required_cols if c not in pdf.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para preprocessing: {missing}")

    pdf = pdf[required_cols].copy()
    for c in ["engine_num", "cycle"] + op_cols + sensor_cols:
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce")

    pdf = pdf.dropna(subset=required_cols).copy()
    pdf["dataset_id"] = pdf["dataset_id"].astype(str)
    pdf["engine_id"] = pdf["engine_id"].astype(str)
    pdf["engine_num"] = pdf["engine_num"].astype(int)
    pdf["cycle"] = pdf["cycle"].astype(int)
    pdf = pdf.sort_values(["dataset_id", "engine_num", "cycle"]).reset_index(drop=True)

    print(f"📌 Filas silver NASA: {len(pdf)}")
    print(f"📌 Sensores finales: {sensor_cols}")

    # -----------------------------------------------------
    # 1) Split por motor
    # -----------------------------------------------------
    train_engines, _ = stratified_engine_split(pdf, TRAIN_RATIO, SEED)
    pdf["split"] = np.where(pdf["engine_id"].isin(train_engines), "train", "test")

    train_df = pdf[pdf["split"] == "train"].copy()
    test_df = pdf[pdf["split"] == "test"].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError("El split por motor dejó train o test vacío")

    # -----------------------------------------------------
    # 2) Marcar healthy SOLO dentro de train
    # -----------------------------------------------------
    pdf = add_healthy_flag(pdf, split_col="split", healthy_ratio=HEALTHY_RATIO)

    healthy_train_df = pdf[
        (pdf["split"] == "train") & (pdf["is_healthy_train"] == 1)
    ].copy()
    if healthy_train_df.empty:
        raise RuntimeError("No hubo filas healthy en train para ajustar preprocessing")

    # -----------------------------------------------------
    # 3) Línea base de régimen operativo (regresión) SOLO con healthy train
    # -----------------------------------------------------
    print(f"⚙️ Ajustando línea base sensor~f(op_settings) (grado={REGRESSION_DEGREE}) en healthy train...")
    regime_baseline = fit_regime_baseline(healthy_train_df, op_cols, sensor_cols, degree=REGRESSION_DEGREE)

    # -----------------------------------------------------
    # 3b) Feature "cycles_since_regime_change" (dinámica transitoria) SOLO con healthy train
    # -----------------------------------------------------
    print(f"⚙️ Ajustando detector de régimen (k<={REGIME_DETECT_CLUSTERS}) para '{DURATION_COL}'...")
    regime_detector = fit_regime_detector(healthy_train_df, op_cols, REGIME_DETECT_CLUSTERS, SEED)
    pdf = compute_cycles_since_regime_change(pdf, op_cols, regime_detector, DURATION_COL)
    # >=WINDOW_SIZE es indistinguible dentro de una ventana de WINDOW_SIZE ciclos.
    pdf[DURATION_COL] = pdf[DURATION_COL].clip(upper=WINDOW_SIZE).astype(float)

    # -----------------------------------------------------
    # 4) Residuo = sensor - línea base, aplicado a TODAS las filas
    # -----------------------------------------------------
    transformed_df = apply_regime_baseline(pdf, regime_baseline)

    # -----------------------------------------------------
    # 5) Normalización global SOLO con healthy train ya zscoreado
    # -----------------------------------------------------
    healthy_train_z_df = transformed_df[
        (transformed_df["split"] == "train") & (transformed_df["is_healthy_train"] == 1)
    ].copy()
    if healthy_train_z_df.empty:
        raise RuntimeError("No hubo filas healthy train zscoreadas para ajustar healthy_global_scaler")

    healthy_global_scaler = StandardScaler()
    healthy_global_scaler.fit(healthy_train_z_df[sensor_cols])
    transformed_df.loc[:, sensor_cols] = healthy_global_scaler.transform(transformed_df[sensor_cols])

    duration_scaler = StandardScaler()
    duration_scaler.fit(healthy_train_z_df[[DURATION_COL]])
    transformed_df.loc[:, [DURATION_COL]] = duration_scaler.transform(transformed_df[[DURATION_COL]])

    # -----------------------------------------------------
    # 6) Gold final
    # -----------------------------------------------------
    gold_cols = [
        "dataset_id",
        "engine_id",
        "engine_num",
        "cycle",
        "split",
        "is_healthy_train",
        *op_cols,
        *sensor_cols,
        DURATION_COL,
    ]

    gold_pdf = (
        transformed_df[gold_cols]
        .sort_values(["dataset_id", "engine_num", "cycle"])
        .reset_index(drop=True)
    )

    print(f"🥇 Escribiendo gold en: {HDFS_GOLD_TRAIN}")
    spark.createDataFrame(gold_pdf).write.mode("overwrite").parquet(HDFS_GOLD_TRAIN)

    # -----------------------------------------------------
    # 7) Guardar artifacts locales
    # -----------------------------------------------------
    print("💾 Guardando artifacts activos locales...")
    local_feature_path = os.path.join(ARTIFACTS_DIR, "selected_features.json")
    local_regime_baseline_path = os.path.join(ARTIFACTS_DIR, "regime_baseline.pkl")
    local_scaler_path = os.path.join(ARTIFACTS_DIR, "healthy_global_scaler.pkl")
    local_regime_detector_path = os.path.join(ARTIFACTS_DIR, "regime_detector_kmeans.pkl")
    local_duration_scaler_path = os.path.join(ARTIFACTS_DIR, "regime_duration_scaler.pkl")
    local_meta_path = os.path.join(ARTIFACTS_DIR, "preprocessing_metadata.json")

    shutil.copy2(FEATURES_JSON, local_feature_path)
    joblib.dump(regime_baseline, local_regime_baseline_path)
    joblib.dump(healthy_global_scaler, local_scaler_path)
    joblib.dump(regime_detector, local_regime_detector_path)
    joblib.dump(duration_scaler, local_duration_scaler_path)

    preprocessing_metadata = {
        "run_id": PREPROCESSING_RUN_ID,
        "created_at_utc": datetime.utcnow().isoformat(),
        "silver_input_path": HDFS_SILVER_TRAIN,
        "gold_output_path": HDFS_GOLD_TRAIN,
        "selected_features_file": str(Path(FEATURES_JSON).name),
        "sensor_cols": sensor_cols,
        "duration_col": DURATION_COL,
        "model_input_cols": sensor_cols + [DURATION_COL],
        "op_setting_cols": op_cols,
        "train_ratio": TRAIN_RATIO,
        "healthy_ratio": HEALTHY_RATIO,
        "regression_degree": REGRESSION_DEGREE,
        "regime_detect_clusters_requested": REGIME_DETECT_CLUSTERS,
        "regime_detect_clusters_fitted": int(regime_detector.n_clusters),
        "split_strategy": "engine_stratified_by_dataset",
        "split_seed": SEED,
        "window_size_reference": WINDOW_SIZE,
        "preprocessing_order": [
            "engine_split",
            "healthy_train_flag",
            "regime_baseline_fit_on_healthy_train",
            "regime_detector_fit_on_healthy_train",
            "cycles_since_regime_change_all_rows",
            "regime_baseline_residual_all_rows",
            "global_scaler_fit_on_healthy_train_residual",
            "global_scaler_transform_all_rows",
            "duration_scaler_fit_on_healthy_train",
            "duration_scaler_transform_all_rows",
        ],
        "gold_columns": gold_cols,
        "n_rows_total": int(len(gold_pdf)),
        "n_rows_train": int((gold_pdf["split"] == "train").sum()),
        "n_rows_test": int((gold_pdf["split"] == "test").sum()),
        "n_train_engines": int(gold_pdf.loc[gold_pdf["split"] == "train", "engine_id"].nunique()),
        "n_test_engines": int(gold_pdf.loc[gold_pdf["split"] == "test", "engine_id"].nunique()),
        "n_healthy_train_rows": int(gold_pdf["is_healthy_train"].sum()),
        "hdfs_artifacts_active_root": HDFS_ARTIFACTS_ACTIVE,
        "hdfs_artifacts_versions_root": HDFS_ARTIFACTS_VERSIONS,
    }
    with open(local_meta_path, "w", encoding="utf-8") as f:
        json.dump(preprocessing_metadata, f, indent=2)

    # -----------------------------------------------------
    # 8) Respaldo HDFS
    # -----------------------------------------------------
    local_artifacts = [
        local_feature_path,
        local_regime_baseline_path,
        local_scaler_path,
        local_regime_detector_path,
        local_duration_scaler_path,
        local_meta_path,
    ]

    print("☁️ Respaldando artifacts de preprocessing en HDFS...")
    backup_manifest = backup_artifacts_to_hdfs(
        spark,
        local_artifacts,
        PREPROCESSING_RUN_ID,
        category="preprocessing",
    )

    manifest_path = os.path.join(ARTIFACTS_DIR, "preprocessing_backup_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"run_id": PREPROCESSING_RUN_ID, "files": backup_manifest}, f, indent=2)

    hdfs_put_via_spark(
        spark,
        manifest_path,
        f"{HDFS_ARTIFACTS_VERSIONS.rstrip('/')}/{PREPROCESSING_RUN_ID}/preprocessing",
        Path(manifest_path).name,
    )

    print("✅ Preprocessing NASA terminado")
    print(f"✅ Gold listo: {HDFS_GOLD_TRAIN}")
    print(f"✅ Artifacts activos locales: {ARTIFACTS_DIR}")
    print(f"✅ Respaldo HDFS run_id: {PREPROCESSING_RUN_ID}")

    spark.stop()


if __name__ == "__main__":
    main()