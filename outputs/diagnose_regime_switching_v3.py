from __future__ import annotations

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.cluster import KMeans

spark = (
    SparkSession.builder.appName("DiagnoseRegimeSwitchingV3")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# gold ya conserva op_settings crudos (sin transformar) junto a los sensores ya
# normalizados con la regresión nueva, no hace falta unir con silver.
gold = spark.read.parquet("hdfs://namenode:9000/user/root/nasa/gold/nasa_train_gold").toPandas()
spark.stop()

op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
sensor_cols = [c for c in gold.columns if c.startswith("sensor_")]

merged = gold.sort_values(["dataset_id", "engine_num", "cycle"]).reset_index(drop=True)

# KMeans SOLO para etiquetar aproximadamente los regímenes con fines de diagnóstico
# (no se usa para normalizar nada, solo para identificar puntos de "cambio de régimen").
diag_rows = []
for ds, g in merged.groupby("dataset_id"):
    unique_op = g[op_cols].drop_duplicates()
    k = min(6, len(unique_op))
    if k < 2:
        g = g.copy()
        g["diag_cluster"] = 0
    else:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        g = g.copy()
        g["diag_cluster"] = km.fit_predict(g[op_cols])
    diag_rows.append(g)

merged = pd.concat(diag_rows, ignore_index=True)

jump_rows = []
for (ds, eng), g in merged.groupby(["dataset_id", "engine_id"]):
    g = g.sort_values("cycle").reset_index(drop=True)
    vals = g[sensor_cols].to_numpy()
    clusters = g["diag_cluster"].to_numpy()
    if len(vals) < 2:
        continue
    diffs = np.abs(vals[1:] - vals[:-1]).mean(axis=1)
    is_switch = clusters[1:] != clusters[:-1]
    for d, sw in zip(diffs, is_switch):
        jump_rows.append({"dataset_id": ds, "is_switch": bool(sw), "mean_abs_diff": float(d)})

jump_df = pd.DataFrame(jump_rows)
print("\n=== [NUEVA normalización] Magnitud del salto: cambio de régimen (aprox, solo diagnóstico) vs sin cambio ===")
summary = jump_df.groupby(["dataset_id", "is_switch"])["mean_abs_diff"].mean()
print(summary)

print("\n=== Razón switch/no-switch (nueva normalización) ===")
for ds in sorted(jump_df["dataset_id"].unique()):
    sub = jump_df[jump_df["dataset_id"] == ds]
    if sub["is_switch"].nunique() < 2:
        continue
    no_sw = sub.loc[~sub["is_switch"], "mean_abs_diff"].mean()
    sw = sub.loc[sub["is_switch"], "mean_abs_diff"].mean()
    print(f"{ds}: no_switch={no_sw:.4f}  switch={sw:.4f}  ratio={sw/no_sw:.3f}")

print("\n=== Referencia (normalización VIEJA, ya medida antes) ===")
print("FD002: no_switch=0.710397 switch=0.820445 ratio=1.155")
print("FD004: no_switch=0.718120 switch=0.838688 ratio=1.168")

print("\n✅ diagnóstico v3 terminado")
