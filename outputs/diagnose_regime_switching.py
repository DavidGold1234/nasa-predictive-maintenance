from __future__ import annotations

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

WINDOW_SIZE = 30

spark = (
    SparkSession.builder.appName("DiagnoseRegimeSwitching")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

pdf = spark.read.parquet("hdfs://namenode:9000/user/root/nasa/gold/nasa_train_gold").toPandas()
spark.stop()

pdf = pdf.sort_values(["dataset_id", "engine_num", "cycle"]).reset_index(drop=True)
sensor_cols = [c for c in pdf.columns if c.startswith("sensor_")]

print("Filas totales:", len(pdf))
print("Datasets:", sorted(pdf["dataset_id"].unique().tolist()))
print("\nClusters distintos usados por dataset:")
print(pdf.groupby("dataset_id")["operating_cluster"].nunique())

switch_rows = []
multi_cluster_rows = []
jump_rows = []

for (ds, eng), g in pdf.groupby(["dataset_id", "engine_id"]):
    g = g.sort_values("cycle").reset_index(drop=True)
    clusters = g["operating_cluster"].to_numpy()
    vals = g[sensor_cols].to_numpy()

    if len(clusters) >= 2:
        switches = np.sum(clusters[1:] != clusters[:-1])
        switch_rows.append(
            {"dataset_id": ds, "n_cycles": len(clusters), "switch_rate": switches / (len(clusters) - 1)}
        )
        diffs = np.abs(vals[1:] - vals[:-1]).mean(axis=1)
        is_switch = clusters[1:] != clusters[:-1]
        for d, sw in zip(diffs, is_switch):
            jump_rows.append({"dataset_id": ds, "is_switch": bool(sw), "mean_abs_diff": float(d)})

    if len(g) >= WINDOW_SIZE:
        n_windows = 0
        n_multi = 0
        max_unique = 0
        for end in range(WINDOW_SIZE - 1, len(g)):
            w = clusters[end - WINDOW_SIZE + 1 : end + 1]
            u = len(np.unique(w))
            n_windows += 1
            max_unique = max(max_unique, u)
            if u > 1:
                n_multi += 1
        multi_cluster_rows.append(
            {
                "dataset_id": ds,
                "n_windows": n_windows,
                "frac_multi_cluster": n_multi / n_windows if n_windows else np.nan,
                "max_unique_clusters_in_window": max_unique,
            }
        )

switch_df = pd.DataFrame(switch_rows)
mc_df = pd.DataFrame(multi_cluster_rows)
jump_df = pd.DataFrame(jump_rows)

print("\n=== Tasa de cambio de cluster operativo entre ciclos consecutivos (promedio por motor), por dataset ===")
print(switch_df.groupby("dataset_id")["switch_rate"].mean())

print("\n=== Fracción de ventanas de 30 ciclos que abarcan >1 cluster operativo, por dataset ===")
print(mc_df.groupby("dataset_id")["frac_multi_cluster"].mean())

print("\n=== Máximo de clusters distintos observados dentro de UNA ventana, por dataset ===")
print(mc_df.groupby("dataset_id")["max_unique_clusters_in_window"].max())

print("\n=== Magnitud promedio del salto entre ciclos consecutivos (|delta| medio entre sensores, ya normalizados) ===")
print("=== comparando cuando SÍ hubo cambio de cluster vs cuando NO, por dataset ===")
print(jump_df.groupby(["dataset_id", "is_switch"])["mean_abs_diff"].mean())

print("\n✅ diagnóstico terminado")
