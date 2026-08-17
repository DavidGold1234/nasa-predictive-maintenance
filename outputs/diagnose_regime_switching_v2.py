from __future__ import annotations

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("DiagnoseRegimeSwitchingV2")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

pdf = spark.read.parquet("hdfs://namenode:9000/user/root/nasa/gold/nasa_train_gold").toPandas()
spark.stop()

pdf = pdf.sort_values(["dataset_id", "engine_num", "cycle"]).reset_index(drop=True)
sensor_cols = [c for c in pdf.columns if c.startswith("sensor_")]

jump_rows = []
for (ds, eng), g in pdf.groupby(["dataset_id", "engine_id"]):
    g = g.sort_values("cycle").reset_index(drop=True)
    vals = g[sensor_cols].to_numpy()
    if len(vals) < 2:
        continue
    diffs = np.abs(vals[1:] - vals[:-1]).mean(axis=1)
    for d in diffs:
        jump_rows.append({"dataset_id": ds, "mean_abs_diff": float(d)})

jump_df = pd.DataFrame(jump_rows)
print("\n=== Magnitud promedio del salto entre ciclos consecutivos (normalización nueva: regresión continua) ===")
print(jump_df.groupby("dataset_id")["mean_abs_diff"].mean())
print("\n=== Comparación de referencia (normalización vieja: KMeans + z-score por cluster) ===")
print("FD001  no-switch=0.646146  (sin clusters, sin cambio aplicable)")
print("FD002  no-switch=0.710397  switch=0.820445")
print("FD003  no-switch=0.663292  (sin clusters, sin cambio aplicable)")
print("FD004  no-switch=0.718120  switch=0.838688")

print("\n✅ diagnóstico v2 terminado")
