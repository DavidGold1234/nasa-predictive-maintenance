from __future__ import annotations

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

sensor_cols = [
    "sensor_9", "sensor_4", "sensor_3", "sensor_17", "sensor_11",
    "sensor_14", "sensor_8", "sensor_7", "sensor_21", "sensor_2", "sensor_15",
]
op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]

rng = np.random.default_rng(7)
rows = []
n_engines = 4
n_cycles = 220

for eng in range(1, n_engines + 1):
    engine_id = f"ROS_{eng}"
    base = rng.normal(loc=50, scale=5, size=len(sensor_cols))
    op_regime_idx = 0
    for cycle in range(1, n_cycles + 1):
        if rng.random() < 0.15:
            op_regime_idx = rng.integers(0, 4)
        op_vals = [
            float(op_regime_idx * 10 + rng.normal(0, 0.5)),
            float(op_regime_idx * 2 + rng.normal(0, 0.2)),
            float(op_regime_idx + rng.normal(0, 0.1)),
        ]
        degradation = (cycle / n_cycles) * 5.0
        sensor_vals = base + degradation + rng.normal(0, 1.0, size=len(sensor_cols))
        row = {
            "dataset_id": "ros",
            "engine_id": engine_id,
            "engine_num": eng,
            "cycle": cycle,
            "timestamp": f"2026-08-16T00:00:{cycle:02d}Z",
            "event_ts": f"2026-08-16T00:00:{cycle:02d}Z",
            "ingest_ts": f"2026-08-16T00:00:{cycle:02d}Z",
        }
        row.update({c: v for c, v in zip(op_cols, op_vals)})
        row.update({c: float(v) for c, v in zip(sensor_cols, sensor_vals)})
        rows.append(row)

pdf = pd.DataFrame(rows)
print(f"Filas sintéticas generadas: {len(pdf)} | motores={pdf['engine_id'].nunique()}")

spark = (
    SparkSession.builder.appName("GenerateSyntheticROSSilver")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
spark.createDataFrame(pdf).write.mode("overwrite").parquet(
    "hdfs://namenode:9000/user/root/nasa/silver/ros_clean"
)
print("✅ ros_clean sintético escrito en HDFS")
spark.stop()
