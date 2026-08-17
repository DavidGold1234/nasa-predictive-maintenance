from __future__ import annotations

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, row_number, to_timestamp
from pyspark.sql.window import Window


BRONZE = os.environ.get(
    "HDFS_ROS_BRONZE",
    "hdfs://namenode:9000/user/root/nasa/bronze/ros_motor_telemetry",
)
SILVER = os.environ.get(
    "HDFS_ROS_SILVER",
    "hdfs://namenode:9000/user/root/nasa/silver/ros_clean",
)


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("ProcessROSBronzeToSilver")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(BRONZE)
    bronze_count = df.count()

    ts_col = "timestamp" if "timestamp" in df.columns else None
    ingest_col = "ingest_ts" if "ingest_ts" in df.columns else None

    if ts_col:
        df = df.withColumn("event_ts", to_timestamp(col(ts_col)))
    else:
        df = df.withColumn("event_ts", current_timestamp())

    if ingest_col is None:
        df = df.withColumn("ingest_ts", current_timestamp())

    required_base = ["dataset_id", "engine_id", "engine_num", "cycle"]
    numeric_cols = [
        "op_setting_1", "op_setting_2", "op_setting_3",
        "sensor_2", "sensor_3", "sensor_4",
        "sensor_7", "sensor_8", "sensor_9",
        "sensor_11", "sensor_14", "sensor_15",
        "sensor_17", "sensor_21",
    ]

    missing = [c for c in required_base + numeric_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en bronze ROS: {missing}")

    clean = (
        df.filter(col("dataset_id").isNotNull())
        .filter(col("engine_id").isNotNull())
        .filter(col("engine_num").isNotNull())
        .filter(col("cycle").isNotNull())
        .filter(col("cycle") > 0)
    )
    for c in numeric_cols:
        clean = clean.filter(col(c).isNotNull())

    clean_count = clean.count()

    # Mantener el raw de silver. Solo quitar duplicados exactos, no colapsar por cycle.
    dedup_cols = [
        "dataset_id", "engine_id", "engine_num", "cycle", "timestamp",
        "op_setting_1", "op_setting_2", "op_setting_3",
        "sensor_2", "sensor_3", "sensor_4",
        "sensor_7", "sensor_8", "sensor_9",
        "sensor_11", "sensor_14", "sensor_15",
        "sensor_17", "sensor_21",
    ]
    dedup_cols = [c for c in dedup_cols if c in clean.columns]

    w = Window.partitionBy(*dedup_cols).orderBy(col("ingest_ts").desc())
    silver_df = (
        clean.withColumn("rn", row_number().over(w))
        .filter(col("rn") == 1)
        .drop("rn")
        .select(
            "timestamp",
            "event_ts",
            "ingest_ts",
            "dataset_id",
            "engine_id",
            "engine_num",
            "cycle",
            "op_setting_1",
            "op_setting_2",
            "op_setting_3",
            "sensor_2",
            "sensor_3",
            "sensor_4",
            "sensor_7",
            "sensor_8",
            "sensor_9",
            "sensor_11",
            "sensor_14",
            "sensor_15",
            "sensor_17",
            "sensor_21",
        )
    )

    silver_count = silver_df.count()
    silver_df.write.mode("overwrite").parquet(SILVER)

    print(f"✅ Silver ROS generado en: {SILVER}")
    print(f"Bronze original    : {bronze_count}")
    print(f"Tras limpieza      : {clean_count}")
    print(f"Registros silver   : {silver_count}")

    spark.stop()


if __name__ == "__main__":
    main()
