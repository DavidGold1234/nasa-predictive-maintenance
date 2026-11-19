from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, from_json, lit, regexp_extract, regexp_replace
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC = os.environ.get("TOPIC_ROS_ALERTS", "ros_motor_alerts")
CHECKPOINT = os.environ.get(
    "ROS_ALERTS_CHECKPOINT",
    "hdfs://namenode:9000/user/root/nasa/checkpoints/ros_alerts_to_postgres",
)

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "tesis")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
POSTGRES_SCHEMA = os.environ.get("POSTGRES_SCHEMA", "public")
# Misma tabla que usa el pipeline batch (evaluate_ros_from_gold.py), para que las
# alertas en vivo aparezcan en el mismo dashboard de Grafana sin panel aparte.
POSTGRES_TABLE = os.environ.get("POSTGRES_TABLE", "eval_ros_predictions")

RETRY_WAIT_SECONDS = int(os.environ.get("ROS_ALERTS_RETRY_WAIT_SECONDS", "5"))

# Un run_id por sesion de streaming, para poder distinguirla en el desplegable
# "Corrida (run_id)" de Grafana igual que las corridas batch.
RUN_ID = os.environ.get(
    "ROS_LIVE_RUN_ID",
    "ros_live_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
)

# Schema compacto — debe coincidir exactamente con el payload de ros_kafka_inference.py
schema = StructType([
    StructField("timestamp",         StringType(),  True),
    StructField("engine_id",         StringType(),  True),
    StructField("cycle",             IntegerType(), True),
    StructField("if_score",          DoubleType(),  True),
    StructField("lstm_score",        DoubleType(),  True),
    StructField("tcn_score",         DoubleType(),  True),
    StructField("transformer_score", DoubleType(),  True),
    StructField("if_status",         StringType(),  True),
    StructField("lstm_status",       StringType(),  True),
    StructField("tcn_status",        StringType(),  True),
    StructField("transformer_status", StringType(), True),
    StructField("final_status",      StringType(),  True),
])


def write_batch_to_postgres(df: DataFrame, batch_id: int) -> None:
    if df.rdd.isEmpty():
        print(f"⚠️ batch_id={batch_id}: vacío")
        return

    jdbc_url = f"jdbc:postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    full_table = f"{POSTGRES_SCHEMA}.{POSTGRES_TABLE}" if POSTGRES_SCHEMA else POSTGRES_TABLE
    props = {
        "user": POSTGRES_USER,
        "password": POSTGRES_PASSWORD,
        "driver": "org.postgresql.Driver",
    }

    (
        df.write
        .mode("append")
        .jdbc(jdbc_url, full_table, properties=props)
    )
    print(f"✅ batch_id={batch_id} -> {full_table}")


def _is_missing_topic_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "UnknownTopicOrPartitionException" in msg
        or "This server does not host this topic-partition" in msg
    )


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("IngestROSAlertsToPostgres")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"✅ ROS alerts -> PostgreSQL preparado | topic={TOPIC} | tabla={POSTGRES_TABLE} | run_id={RUN_ID}")

    waiting_printed = False

    while True:
        query = None
        try:
            raw = (
                spark.readStream
                .format("kafka")
                .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
                .option("subscribe", TOPIC)
                .option("startingOffsets", "latest")
                .option("failOnDataLoss", "false")
                .load()
            )

            # engine_id sigue el patron {DATASET_ID}_{ENGINE_NUM} (ver ros_kafka_publisher.py);
            # se separan aqui para que la fila calce con el esquema de eval_ros_predictions,
            # la misma tabla que ya lee el dashboard de Grafana para el pipeline batch.
            parsed = (
                raw.selectExpr("CAST(value AS STRING) AS json_str")
                .select(from_json(col("json_str"), schema).alias("data"))
                .select("data.*")
                .withColumn("run_id", lit(RUN_ID))
                .withColumn("dataset_id", regexp_replace(col("engine_id"), "_[0-9]+$", ""))
                .withColumn("engine_num", regexp_extract(col("engine_id"), "_([0-9]+)$", 1).cast("int"))
                .drop("timestamp")
            )

            query = (
                parsed.writeStream
                .outputMode("append")
                .option("checkpointLocation", CHECKPOINT)
                .foreachBatch(write_batch_to_postgres)
                .start()
            )

            print("🚀 Ingest ROS alerts -> PostgreSQL iniciado")
            query.awaitTermination()
            break

        except Exception as exc:
            if query is not None:
                try:
                    query.stop()
                except Exception:
                    pass

            if _is_missing_topic_error(exc):
                if not waiting_printed:
                    print(f"⏳ Esperando topic '{TOPIC}'... listener activo")
                    waiting_printed = True
                time.sleep(RETRY_WAIT_SECONDS)
                continue

            raise


if __name__ == "__main__":
    main()