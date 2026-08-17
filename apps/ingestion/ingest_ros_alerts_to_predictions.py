from __future__ import annotations

import os
import time

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, current_timestamp, from_json
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
POSTGRES_TABLE = os.environ.get("POSTGRES_TABLE", "ros_alerts_live")

RETRY_WAIT_SECONDS = int(os.environ.get("ROS_ALERTS_RETRY_WAIT_SECONDS", "5"))

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

    print(f"✅ ROS alerts -> PostgreSQL preparado | topic={TOPIC} | checkpoint={CHECKPOINT}")

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

            parsed = (
                raw.selectExpr("CAST(value AS STRING) AS json_str")
                .select(from_json(col("json_str"), schema).alias("data"))
                .select("data.*")
                .withColumn("ingested_at", current_timestamp())
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