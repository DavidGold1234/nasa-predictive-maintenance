from __future__ import annotations

import os
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, from_json
from pyspark.sql.streaming import DataStreamWriter
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType


KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.environ.get("ROS_KAFKA_TOPIC", "ros_motor_telemetry")
STARTING_OFFSETS = os.environ.get("KAFKA_STARTING_OFFSETS", "latest")
FAIL_ON_DATA_LOSS = os.environ.get("KAFKA_FAIL_ON_DATA_LOSS", "false")
TRIGGER_SECONDS = os.environ.get("ROS_BRONZE_TRIGGER_SECONDS", "10")
RETRY_WAIT_SECONDS = int(os.environ.get("ROS_BRONZE_RETRY_WAIT_SECONDS", "5"))

CHECKPOINT = os.environ.get(
    "HDFS_ROS_BRONZE_CHECKPOINT",
    "hdfs://namenode:9000/user/root/nasa/checkpoints/ros_bronze",
)
OUTPUT = os.environ.get(
    "HDFS_ROS_BRONZE",
    "hdfs://namenode:9000/user/root/nasa/bronze/ros_motor_telemetry",
)


ROS_SCHEMA = StructType(
    [
        StructField("timestamp", StringType(), True),
        StructField("dataset_id", StringType(), True),
        StructField("engine_id", StringType(), True),
        StructField("engine_num", IntegerType(), True),
        StructField("cycle", IntegerType(), True),
        StructField("op_setting_1", DoubleType(), True),
        StructField("op_setting_2", DoubleType(), True),
        StructField("op_setting_3", DoubleType(), True),
        StructField("sensor_2", DoubleType(), True),
        StructField("sensor_3", DoubleType(), True),
        StructField("sensor_4", DoubleType(), True),
        StructField("sensor_7", DoubleType(), True),
        StructField("sensor_8", DoubleType(), True),
        StructField("sensor_9", DoubleType(), True),
        StructField("sensor_11", DoubleType(), True),
        StructField("sensor_14", DoubleType(), True),
        StructField("sensor_15", DoubleType(), True),
        StructField("sensor_17", DoubleType(), True),
        StructField("sensor_21", DoubleType(), True),
    ]
)


def _build_writer(df) -> DataStreamWriter:
    return (
        df.writeStream.format("parquet")
        .outputMode("append")
        .option("path", OUTPUT)
        .option("checkpointLocation", CHECKPOINT)
        .trigger(processingTime=f"{TRIGGER_SECONDS} seconds")
    )


def _is_missing_topic_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "UnknownTopicOrPartitionException" in msg
        or "This server does not host this topic-partition" in msg
    )


def main() -> None:
    spark = (
        SparkSession.builder.appName("IngestROSKafkaToBronze")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print("✅ Ingesta ROS Kafka -> Bronze preparada")
    print(f"topic         : {KAFKA_TOPIC}")
    print(f"bootstrap     : {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"output bronze : {OUTPUT}")
    print(f"checkpoint    : {CHECKPOINT}")
    print(f"offsets       : {STARTING_OFFSETS}")
    print(f"failOnDataLoss: {FAIL_ON_DATA_LOSS}")

    waiting_printed = False

    while True:
        query = None
        try:
            kafka_df = (
                spark.readStream.format("kafka")
                .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
                .option("subscribe", KAFKA_TOPIC)
                .option("startingOffsets", STARTING_OFFSETS)
                .option("failOnDataLoss", FAIL_ON_DATA_LOSS)
                .load()
            )

            parsed = (
                kafka_df.selectExpr("CAST(value AS STRING) AS raw_json", "timestamp AS kafka_ts")
                .withColumn("payload", from_json(col("raw_json"), ROS_SCHEMA))
                .select("payload.*", "raw_json", "kafka_ts")
                .withColumn("ingest_ts", current_timestamp())
            )

            required = [
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
            ]

            bronze = parsed
            for name in required:
                bronze = bronze.filter(col(name).isNotNull())

            query = _build_writer(bronze).start()
            print("🚀 Ingesta ROS Kafka -> Bronze iniciada")
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
                    print(f"⏳ Esperando topic '{KAFKA_TOPIC}'... listener activo")
                    waiting_printed = True
                time.sleep(RETRY_WAIT_SECONDS)
                continue

            raise


if __name__ == "__main__":
    main()