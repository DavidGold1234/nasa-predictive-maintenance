from __future__ import annotations

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, from_json
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.environ.get("ROS_KAFKA_TOPIC", "ros_motor_telemetry")
OUTPUT = os.environ.get(
    "HDFS_ROS_BRONZE_BATCH",
    "hdfs://namenode:9000/user/root/nasa/bronze/ros_motor_telemetry_batch",
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

spark = (
    SparkSession.builder.appName("BatchKafkaToBronze")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

kafka_df = (
    spark.read.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "earliest")
    .option("endingOffsets", "latest")
    .load()
)

parsed = (
    kafka_df.selectExpr("CAST(value AS STRING) AS raw_json")
    .withColumn("payload", from_json(col("raw_json"), ROS_SCHEMA))
    .select("payload.*")
    .withColumn("ingest_ts", current_timestamp())
)

required = [
    "dataset_id", "engine_id", "engine_num", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_2", "sensor_3", "sensor_4",
    "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_14", "sensor_15",
    "sensor_17", "sensor_21",
]

bronze = parsed
for name in required:
    bronze = bronze.filter(col(name).isNotNull())

n = bronze.count()
print(f"Filas leídas de Kafka (batch, earliest->latest): {n}")

bronze.write.mode("overwrite").parquet(OUTPUT)
print(f"✅ Escrito en: {OUTPUT}")

spark.stop()
