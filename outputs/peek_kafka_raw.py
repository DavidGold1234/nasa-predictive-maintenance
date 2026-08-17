from __future__ import annotations

import os
from pyspark.sql import SparkSession

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.environ.get("ROS_KAFKA_TOPIC", "ros_motor_telemetry")

spark = (
    SparkSession.builder.appName("PeekKafkaRaw")
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

print("total rows:", kafka_df.count())
raw = kafka_df.selectExpr("CAST(value AS STRING) AS raw_json").limit(5).collect()
for r in raw:
    print("RAW:", r["raw_json"])

spark.stop()
