#!/usr/bin/env bash
set -euo pipefail

SPARK_MASTER_CONTAINER="${SPARK_MASTER_CONTAINER:-spark-master}"
SPARK_MASTER_URL="${SPARK_MASTER_URL:-spark://spark-master:7077}"
POSTGRES_PACKAGE="${POSTGRES_PACKAGE:-org.postgresql:postgresql:42.7.4}"
KAFKA_PACKAGE="${KAFKA_PACKAGE:-org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0}"

run_spark_stream() {
  local script_path="$1"
  local total_cores="${2:-2}"
  local executor_cores="${3:-1}"
  local executor_memory="${4:-1g}"

  MSYS_NO_PATHCONV=1 docker exec -it "$SPARK_MASTER_CONTAINER" bash -lc "
    mkdir -p /tmp/.ivy2/cache /tmp/.ivy2/jars
    export HOME=/tmp
    export IVY_HOME=/tmp/.ivy2
    /opt/spark/bin/spark-submit \
      --master $SPARK_MASTER_URL \
      --total-executor-cores $total_cores \
      --executor-cores $executor_cores \
      --executor-memory $executor_memory \
      --conf spark.jars.ivy=/tmp/.ivy2 \
      --packages $POSTGRES_PACKAGE,$KAFKA_PACKAGE \
      $script_path
  "
}

case "${1:-}" in
  bronze)
    run_spark_stream /apps/ingestion/ingest_ros_kafka_to_bronze.py 2 1 512m
    ;;

  inference)
  MSYS_NO_PATHCONV=1 docker exec -it "$SPARK_MASTER_CONTAINER" bash -lc "
    export ARTIFACTS_DIR='${ARTIFACTS_DIR:-/data/models/artifacts}'
    python3 /src/models/ros_kafka_inference.py
  "
  ;;

  alerts)
    run_spark_stream /apps/ingestion/ingest_ros_alerts_to_predictions.py 2 1 512m
    ;;
  *)
    cat <<USAGE
Uso: $0 {bronze|inference|alerts}

bronze     Kafka ros_motor_telemetry -> HDFS bronze
inference  Kafka ros_motor_telemetry -> ros_motor_alerts usando artifacts locales
alerts     Kafka ros_motor_alerts -> PostgreSQL
USAGE
    exit 1
    ;;
esac