#!/usr/bin/env bash
set -euo pipefail

SPARK_MASTER_CONTAINER="${SPARK_MASTER_CONTAINER:-spark-master}"
SPARK_MASTER_URL="${SPARK_MASTER_URL:-spark://spark-master:7077}"
POSTGRES_PACKAGE="${POSTGRES_PACKAGE:-org.postgresql:postgresql:42.7.4}"

run_spark_py() {
  local script_path="$1"
  shift || true
  MSYS_NO_PATHCONV=1 docker exec -i "$SPARK_MASTER_CONTAINER" bash -lc "
    mkdir -p /tmp/.ivy2/cache /tmp/.ivy2/jars
    export HOME=/tmp
    export IVY_HOME=/tmp/.ivy2
    /opt/spark/bin/spark-submit \
      --master $SPARK_MASTER_URL \
      --conf spark.jars.ivy=/tmp/.ivy2 \
      --packages $POSTGRES_PACKAGE \
      $script_path
  "
}

case "${1:-all}" in
  silver)
    run_spark_py /apps/processing/process_ros_bronze_to_silver.py
    ;;
  gold)
    run_spark_py /src/features/build_ros_gold.py
    ;;
  evaluate)
    run_spark_py /src/models/evaluate_ros_from_gold.py
    ;;
  all)
    run_spark_py /apps/processing/process_ros_bronze_to_silver.py
    run_spark_py /src/features/build_ros_gold.py
    run_spark_py /src/models/evaluate_ros_from_gold.py
    ;;
  *)
    cat <<USAGE
Uso: $0 {silver|gold|evaluate|all}

silver     bronze ROS -> silver limpio
gold       silver ROS -> ros_gold listo para modelado
evaluate   ros_gold -> PostgreSQL
all        silver + gold + evaluate
USAGE
    exit 1
    ;;
esac
