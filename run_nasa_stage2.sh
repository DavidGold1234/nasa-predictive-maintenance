#!/usr/bin/env bash
set -euo pipefail

SPARK_MASTER_CONTAINER="${SPARK_MASTER_CONTAINER:-spark-master}"
SPARK_MASTER_URL="${SPARK_MASTER_URL:-spark://spark-master:7077}"
POSTGRES_PACKAGE="${POSTGRES_PACKAGE:-org.postgresql:postgresql:42.7.4}"

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

run_spark_py() {
  local script_path="$1"
  shift || true
  MSYS_NO_PATHCONV=1 docker exec -i "$SPARK_MASTER_CONTAINER" bash -lc "
    mkdir -p /tmp/.ivy2/cache /tmp/.ivy2/jars
    export HOME=/tmp
    export IVY_HOME=/tmp/.ivy2
    export ARTIFACTS_DIR='${ARTIFACTS_DIR:-/data/models/artifacts}'
    export HDFS_ARTIFACTS_ACTIVE='${HDFS_ARTIFACTS_ACTIVE:-hdfs://namenode:9000/user/root/nasa/artifacts/active}'
    export HDFS_ARTIFACTS_VERSIONS='${HDFS_ARTIFACTS_VERSIONS:-hdfs://namenode:9000/user/root/nasa/artifacts/versions}'
    /opt/spark/bin/spark-submit \
      --master $SPARK_MASTER_URL \
      --conf spark.jars.ivy=/tmp/.ivy2 \
      --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
      --conf spark.pyspark.python=python3 \
      --packages $POSTGRES_PACKAGE \
      $script_path
  "
}

case "${1:-all}" in
  train)
    log "📊 train + evaluación interna -> PostgreSQL"
    run_spark_py /src/models/train_and_evaluate.py
    ;;
  external)
    log "🌍 evaluación externa desde silver limpio -> PostgreSQL"
    run_spark_py /src/models/external_evaluation.py
    ;;
  all)
    log "📊 train + evaluación interna -> PostgreSQL"
    run_spark_py /src/models/train_and_evaluate.py
    log "🌍 evaluación externa desde silver limpio -> PostgreSQL"
    run_spark_py /src/models/external_evaluation.py
    ;;
  *)
    cat <<USAGE
Uso: $0 {train|external|all}

train     train + evaluación interna -> PostgreSQL
external  evaluación externa desde silver limpio -> PostgreSQL
all       train + interna + externa
USAGE
    exit 1
    ;;
esac

log "✅ NASA stage 2 listo"