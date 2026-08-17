#!/usr/bin/env bash
set -euo pipefail

SPARK_MASTER_CONTAINER="${SPARK_MASTER_CONTAINER:-spark-master}"
SPARK_MASTER_URL="${SPARK_MASTER_URL:-spark://spark-master:7077}"

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log "🚀 NASA stage 1: ingest -> silver -> train_gold"

python apps/ingestion/ingest_nasa_to_hdfs.py

MSYS_NO_PATHCONV=1 docker exec -i "$SPARK_MASTER_CONTAINER" bash -lc "
/opt/spark/bin/spark-submit \
  --master $SPARK_MASTER_URL \
  /apps/processing/process_nasa_spark.py
"

MSYS_NO_PATHCONV=1 docker exec -i "$SPARK_MASTER_CONTAINER" bash -lc "
export FEATURES_JSON='${FEATURES_JSON:-/src/features/selected_features.json}'
export ARTIFACTS_DIR='${ARTIFACTS_DIR:-/data/models/artifacts}'
export HDFS_ARTIFACTS_ACTIVE='${HDFS_ARTIFACTS_ACTIVE:-hdfs://namenode:9000/user/root/nasa/artifacts/active}'
export HDFS_ARTIFACTS_VERSIONS='${HDFS_ARTIFACTS_VERSIONS:-hdfs://namenode:9000/user/root/nasa/artifacts/versions}'
/opt/spark/bin/spark-submit \
  --master $SPARK_MASTER_URL \
  /src/preprocessing/run_preprocessing.py
"

log "✅ NASA stage 1 listo"
echo "Gold: hdfs://namenode:9000/user/root/nasa/gold/nasa_train_gold"
echo "Artifacts activos: ${ARTIFACTS_DIR:-/data/models/artifacts}"
