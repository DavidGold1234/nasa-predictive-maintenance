#!/usr/bin/env bash
set -euo pipefail

SPARK_MASTER_CONTAINER="${SPARK_MASTER_CONTAINER:-spark-master}"

MSYS_NO_PATHCONV=1 docker exec -i "$SPARK_MASTER_CONTAINER" bash -lc "
  export ARTIFACTS_DIR='${ARTIFACTS_DIR:-/data/models/artifacts}'
  export VERSIONS_DIR='${VERSIONS_DIR:-/data/models/versions}'
  export HDFS_ARTIFACTS_ACTIVE='${HDFS_ARTIFACTS_ACTIVE:-hdfs://namenode:9000/user/root/nasa/artifacts/active}'
  export HDFS_ARTIFACTS_VERSIONS='${HDFS_ARTIFACTS_VERSIONS:-hdfs://namenode:9000/user/root/nasa/artifacts/versions}'
  python3 /src/models/retrain_ros_models.py
"

echo "✅ ROS retrain listo"
