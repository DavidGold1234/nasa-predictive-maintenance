#!/usr/bin/env bash
# Demo de un solo comando para la sustentacion: lanza 3 motores ROS nuevos
# (mild/medium/aggressive) bajo un DATASET_ID unico y visible (DEMO_<fecha_hora>),
# los deja correr un tiempo configurable, y luego corre el pipeline completo
# (bronze -> silver -> gold -> evaluate) para que el nuevo run_id aparezca
# automaticamente en el desplegable de Grafana.
#
# Uso:
#   ./run_demo_defensa.sh                # 3 motores, 5 minutos, perfiles mild/medium/aggressive
#   ./run_demo_defensa.sh 600             # 3 motores, 10 minutos
#
# Requiere: Docker levantado (docker compose up -d), Kafka sano.

set -euo pipefail

DURATION_SECONDS="${1:-300}"   # 5 min por defecto
ROS_CONTAINER="${ROS_CONTAINER:-ros-gazebo}"
SPARK_MASTER_CONTAINER="${SPARK_MASTER_CONTAINER:-spark-master}"
SPARK_MASTER_URL="${SPARK_MASTER_URL:-spark://spark-master:7077}"
POSTGRES_PACKAGE="${POSTGRES_PACKAGE:-org.postgresql:postgresql:42.7.4}"
KAFKA_PACKAGE="${KAFKA_PACKAGE:-org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0}"

DATASET_ID="DEMO_$(date +%Y%m%d_%H%M%S)"

echo "======================================================================"
echo " DEMO DEFENSA — dataset_id = $DATASET_ID"
echo " Duracion de la corrida: ${DURATION_SECONDS}s"
echo "======================================================================"
echo ""

# 1) roscore idempotente (mismo patron que run_ros_publishers.sh)
if ! docker exec "$ROS_CONTAINER" bash -c "pgrep -f 'bin/roscore' >/dev/null 2>&1"; then
    echo "roscore no esta corriendo — arrancandolo..."
    MSYS_NO_PATHCONV=1 docker exec -d "$ROS_CONTAINER" bash -lc "
        source /opt/ros/noetic/setup.bash
        export ROS_MASTER_URI=http://localhost:11311
        export ROS_HOSTNAME=localhost
        roscore
    "
    sleep 3
fi
echo "roscore OK."

# 2) baseline de offset Kafka
BASELINE=$(docker exec kafka bash -c "kafka-run-class kafka.tools.GetOffsetShell --broker-list localhost:9092 --topic ros_motor_telemetry --time -1" 2>/dev/null | cut -d: -f3)
echo "Offset Kafka antes de la demo: $BASELINE"
echo ""

# 3) lanzar 3 motores: mild, medium, aggressive
launch_engine() {
  local num="$1" seed="$2" profile="$3"
  echo "  -> Motor $num (perfil=$profile)"
  MSYS_NO_PATHCONV=1 docker exec -d "$ROS_CONTAINER" bash -lc "
    cd /root/ros_ws
    source /opt/ros/noetic/setup.bash
    export ROS_MASTER_URI=http://localhost:11311
    export ROS_HOSTNAME=localhost
    export KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    export DATASET_ID=$DATASET_ID
    export ENGINE_NUM=$num
    export RUN_SEED=$seed
    export DEGRADATION_PROFILE=$profile
    export PUBLISH_HZ=10
    export INFERENCE_HZ=1
    python3 src/ros_kafka_publisher.py > /tmp/demo_publisher${num}.log 2>&1
  "
}

echo "Lanzando 3 motores nuevos bajo dataset_id=$DATASET_ID ..."
launch_engine 1 1 mild
launch_engine 2 2 medium
launch_engine 3 3 aggressive
sleep 3
echo ""
echo "Motores vivos:"
docker exec "$ROS_CONTAINER" bash -c "pgrep -af ros_kafka_publisher.py | grep -v 'bash -lc'" || true
echo ""

# 4) dejarlos correr, con progreso cada 60s
echo "Corriendo durante ${DURATION_SECONDS}s (Ctrl+C para cortar antes)..."
ELAPSED=0
while [ "$ELAPSED" -lt "$DURATION_SECONDS" ]; do
    STEP=60
    REMAIN=$((DURATION_SECONDS - ELAPSED))
    [ "$STEP" -gt "$REMAIN" ] && STEP="$REMAIN"
    sleep "$STEP"
    ELAPSED=$((ELAPSED + STEP))
    OFFSET=$(docker exec kafka bash -c "kafka-run-class kafka.tools.GetOffsetShell --broker-list localhost:9092 --topic ros_motor_telemetry --time -1" 2>/dev/null | cut -d: -f3)
    echo "  [$(date +%H:%M:%S)] elapsed=${ELAPSED}s offset=$OFFSET (+$((OFFSET - BASELINE)) desde el inicio)"
done

# 5) detener SOLO los publishers (asume que no hay otros corriendo en paralelo)
echo ""
echo "Deteniendo motores..."
docker exec "$ROS_CONTAINER" bash -c "pkill -f ros_kafka_publisher.py" || true

FINAL_OFFSET=$(docker exec kafka bash -c "kafka-run-class kafka.tools.GetOffsetShell --broker-list localhost:9092 --topic ros_motor_telemetry --time -1" 2>/dev/null | cut -d: -f3)
echo "Offset final: $FINAL_OFFSET (+$((FINAL_OFFSET - BASELINE)) mensajes nuevos)"
echo ""

# 6) pipeline completo: bronze -> silver -> gold -> evaluate
echo "======================================================================"
echo " Corriendo pipeline: bronze -> silver -> gold -> evaluate"
echo "======================================================================"

echo ""
echo "--- bronze ---"
MSYS_NO_PATHCONV=1 docker exec -i "$SPARK_MASTER_CONTAINER" bash -lc "
  mkdir -p /tmp/.ivy2/cache /tmp/.ivy2/jars
  export HOME=/tmp
  export IVY_HOME=/tmp/.ivy2
  /opt/spark/bin/spark-submit --master $SPARK_MASTER_URL --conf spark.jars.ivy=/tmp/.ivy2 --packages $POSTGRES_PACKAGE,$KAFKA_PACKAGE /outputs/batch_kafka_to_bronze.py
" 2>&1 | tail -n 15

echo ""
echo "--- silver ---"
MSYS_NO_PATHCONV=1 docker exec -i "$SPARK_MASTER_CONTAINER" bash -lc "
  export HOME=/tmp
  export IVY_HOME=/tmp/.ivy2
  export HDFS_ROS_BRONZE=hdfs://namenode:9000/user/root/nasa/bronze/ros_motor_telemetry_batch
  /opt/spark/bin/spark-submit --master $SPARK_MASTER_URL --conf spark.jars.ivy=/tmp/.ivy2 --packages $POSTGRES_PACKAGE /apps/processing/process_ros_bronze_to_silver.py
" 2>&1 | tail -n 10

echo ""
echo "--- gold ---"
MSYS_NO_PATHCONV=1 docker exec -i "$SPARK_MASTER_CONTAINER" bash -lc "
  export HOME=/tmp
  export IVY_HOME=/tmp/.ivy2
  /opt/spark/bin/spark-submit --master $SPARK_MASTER_URL --conf spark.jars.ivy=/tmp/.ivy2 --packages $POSTGRES_PACKAGE /src/features/build_ros_gold.py
" 2>&1 | tail -n 10

echo ""
echo "--- evaluate ---"
MSYS_NO_PATHCONV=1 docker exec -i "$SPARK_MASTER_CONTAINER" bash -lc "
  export HOME=/tmp
  export IVY_HOME=/tmp/.ivy2
  /opt/spark/bin/spark-submit --master $SPARK_MASTER_URL --conf spark.jars.ivy=/tmp/.ivy2 --packages $POSTGRES_PACKAGE /src/models/evaluate_ros_from_gold.py
" 2>&1 | tail -n 15

echo ""
echo "======================================================================"
echo " LISTO. Abre Grafana (localhost:3000), dashboard ROS, y selecciona"
echo " en el desplegable 'Corrida (run_id)' la mas reciente (arriba de todo,"
echo " ordenada por fecha descendente) para ver los resultados de HOY."
echo " Motores de esta demo: ${DATASET_ID}_1 (mild), ${DATASET_ID}_2 (medium), ${DATASET_ID}_3 (aggressive)"
echo "======================================================================"
