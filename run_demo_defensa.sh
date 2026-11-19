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
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-postgres}"

# Numeracion de motores continua (no reiniciar en 1/2/3 cada demo): se consulta el
# engine_num mas alto ya usado en ROS001 + cualquier demo anterior, para que los
# motores nuevos se vean como continuacion de la misma flota (ej. _7, _8, _9) en vez
# de chocar visualmente con ROS001_1..6 (que llevan miles de ciclos y ya estan CRITICAL).
# dataset_id sigue siendo DEMO_<timestamp> (no ROS001) para no alterar el conteo
# historico oficial de ROS001 (1 030 277 mensajes) ya reportado en la tesis.
NEXT_ENGINE=$(docker exec "$POSTGRES_CONTAINER" psql -U postgres -d tesis -t -A -c \
  "SELECT COALESCE(MAX(engine_num), 6) FROM eval_ros_predictions WHERE dataset_id LIKE 'ROS001%' OR dataset_id LIKE 'DEMO_%'" \
  2>/dev/null | tr -d '[:space:]')
[ -z "$NEXT_ENGINE" ] && NEXT_ENGINE=6
ENGINE_1=$((NEXT_ENGINE + 1))
ENGINE_2=$((NEXT_ENGINE + 2))
ENGINE_3=$((NEXT_ENGINE + 3))

echo "======================================================================"
echo " DEMO DEFENSA — dataset_id = $DATASET_ID"
echo " Motores nuevos: _${ENGINE_1} (mild), _${ENGINE_2} (medium), _${ENGINE_3} (aggressive)"
echo " Duracion de la corrida: ${DURATION_SECONDS}s"
echo "======================================================================"
echo ""

# 1) roscore idempotente — se verifica con una prueba FUNCIONAL (rosnode list),
#    no solo con pgrep, porque pgrep puede dar falsos positivos con procesos
#    zombie o en pleno arranque.
roscore_vivo() {
    docker exec "$ROS_CONTAINER" bash -lc "
        source /opt/ros/noetic/setup.bash
        export ROS_MASTER_URI=http://localhost:11311
        timeout 5 rosnode list >/dev/null 2>&1
    "
}

if ! roscore_vivo; then
    echo "roscore no responde — arrancandolo..."
    docker exec "$ROS_CONTAINER" bash -c "pkill -9 -f 'bin/roscore'" >/dev/null 2>&1 || true
    sleep 1
    MSYS_NO_PATHCONV=1 docker exec -d "$ROS_CONTAINER" bash -lc "
        source /opt/ros/noetic/setup.bash
        export ROS_MASTER_URI=http://localhost:11311
        export ROS_HOSTNAME=localhost
        roscore
    "
    for i in 1 2 3 4 5 6; do
        sleep 3
        if roscore_vivo; then break; fi
    done
    if ! roscore_vivo; then
        echo "ERROR: roscore no respondio tras varios intentos. Abortando."
        exit 1
    fi
fi
echo "roscore OK (verificado con rosnode list)."

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
launch_engine "$ENGINE_1" 1 mild
launch_engine "$ENGINE_2" 2 medium
launch_engine "$ENGINE_3" 3 aggressive
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
echo " Motores de esta demo: ${DATASET_ID}_${ENGINE_1} (mild), ${DATASET_ID}_${ENGINE_2} (medium), ${DATASET_ID}_${ENGINE_3} (aggressive)"
echo "======================================================================"
