#!/usr/bin/env bash
# Re-lanza los 6 motores ORIGINALES de ROS001 (mismos engine_id, mismo dataset_id)
# simulando que fueron reparados: el cycle publicado CONTINUA donde se quedo cada
# motor (no reinicia en 1, no choca con el historico en gold), pero la curva de
# degradacion SI se reinicia (local_cycle interno arranca en 1), asi el score
# vuelve a HEALTHY y sube de nuevo hasta un segundo pico CRITICAL - el patron
# "diente de sierra" de mantenimiento predictivo real.
#
# Uso: ./run_ros001_mantenimiento.sh [duracion_segundos]
set -euo pipefail

DURATION_SECONDS="${1:-900}"
ROS_CONTAINER="${ROS_CONTAINER:-ros-gazebo}"
SPARK_MASTER_CONTAINER="${SPARK_MASTER_CONTAINER:-spark-master}"
SPARK_MASTER_URL="${SPARK_MASTER_URL:-spark://spark-master:7077}"
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-postgres}"
POSTGRES_PACKAGE="${POSTGRES_PACKAGE:-org.postgresql:postgresql:42.7.4}"
KAFKA_PACKAGE="${KAFKA_PACKAGE:-org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0}"

DATASET_ID="ROS001"

echo "======================================================================"
echo " MANTENIMIENTO ROS001 — reparando y relanzando motores _1.._6"
echo " Duracion de la corrida: ${DURATION_SECONDS}s"
echo "======================================================================"
echo ""

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
echo "roscore OK."

BASELINE=$(docker exec kafka bash -c "kafka-run-class kafka.tools.GetOffsetShell --broker-list localhost:9092 --topic ros_motor_telemetry --time -1" 2>/dev/null | cut -d: -f3)
echo "Offset Kafka antes: $BASELINE"
echo ""

launch_engine() {
  local num="$1" seed="$2" profile="$3" start_cycle="$4"
  echo "  -> Motor ROS001_$num (perfil=$profile, start_cycle=$start_cycle)"
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
    export START_CYCLE=$start_cycle
    python3 src/ros_kafka_publisher.py > /tmp/mant_publisher${num}.log 2>&1
  "
}

echo "Consultando ultimo cycle registrado por motor..."
for num in 1 2 3 4 5 6; do
  LAST_CYCLE=$(docker exec "$POSTGRES_CONTAINER" psql -U postgres -d tesis -t -A -c \
    "SELECT COALESCE(MAX(cycle), 0) FROM eval_ros_predictions WHERE dataset_id='ROS001' AND engine_id='ROS001_${num}'" \
    2>/dev/null | tr -d '[:space:]')
  [ -z "$LAST_CYCLE" ] && LAST_CYCLE=0
  eval "START_${num}=$((LAST_CYCLE + 10))"
done

# Registrar el punto de reinicio de cada motor, para que el dashboard pueda calcular
# metricas "desde el ultimo mantenimiento" en vez de solo el primer critical de siempre.
docker exec "$POSTGRES_CONTAINER" psql -U postgres -d tesis -c "
  CREATE TABLE IF NOT EXISTS ros_maintenance_events (
      dataset_id   TEXT NOT NULL,
      engine_id    TEXT NOT NULL,
      start_cycle  BIGINT NOT NULL,
      created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY (engine_id, start_cycle)
  );
  INSERT INTO ros_maintenance_events (dataset_id, engine_id, start_cycle) VALUES
    ('$DATASET_ID', '${DATASET_ID}_1', $START_1),
    ('$DATASET_ID', '${DATASET_ID}_2', $START_2),
    ('$DATASET_ID', '${DATASET_ID}_3', $START_3),
    ('$DATASET_ID', '${DATASET_ID}_4', $START_4),
    ('$DATASET_ID', '${DATASET_ID}_5', $START_5),
    ('$DATASET_ID', '${DATASET_ID}_6', $START_6)
  ON CONFLICT DO NOTHING;
" >/dev/null

echo "Lanzando motores reparados..."
launch_engine 1 11 mild      "$START_1"
launch_engine 2 12 mild      "$START_2"
launch_engine 3 13 medium    "$START_3"
launch_engine 4 14 medium    "$START_4"
launch_engine 5 15 aggressive "$START_5"
launch_engine 6 16 aggressive "$START_6"
sleep 3
echo ""
echo "Motores vivos:"
docker exec "$ROS_CONTAINER" bash -c "pgrep -af ros_kafka_publisher.py | grep -v 'bash -lc'" || true
echo ""

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

echo ""
echo "Deteniendo motores..."
docker exec "$ROS_CONTAINER" bash -c "pkill -f ros_kafka_publisher.py" || true

FINAL_OFFSET=$(docker exec kafka bash -c "kafka-run-class kafka.tools.GetOffsetShell --broker-list localhost:9092 --topic ros_motor_telemetry --time -1" 2>/dev/null | cut -d: -f3)
echo "Offset final: $FINAL_OFFSET (+$((FINAL_OFFSET - BASELINE)) mensajes nuevos)"
echo ""

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
echo " LISTO. En Grafana, dashboard ROS, selecciona la corrida (run_id) mas"
echo " reciente para ver el patron completo: degradacion -> CRITICAL ->"
echo " mantenimiento -> HEALTHY -> degradacion de nuevo, en ROS001_1.._6."
echo "======================================================================"
