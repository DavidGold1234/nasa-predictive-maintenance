#!/usr/bin/env bash
# Lanza N instancias del ros_kafka_publisher en paralelo dentro del contenedor ros-gazebo.
# Cada instancia representa un motor independiente con su propio ENGINE_NUM y RUN_SEED.
# Todos comparten el mismo DATASET_ID y publican al mismo topic Kafka.
#
# Uso:
#   ./run_ros_publishers.sh                          # 3 motores, perfil medium, 10 Hz
#   ./run_ros_publishers.sh 5 aggressive 10          # 5 motores, perfil aggressive, 10 Hz
#   ./run_ros_publishers.sh 1 mild 1                 # 1 motor, perfil mild, 1 Hz (NASA-equivalente)

set -euo pipefail

N_ENGINES="${1:-3}"
DEGRADATION_PROFILE="${2:-medium}"
PUBLISH_HZ="${3:-10}"

ROS_CONTAINER="${ROS_CONTAINER:-ros-gazebo}"
DATASET_ID="${DATASET_ID:-ROS001}"
INFERENCE_HZ="${INFERENCE_HZ:-1}"
KAFKA_BOOTSTRAP="${KAFKA_BOOTSTRAP_SERVERS:-kafka:9092}"

# rospy.init_node() requiere un roscore/master ROS activo; el contenedor no arranca
# ninguno por defecto (Dockerfile solo hace CMD ["/bin/bash"]). Sin esto, rospy.Rate.sleep()
# no tiene una noción de tiempo válida y los publishers no respetan PUBLISH_HZ (ver
# docs/ExperimentosNASA_2026-08-16.md, Etapa 8). Se arranca de forma idempotente.
if ! docker exec "$ROS_CONTAINER" bash -c "pgrep -f 'bin/roscore' >/dev/null 2>&1"; then
    echo "🔧 roscore no está corriendo en '$ROS_CONTAINER' — arrancándolo..."
    MSYS_NO_PATHCONV=1 docker exec -d "$ROS_CONTAINER" bash -lc "
        source /opt/ros/noetic/setup.bash
        export ROS_MASTER_URI=http://localhost:11311
        export ROS_HOSTNAME=localhost
        roscore
    "
    sleep 3
    if ! docker exec "$ROS_CONTAINER" bash -c "pgrep -f 'bin/roscore' >/dev/null 2>&1"; then
        echo "❌ No se pudo arrancar roscore en '$ROS_CONTAINER'. Abortando." >&2
        exit 1
    fi
    echo "✅ roscore arrancado."
else
    echo "✅ roscore ya está corriendo en '$ROS_CONTAINER'."
fi

echo "🚀 Lanzando $N_ENGINES motores | profile=$DEGRADATION_PROFILE | publish_hz=$PUBLISH_HZ | inference_hz=$INFERENCE_HZ"
echo "   topic=ros_motor_telemetry | dataset=$DATASET_ID"
echo ""

PIDS=()

for i in $(seq 1 "$N_ENGINES"); do
    echo "  ▶ Motor ENGINE_NUM=$i (RUN_SEED=$i)"
    MSYS_NO_PATHCONV=1 docker exec -d "$ROS_CONTAINER" bash -lc "
        export PUBLISH_HZ=$PUBLISH_HZ
        export INFERENCE_HZ=$INFERENCE_HZ
        export DATASET_ID=$DATASET_ID
        export ENGINE_NUM=$i
        export RUN_SEED=$i
        export DEGRADATION_PROFILE=$DEGRADATION_PROFILE
        export KAFKA_BOOTSTRAP_SERVERS=$KAFKA_BOOTSTRAP
        export ROS_MASTER_URI=http://localhost:11311
        export ROS_HOSTNAME=localhost
        source /opt/ros/noetic/setup.bash
        python3 /root/ros_ws/src/ros_kafka_publisher.py
    "
done

echo ""
echo "✅ $N_ENGINES motores corriendo en background dentro de '$ROS_CONTAINER'"
echo ""
echo "Para ver logs de un motor:"
echo "  docker exec $ROS_CONTAINER bash -lc 'tail -f /proc/\$(pgrep -f ENGINE_NUM=1)/fd/1 2>/dev/null || journalctl -f'"
echo ""
echo "Para detener todos los publishers:"
echo "  docker exec $ROS_CONTAINER bash -lc 'pkill -f ros_kafka_publisher.py || true'"
echo ""
echo "Throughput estimado:"
python3 -c "
n=$N_ENGINES; hz=$PUBLISH_HZ
print(f'  raw msgs/s  : {n*hz}')
print(f'  raw msgs/min: {n*hz*60}')
print(f'  ciclos/s    : {n*$INFERENCE_HZ}  (efectivos para inferencia)')
" 2>/dev/null || echo "  raw msgs/min ≈ $((N_ENGINES * PUBLISH_HZ * 60))"
