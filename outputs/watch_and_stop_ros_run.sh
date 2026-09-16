#!/usr/bin/env bash
set -uo pipefail

BASELINE=12448
TARGET=$((BASELINE + 1000000))
MAX_SECONDS=7200   # 2h de margen (se espera llegar en ~66 min)
START_TS=$(date +%s)

echo "[$(date -u +'%H:%M:%S')] Iniciando vigilancia. Baseline=$BASELINE Target=$TARGET MaxSeconds=$MAX_SECONDS"

while true; do
  sleep 600
  NOW_TS=$(date +%s)
  ELAPSED=$((NOW_TS - START_TS))
  OFFSET=$(docker exec kafka bash -c "kafka-run-class kafka.tools.GetOffsetShell --broker-list localhost:9092 --topic ros_motor_telemetry --time -1" 2>/dev/null | cut -d: -f3)
  ALIVE=$(docker exec ros-gazebo bash -c "pgrep -af ros_kafka_publisher.py | grep -c 'python3 src'" 2>/dev/null)
  echo "[$(date -u +'%H:%M:%S')] elapsed=${ELAPSED}s offset=${OFFSET:-desconocido} publishers_vivos=${ALIVE:-0}"

  if [ -z "$OFFSET" ]; then
    echo "[$(date -u +'%H:%M:%S')] ADVERTENCIA: no se pudo leer el offset de Kafka (revisar contenedor)."
  fi

  if [ -n "$OFFSET" ] && [ "$OFFSET" -ge "$TARGET" ]; then
    echo "[$(date -u +'%H:%M:%S')] Meta de volumen alcanzada (offset=$OFFSET >= target=$TARGET). Deteniendo publishers."
    docker exec ros-gazebo bash -c "pkill -f ros_kafka_publisher.py" 2>/dev/null
    echo "RESULTADO=meta_volumen offset_final=$OFFSET elapsed=${ELAPSED}s"
    break
  fi

  if [ "$ELAPSED" -ge "$MAX_SECONDS" ]; then
    echo "[$(date -u +'%H:%M:%S')] Tiempo maximo alcanzado (${ELAPSED}s >= ${MAX_SECONDS}s). Deteniendo publishers."
    docker exec ros-gazebo bash -c "pkill -f ros_kafka_publisher.py" 2>/dev/null
    echo "RESULTADO=tiempo_maximo offset_final=${OFFSET:-desconocido} elapsed=${ELAPSED}s"
    break
  fi

  if [ "${ALIVE:-0}" -eq 0 ]; then
    echo "[$(date -u +'%H:%M:%S')] ADVERTENCIA: no queda ningun publisher vivo (se cayeron todos). Deteniendo vigilancia."
    echo "RESULTADO=publishers_caidos offset_final=${OFFSET:-desconocido} elapsed=${ELAPSED}s"
    break
  fi
done

echo "[$(date -u +'%H:%M:%S')] Vigilancia terminada."
