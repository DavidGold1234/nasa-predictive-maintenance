#!/usr/bin/env python3

import os
import json
import math
import random
from datetime import datetime, timezone

import rospy
from std_msgs.msg import String
from kafka import KafkaProducer


TOPIC_ROS = "/motor_telemetry"
TOPIC_KAFKA = os.environ.get("ROS_KAFKA_TOPIC", "ros_motor_telemetry")
BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

# Ingesta alta frecuencia, inferencia a frecuencia efectiva controlada
PUBLISH_HZ = int(os.environ.get("PUBLISH_HZ", "10"))
INFERENCE_HZ = int(os.environ.get("INFERENCE_HZ", "1"))
if PUBLISH_HZ <= 0 or INFERENCE_HZ <= 0:
    raise ValueError("PUBLISH_HZ e INFERENCE_HZ deben ser enteros positivos")

# Idealmente, por cada segundo se publiquen PUBLISH_HZ raws y se emitan INFERENCE_HZ ciclos efectivos.
RAW_PER_EFFECTIVE_CYCLE = PUBLISH_HZ / INFERENCE_HZ

# Identidad del motor/corrida
DATASET_ID = os.environ.get("DATASET_ID", "ROS001")
ENGINE_NUM = int(os.environ.get("ENGINE_NUM", "1"))
ENGINE_ID = f"{DATASET_ID}_{ENGINE_NUM}"

# Variación controlada entre corridas y perfiles
RUN_SEED = int(os.environ.get("RUN_SEED", "0"))
DEGRADATION_PROFILE = os.environ.get("DEGRADATION_PROFILE", "medium").strip().lower()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def init_engine_state(engine_num: int):
    seed = 2026 + engine_num * 97 + RUN_SEED * 1009
    rng = random.Random(seed)

    if DEGRADATION_PROFILE == "mild":
        profile = {
            "base_lin": 0.000008,
            "base_quad": 0.000000040,
            "rw_low": -0.00025,
            "rw_high": 0.00030,
            "rw_decay": 0.94,
            "rw_min": -0.004,
            "rw_max": 0.012,
            "shock1_p": 0.010,
            "shock2_p": 0.018,
            "shock3_p": 0.026,
            "shock1_mag": (0.002, 0.006),
            "shock2_mag": (0.004, 0.010),
            "shock3_mag": (0.006, 0.015),
            "shock_decay": 0.90,
            "burst_scale": 7.0,
            "burst_cap": 0.045,
        }
    elif DEGRADATION_PROFILE == "aggressive":
        profile = {
            "base_lin": 0.000014,
            "base_quad": 0.000000075,
            "rw_low": -0.00045,
            "rw_high": 0.00065,
            "rw_decay": 0.90,
            "rw_min": -0.004,
            "rw_max": 0.020,
            "shock1_p": 0.025,
            "shock2_p": 0.040,
            "shock3_p": 0.060,
            "shock1_mag": (0.004, 0.012),
            "shock2_mag": (0.008, 0.020),
            "shock3_mag": (0.012, 0.030),
            "shock_decay": 0.83,
            "burst_scale": 9.0,
            "burst_cap": 0.080,
        }
    else:  # medium
        profile = {
            "base_lin": 0.000010,
            "base_quad": 0.000000055,
            "rw_low": -0.00035,
            "rw_high": 0.00045,
            "rw_decay": 0.92,
            "rw_min": -0.004,
            "rw_max": 0.012,
            "shock1_p": 0.018,
            "shock2_p": 0.028,
            "shock3_p": 0.040,
            "shock1_mag": (0.003, 0.010),
            "shock2_mag": (0.006, 0.018),
            "shock3_mag": (0.008, 0.024),
            "shock_decay": 0.86,
            "burst_scale": 8.0,
            "burst_cap": 0.060,
        }

    return {
        "rng": rng,
        "rw": 0.0,
        "shock": 0.0,
        "profile": profile,
    }


def compute_regime(cycle: int, engine_state: dict):
    rng = engine_state["rng"]

    if cycle < 180:
        load_base, slope_base, speed_base = 0.30, 0.0, 0.40
    elif cycle < 360:
        load_base, slope_base, speed_base = 0.48, 4.0, 0.58
    else:
        load_base, slope_base, speed_base = 0.70, 8.0, 0.80

    # pequeñas variaciones operacionales
    load = load_base + rng.uniform(-0.015, 0.015)
    slope = slope_base + rng.uniform(-0.35, 0.35)
    speed_cmd = speed_base + rng.uniform(-0.02, 0.02)

    return load, slope, speed_cmd


def compute_degradation(cycle: int, engine_state: dict):
    rng = engine_state["rng"]
    p = engine_state["profile"]

    base = p["base_lin"] * cycle + p["base_quad"] * (cycle ** 2)

    # random walk amortiguado
    engine_state["rw"] = (
        p["rw_decay"] * engine_state["rw"] + rng.uniform(p["rw_low"], p["rw_high"])
    )
    engine_state["rw"] = clamp(engine_state["rw"], p["rw_min"], p["rw_max"])

    # picos/eventos locales de degradación
    if cycle > 120 and rng.random() < p["shock1_p"]:
        engine_state["shock"] += rng.uniform(*p["shock1_mag"])
    if cycle > 260 and rng.random() < p["shock2_p"]:
        engine_state["shock"] += rng.uniform(*p["shock2_mag"])
    if cycle > 420 and rng.random() < p["shock3_p"]:
        engine_state["shock"] += rng.uniform(*p["shock3_mag"])

    # decaimiento del pico
    engine_state["shock"] *= p["shock_decay"]

    # ondulación leve dependiente del motor y corrida
    periodic = (
        0.0012 * math.sin(cycle / 27.0 + ENGINE_NUM * 0.7 + RUN_SEED * 0.13)
        + 0.0008 * math.sin(cycle / 9.5 + ENGINE_NUM * 0.33 + RUN_SEED * 0.07)
    )

    degradation = max(
        0.0,
        base + engine_state["rw"] + engine_state["shock"] + periodic,
    )

    local_burst = min(engine_state["shock"] * p["burst_scale"], p["burst_cap"])
    return degradation, local_burst


def build_row(cycle: int, engine_state: dict):
    rng = engine_state["rng"]
    load, slope, speed_cmd = compute_regime(cycle, engine_state)
    degradation, local_burst = compute_degradation(cycle, engine_state)

    # señales base ROS con tendencia + picos
    current = (
        2.0
        + 0.85 * load
        + 0.035 * slope
        + 15.0 * degradation
        + 0.9 * local_burst
        + rng.uniform(-0.02, 0.02)
    )

    rpm = (
        120.0
        - 7.0 * load
        - 0.55 * slope
        - 180.0 * degradation
        - 6.0 * local_burst
        + rng.uniform(-0.6, 0.6)
    )

    temp = (
        30.0
        + 2.8 * load
        + 0.22 * slope
        + 72.0 * degradation
        + 8.0 * local_burst
        + rng.uniform(-0.10, 0.10)
    )

    voltage = (
        12.0
        - 1.0 * degradation
        - 0.10 * local_burst
        + rng.uniform(-0.015, 0.015)
    )

    # adaptación ROS -> escala más cercana a C-MAPSS
    current_n = current / 5.0
    rpm_n = rpm / 150.0
    temp_n = temp / 100.0
    voltage_n = voltage / 15.0

    row = {
        "timestamp": utc_now_iso(),
        "dataset_id": DATASET_ID,
        "engine_id": ENGINE_ID,
        "engine_num": ENGINE_NUM,
        "cycle": cycle,
        "op_setting_1": load,
        "op_setting_2": slope,
        "op_setting_3": speed_cmd,
        "sensor_2": current_n,
        "sensor_3": current_n * 0.985 + 0.10 * degradation + rng.uniform(-0.004, 0.004),
        "sensor_4": current_n * 1.015 + 0.16 * degradation + 0.15 * local_burst + rng.uniform(-0.004, 0.004),
        "sensor_7": rpm_n,
        "sensor_8": rpm_n * 0.992 - 0.08 * degradation - 0.04 * local_burst + rng.uniform(-0.004, 0.004),
        "sensor_9": rpm_n * 1.008 - 0.11 * degradation - 0.06 * local_burst + rng.uniform(-0.004, 0.004),
        "sensor_11": temp_n,
        "sensor_14": temp_n * 0.992 + 0.14 * degradation + 0.10 * local_burst + rng.uniform(-0.004, 0.004),
        "sensor_15": temp_n * 1.008 + 0.18 * degradation + 0.14 * local_burst + rng.uniform(-0.004, 0.004),
        "sensor_17": voltage_n - 0.18 * degradation - 0.08 * local_burst + rng.uniform(-0.003, 0.003),
        "sensor_21": degradation + 0.35 * local_burst + rng.uniform(-0.002, 0.002),
    }
    return row


def apply_raw_jitter(row: dict, raw_idx: int, raw_per_cycle_estimate: float):
    """
    Añade jitter pequeño a mensajes de alta frecuencia sin cambiar el cycle efectivo.
    Esto hace más dinámico el bronze, pero no acelera artificialmente la degradación.
    """
    denom = max(1.0, raw_per_cycle_estimate)
    phase = (raw_idx / denom) * 2.0 * math.pi
    out = dict(row)

    out["timestamp"] = utc_now_iso()

    # jitter muy pequeño, solo observacional
    out["sensor_2"] += 0.0008 * math.sin(phase)
    out["sensor_3"] += 0.0010 * math.sin(phase + 0.2)
    out["sensor_4"] += 0.0012 * math.sin(phase + 0.5)

    out["sensor_7"] += 0.0010 * math.cos(phase)
    out["sensor_8"] += 0.0010 * math.cos(phase + 0.3)
    out["sensor_9"] += 0.0012 * math.cos(phase + 0.6)

    out["sensor_11"] += 0.0008 * math.sin(phase + 0.4)
    out["sensor_14"] += 0.0010 * math.sin(phase + 0.7)
    out["sensor_15"] += 0.0012 * math.sin(phase + 0.9)

    out["sensor_17"] += 0.0006 * math.cos(phase + 0.8)
    out["sensor_21"] += 0.0005 * math.sin(phase + 1.1)

    return out


def main():
    rospy.init_node("ros_kafka_publisher", anonymous=True)
    pub = rospy.Publisher(TOPIC_ROS, String, queue_size=10)

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    if INFERENCE_HZ > PUBLISH_HZ:
        rospy.logwarn(
            "INFERENCE_HZ (%s) es mayor que PUBLISH_HZ (%s). Cada raw intentará actuar como ciclo efectivo.",
            INFERENCE_HZ,
            PUBLISH_HZ,
        )
    elif not RAW_PER_EFFECTIVE_CYCLE.is_integer():
        rospy.logwarn(
            "PUBLISH_HZ/INFERENCE_HZ = %.4f no es entero. Se usará acumulador para repartir 3/4 raws por ciclo según corresponda.",
            RAW_PER_EFFECTIVE_CYCLE,
        )

    rate = rospy.Rate(PUBLISH_HZ)
    cycle = 1
    raw_count_in_cycle = 0
    cycle_progress_acc = 0
    engine_state = init_engine_state(ENGINE_NUM)
    current_cycle_row = None

    rospy.loginfo(
        f"ROS publisher listo | engine={ENGINE_ID} | "
        f"publish_hz={PUBLISH_HZ} | inference_hz={INFERENCE_HZ} | "
        f"raw/effective≈{RAW_PER_EFFECTIVE_CYCLE:.3f} | "
        f"profile={DEGRADATION_PROFILE} | run_seed={RUN_SEED}"
    )

    try:
        while not rospy.is_shutdown():
            # generar una sola vez por ciclo efectivo
            if raw_count_in_cycle == 0 or current_cycle_row is None:
                current_cycle_row = build_row(cycle, engine_state)

            row = apply_raw_jitter(
                current_cycle_row,
                raw_idx=raw_count_in_cycle,
                raw_per_cycle_estimate=RAW_PER_EFFECTIVE_CYCLE,
            )

            pub.publish(json.dumps(row))
            producer.send(TOPIC_KAFKA, row)

            raw_count_in_cycle += 1
            cycle_progress_acc += INFERENCE_HZ

            rospy.loginfo(
                f"engine={ENGINE_ID} cycle={cycle} "
                f"| raw_in_cycle={raw_count_in_cycle} "
                f"| acc={cycle_progress_acc}/{PUBLISH_HZ} "
                f"| publish_hz={PUBLISH_HZ} "
                f"| profile={DEGRADATION_PROFILE}"
            )

            if cycle_progress_acc >= PUBLISH_HZ:
                cycle += 1
                raw_count_in_cycle = 0
                cycle_progress_acc -= PUBLISH_HZ

            rate.sleep()
    finally:
        try:
            producer.flush(timeout=10)
        finally:
            producer.close(timeout=10)


if __name__ == "__main__":
    main()
