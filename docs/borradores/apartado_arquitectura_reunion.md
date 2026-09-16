# Arquitectura del proyecto — material para reunión con tutor técnico
(Borrador aparte, no forma parte todavía de ImplementacionContexto.md)

## Infraestructura
Estación de trabajo local, orquestada con Docker Compose (9 contenedores en una red interna compartida). [Completar con vCPU/RAM reales del equipo, igual que hizo tu compañero con "VPS 8 vCPU / 24GB RAM"].

## Ingesta de datos
- **NASA C-MAPSS**: 160 359 filas, 709 trayectorias de motores. Ingesta batch (.txt → HDFS bronze).
- **Apache Kafka**: telemetría del simulador ROS en tiempo real, 10 Hz, tópico `ros_motor_telemetry`.
- **Zookeeper**: coordinación de metadata de los brokers de Kafka.

## Almacenamiento
- **HDFS**, siguiendo la **arquitectura medallion** (bronze → silver → gold), con **dos dominios paralelos e independientes**: NASA y ROS — cada uno con su propia línea base de normalización, nunca se mezclan.
- **PostgreSQL**: almacenamiento de resultados de evaluación (`eval_interna_*`, `eval_externa_*`, `eval_ros_*`).

## Procesamiento
- **Apache Spark (PySpark)**, clúster standalone: 1 master + 1 worker, 6 núcleos.
- Transformaciones bronze → silver → gold: limpieza, normalización condicionada por régimen operativo (regresión), e ingeniería de la feature `cycles_since_regime_change`.

## Analítica avanzada
- **Ensamble de 4 modelos**: Isolation Forest, autoencoder LSTM, autoencoder TCN, autoencoder Transformer — entrenados exclusivamente sobre ventanas de condición sana (30 ciclos).
- **Transferencia de aprendizaje** (fine-tuning): encoder y proyección latente congelados, solo se reentrena el decoder; tasa de aprendizaje 10 veces menor. Dominio NASA → dominio ROS.
- **Evaluación**: correlación de Spearman y monotonicidad. Fusión del ensamble por disyunción (OR): el estado crítico se dispara si al menos un autoencoder lo reporta, con Isolation Forest como confirmación secundaria.

## Visualización y acceso
- **Grafana**: consulta directa a PostgreSQL con refresco periódico. La ingesta (Kafka) sí es en tiempo real, pero el flujo bronze→silver→gold validado en este proyecto opera por lotes, no en streaming continuo de punta a punta.
- **Jupyter**: análisis exploratorio de datos (EDA).
- **Tableau**: apoyo visual complementario para la selección de sensores.

## Orquestación y automatización
- **Docker Compose**: 9 servicios (Kafka, Zookeeper, HDFS namenode/datanode, Spark master/worker, PostgreSQL, Grafana, Jupyter, nodo ROS).
- **Scripts parametrizables por etapa**, ejecutados bajo demanda —a diferencia del enfoque de tu compañero con Airflow y DAGs programados por cadencia, acá la orquestación es manual/script-based—: `run_nasa_stage1.sh`, `run_nasa_stage2.sh`, `run_ros_publishers.sh`, `run_ros_batch.sh`, `run_ros_live.sh`, `run_ros_retrain.sh`.

---

**Diferencia honesta a mencionarle a tu tutor si pregunta**: tu compañero automatiza con Airflow (DAGs programados cada 6 horas / diario); tu proyecto ejecuta cada etapa bajo demanda vía scripts de shell parametrizables. Es una decisión de alcance razonable para un proyecto de tesis con validación puntual (no un sistema de producción continua), pero si el tutor pide justificarlo, esa es la explicación: el objetivo del proyecto es demostrar y validar la arquitectura, no operar un pipeline en producción 24/7.
