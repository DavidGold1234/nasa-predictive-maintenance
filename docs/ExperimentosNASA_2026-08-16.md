# Bitácora de experimentos — Pipeline NASA CMAPSS (2026-08-16)

Registro de la sesión de trabajo en la que se diagnosticó y corrigió el desempeño de los modelos de detección de anomalías entrenados sobre NASA CMAPSS (`train_and_evaluate.py`), y de los cambios de preprocesamiento asociados. Todo lo documentado aquí se ejecutó y midió realmente sobre el pipeline (Spark + PyTorch, contenedores Docker), no son estimaciones.

## Punto de partida

Antes de esta sesión, la comparación entre modelos usaba el `val_loss` (error de reconstrucción) crudo como referencia, lo que llevaba a pensar que el LSTM Autoencoder estaba "roto" (val_loss≈0.50) frente al CNN Autoencoder (val_loss≈0.002). El primer paso fue correr el pipeline real y mirar las métricas que de verdad miden calidad de detección: `spearman` (correlación score vs -RUL), `early_late_delta` y `monotonicity` (qué tan limpia es la tendencia ascendente del score a medida que el motor se degrada), calculadas por `group_internal_metrics()` en `train_and_evaluate.py`.

**Conclusión del diagnóstico inicial**: el LSTM no estaba roto. El CNN Autoencoder original no tenía un cuello de botella real (conservaba los 30 timesteps en todas las capas, sin comprimir), por lo que reconstruía casi cualquier ventana casi perfecto —de ahí su val_loss bajísimo— pero era el peor de los tres modelos detectando la tendencia real de degradación.

## Experimento 1 — Bottleneck real en el CNN

**Cambio**: se rediseñó `CNNAutoencoder` con downsampling temporal real (dos convoluciones `stride=2`: 30→15→8 pasos) + proyección a un vector latente de 32 dimensiones, igual que el LSTM.

**Resultado** (test interno NASA):

| Modelo | spearman | monotonicity |
|---|---|---|
| IsolationForest | 0.789 | 0.107 |
| LSTM_AE | 0.774 | 0.356 |
| CNN_AE (antes, sin bottleneck) | 0.640 | 0.041 |
| CNN_AE (con bottleneck real) | 0.640 | **0.097** (2.4x mejor) |

Mejora real pero insuficiente: el CNN seguía siendo, por lejos, el modelo más débil.

## Experimento 2 — Normalización por regresión continua (en vez de KMeans + z-score por cluster)

**Hipótesis**: la normalización original agrupaba las condiciones operativas en 6 clusters (KMeans) y aplicaba z-score por cluster. Se diagnosticó empíricamente (`outputs/diagnose_regime_switching.py`) que en FD002/FD004 el régimen operativo cambia en el **82.5% de los ciclos consecutivos**, y el **100% de las ventanas de 30 ciclos cruzan más de un cluster** — cada ciclo normalizado con una media/std distinta, metiendo saltos artificiales en la ventana.

**Cambio**: se reemplazó el z-score discreto por cluster con una línea base continua por regresión polinómica (`sensor ~ PolynomialFeatures(degree=2)(op_settings)`, ajustada solo en sano-train), en `run_preprocessing.py`, `build_ros_gold.py`, `external_evaluation.py` y `ros_kafka_inference.py`.

**Resultado — mejora pequeña y mixta, no la esperada**:

| Modelo | spearman (antes) | spearman (regresión) | monotonicity (antes) | monotonicity (regresión) |
|---|---|---|---|---|
| IsolationForest | 0.789 | 0.797 | 0.107 | 0.115 |
| LSTM_AE | 0.774 | 0.782 | 0.356 | 0.348 |
| CNN_AE | 0.640 | 0.669 | 0.097 | 0.081 |

**Por qué no funcionó como se esperaba**: se verificó con un segundo diagnóstico (`outputs/diagnose_regime_switching_v3.py`) que la razón salto/no-salto **no mejoró** (FD002: 1.155→1.149; FD004: 1.168→1.202). Causa raíz real: los `op_settings` de FD002/FD004 no son continuos — el motor salta entre ~6 puntos de vuelo **discretos** por diseño del propio dataset CMAPSS. Una regresión evaluada en esos mismos 6 puntos no es más suave que un lookup por cluster; el salto adicional es más probablemente una **dinámica transitoria real** (el sensor tarda en estabilizarse tras el cambio de régimen), no un artefacto de normalización.

Este experimento se documenta como hallazgo metodológico honesto (hipótesis → prueba empírica → resultado modesto), no como "la solución".

## Experimento 3 — Feature `cycles_since_regime_change` (dinámica transitoria)

**Hipótesis derivada del experimento 2**: si el salto es una dinámica transitoria, dársela al modelo como información explícita (en vez de esperar que la infiera) debería ayudar más que seguir afinando la normalización.

**Cambio**: nueva feature `cycles_since_regime_change` (0 = el régimen acaba de cambiar; sube 1 por cada ciclo que el motor se mantiene en el mismo régimen; tope en `WINDOW_SIZE=30`). Se detecta con un KMeans liviano (k≤6, solo para esta detección, no para normalizar) y se escala con su propio `StandardScaler`. Pasa a ser una **12ª columna de entrada** del modelo (antes 11 sensores). Implementado en `run_preprocessing.py` + `train_and_evaluate.py` (alcance NASA únicamente).

**Resultado — mejora real, más clara**:

| Modelo | spearman | monotonicity | val_loss |
|---|---|---|---|
| IsolationForest | **0.822** (+0.025) | 0.111 | — |
| LSTM_AE | **0.798** (+0.016) | **0.356** (vuelve al nivel original) | 0.480 (-5% vs regresión sola) |
| CNN_AE | 0.671 | 0.074 | 0.371 (-8%) |

Isolation Forest fue el más beneficiado (puede usar la feature directamente para refinar sus splits). El CNN, pese a bajar su val_loss, se mantuvo como el modelo más débil en monotonicity — reforzando que su problema es arquitectónico, no de datos.

## Experimento 4 — Reemplazo de CNN por TCN + nuevo modelo Transformer (4 modelos)

**Diagnóstico final del CNN**: aunque su bottleneck ya comprimía bien, la capa `Linear` final combinaba el vector aplanado (128 valores) **sin ninguna noción de orden temporal** — a diferencia del LSTM, que procesa la secuencia de forma recurrente y causal. Ensanchar el kernel no habría resuelto esto (el aplanado ya tenía acceso a toda la ventana; el problema era la falta de sesgo inductivo secuencial, no de campo de visión).

**Cambio**: se reemplazó `CNNAutoencoder` por `TCNAutoencoder` (Temporal Convolutional Network: convoluciones **causales** —solo ven el pasado— y **dilatadas** —dilatación 1,2,4,8 con kernel=3, campo receptivo de 31 pasos, cubre la ventana completa de forma causal—, diseño estándar de bloques residuales de Bai et al. 2018). Se agregó además un cuarto modelo, `TransformerAutoencoder` (self-attention con codificación posicional senoidal), a pedido explícito para completar un ensemble de 4 modelos. Implementado en `train_and_evaluate.py` (alcance NASA únicamente).

**Resultado final — TCN y Transformer superan a CNN en las tres métricas a la vez**:

| Modelo | spearman | monotonicity | val_loss |
|---|---|---|---|
| IsolationForest | 0.822 | 0.111 | — |
| **LSTM_AE** | 0.798 | **0.356** | 0.480 |
| **TCN_AE** (reemplaza a CNN_AE) | 0.693 | 0.192 (2.6x vs CNN) | 0.429 |
| **Transformer_AE** (nuevo) | 0.695 | 0.243 (3.3x vs CNN) | 0.426 |
| *CNN_AE (descartado, referencia)* | *0.671* | *0.074* | *0.405* |

Confirma la hipótesis: la falta de sesgo inductivo secuencial —no el cómputo ni los datos— era la limitación real del CNN. Ni TCN ni Transformer alcanzan todavía a LSTM_AE en `monotonicity` (0.356), e IsolationForest sigue siendo el líder en `spearman` (0.822) pese a ser el modelo más simple del ensemble.

## Resumen de la evolución completa (spearman / monotonicity, LSTM y modelo "CNN→TCN/Transformer")

| Etapa | LSTM_AE | CNN_AE / TCN_AE / Transformer_AE |
|---|---|---|
| 0. Baseline | 0.774 / 0.356 | CNN: 0.640 / 0.041 |
| 1. CNN con bottleneck real | 0.774 / 0.356 | CNN: 0.640 / 0.097 |
| 2. + normalización por regresión | 0.782 / 0.348 | CNN: 0.669 / 0.081 |
| 3. + feature `cycles_since_regime_change` | 0.798 / 0.356 | CNN: 0.671 / 0.074 |
| 4. CNN → TCN + Transformer (4 modelos) | 0.798 / 0.356 | TCN: 0.693 / 0.192 · Transformer: 0.695 / 0.243 |

## Etapa 5 — Sincronización de los 5 archivos ROS/evaluación externa

Los 4 archivos que quedaron pendientes (`evaluate_ros_from_gold.py`, `ros_kafka_inference.py`, `retrain_ros_models.py`, `external_evaluation.py`) más uno que apareció al revisar a fondo (`build_ros_gold.py`, que no generaba `cycles_since_regime_change` para ROS) se sincronizaron en una sola pasada:

- **`build_ros_gold.py`**: nueva `fit_ros_regime_duration()` — detector de régimen ROS-específico (KMeans propio, k≤6). A diferencia de la normalización, **nunca** usa el detector NASA como fallback (viven en espacios numéricos distintos — altitud/Mach/acelerador vs torque/pendiente/temperatura — el detector NASA metería casi todo en el mismo cluster). Con pocos datos sanos, la feature queda deshabilitada (0 constante) en vez de dar una señal cruzada sin sentido. Nuevos artifacts: `regime_detector_kmeans.pkl`, `regime_duration_scaler.pkl`.
- **`external_evaluation.py`**: calcula la misma feature pero usando el detector NASA directo (mismo dominio, sin adaptación) + cambio de modelos.
- **`evaluate_ros_from_gold.py`** y **`retrain_ros_models.py`**: cambio de modelos, 12 features de entrada, `fuse_status()`/`freeze_encoder()` extendidos a los 3 autoencoders (LSTM, TCN, Transformer).
- **`ros_kafka_inference.py`** (el más delicado): agrega estado por motor en memoria (`last_regime_label`, `cycles_since_change`) para calcular la feature fila por fila en el streaming en vivo, igual que ya mantiene el buffer de ventana por motor.
- **`ingest_ros_alerts_to_predictions.py`**: schema actualizado al nuevo payload.

**Validación real ejecutada** (no hay datos ROS/Gazebo reales en este entorno — se generó `ros_clean` sintético con `outputs/generate_synthetic_ros_silver.py`, 4 motores × 220 ciclos): se corrió la cadena completa `build_ros_gold.py` → `retrain_ros_models.py` → `evaluate_ros_from_gold.py` sin errores (ventanas con forma `(N, 30, 12)` confirmada), se probó `ros_kafka_inference.py` de forma aislada confirmando que el contador de "ciclos desde el cambio de régimen" resetea correctamente al cambiar de régimen, y se corrió `external_evaluation.py` contra datos reales de NASA FD001 (100 motores, 10196 ventanas, sin errores).

## Bugs incidentales encontrados y corregidos

1. `write_to_postgres()` en `train_and_evaluate.py` tenía un bug latente de Python 3: `except Exception as first_error:` — Python borra esa variable al salir del bloque `except`, así que si tanto la escritura por SQLAlchemy como por Spark JDBC fallaban, el mensaje de error final crasheaba con `UnboundLocalError` en vez de mostrar el error real. Se corrigió copiando el mensaje a una variable normal dentro del bloque (mismo patrón ya usado correctamente en `external_evaluation.py`).
2. `build_ros_gold.py`'s `latest_active_version()` devolvía `None` cuando no existe `active_version.json` todavía (primera corrida antes de cualquier reentrenamiento) — la columna `artifact_version` quedaba con `None` en todas las filas y Spark no podía inferirle un tipo (`CANNOT_DETERMINE_TYPE`), tumbando el script en el último paso. Se corrigió devolviendo `"unknown"` en vez de `None`.
3. **Hallazgo sin corregir (fuera de alcance)**: `load_pretrained_weights()` y `freeze_encoder()` en `retrain_ros_models.py` están definidas pero **nunca se llaman** en `main()` — el fine-tuning "congela encoder NASA, entrena decoder ROS" descrito en los docstrings no está realmente conectado; el reentrenamiento ROS actual entrena todo desde cero cada vez. Es un gap preexistente, no introducido en esta sesión.

## Archivos modificados en esta sesión

- `src/preprocessing/run_preprocessing.py` — normalización por regresión + detección de régimen + feature `cycles_since_regime_change`.
- `src/features/build_ros_gold.py` — adaptación ROS de la normalización por regresión + detección de régimen ROS + corrección del bug de `artifact_version`.
- `src/models/train_and_evaluate.py` — `TCNAutoencoder`, `TransformerAutoencoder`, 12 features de entrada, corrección de bug en `write_to_postgres`.
- `src/models/evaluate_ros_from_gold.py`, `src/models/ros_kafka_inference.py`, `src/models/external_evaluation.py` — migrados a la normalización por regresión, a `cycles_since_regime_change` (12 features) y a `TCNAutoencoder`+`TransformerAutoencoder`.
- `src/models/retrain_ros_models.py`, `src/models/deploy_latest_version.py`, `src/models/rollback_model_version.py` — nombres de artifacts actualizados (`regime_baseline.pkl`, `regime_detector_kmeans.pkl`, `regime_duration_scaler.pkl`) y modelos TCN/Transformer.
- `apps/ingestion/ingest_ros_alerts_to_predictions.py` — schema actualizado (sin `operating_cluster`, con `tcn_score`/`transformer_score`).
- `outputs/diagnose_regime_switching*.py`, `outputs/score_only_4models.py`, `outputs/generate_synthetic_ros_silver.py` — scripts de diagnóstico y validación usados durante la sesión (no forman parte del pipeline de producción).

## Etapa 6 — Cierre de los 3 pendientes (Postgres, fuse_status, fine-tuning NASA→ROS)

1. **Artifacts NASA restaurados**: se recuperaron los artifacts reales del run `nasa_train_20260816_231754` (modelos) y `preprocessing_20260816_220101` (preprocesamiento) desde el respaldo en HDFS, sobrescribiendo los que habían quedado contaminados por las pruebas sintéticas ROS. Verificado con `model_metadata.json`: `run_id=nasa_train_20260816_231754`, `n_features=12`, 4 modelos. También se eliminaron `active_version.json`/`retrain_metadata.json`, que habían quedado describiendo el reentrenamiento ROS sintético (información engañosa sobre el estado activo real).
2. **Postgres**: se inspeccionó el schema real de las 6 tablas de evaluación antes de tocar nada. Solo `eval_interna_predictions` y `eval_externa_predictions` tenían columnas obsoletas (`cnn_score`, y `operating_cluster` en la interna) — son las únicas en formato ancho (una columna por modelo). `eval_interna_metrics`/`eval_interna_runs`/`eval_externa_summary`/`eval_externa_weighted` son formato largo (columna `model` como valor de fila) y nunca tuvieron el problema. Se borraron solo esas 2 tablas; se recrean automáticamente con el schema correcto en la próxima escritura.
3. **`fuse_status()`**: decisión final del usuario fue **dejarla igual** (los 3 autoencoders — LSTM, TCN, Transformer — como gatillo primario OR, IF como confirmación secundaria). No se requirió cambio de código.
4. **Fine-tuning NASA→ROS conectado**: `load_pretrained_weights()` y `freeze_encoder()` estaban definidas pero huérfanas (nunca llamadas desde `main()` de `retrain_ros_models.py`) — el "reentrenamiento" ROS en realidad entrenaba todo desde cero. Se agregó `prepare_for_finetune()` que las conecta: carga los pesos NASA activos, congela encoder+proyección latente si `FREEZE_ENCODER=1` (default), y usa `FINETUNE_LR` (10x más bajo) en vez de `LEARNING_RATE` cuando hay fine-tuning real. Validado en modo solo-lectura contra los artifacts NASA reales (sin reentrenar, para no volver a contaminarlos): los 3 modelos cargan los pesos con `strict=True` sin error y quedan con la proporción esperada de parámetros congelados (LSTM ~4% entrenable — encoder bidireccional grande —, TCN y Transformer ~50% cada uno). `retrain_metadata.json` ahora registra `*_finetuned_from_nasa` y `finetune_lr` para trazabilidad.

## Pendientes conocidos (sin resolver todavía)

1. Nunca se hizo una comparación limpia "regresión vs KMeans" ya con la feature de duración añadida a ambos métodos.
2. El fine-tuning NASA→ROS ya está conectado en el código pero **no se ha validado con un reentrenamiento real completo** (solo se probó la carga de pesos + congelamiento en modo lectura, para no volver a sobrescribir los artifacts NASA reales). La próxima vez que se corra `retrain_ros_models.py` con datos ROS reales, conviene revisar las curvas de val_loss para confirmar que el fine-tuning converge razonablemente con `FINETUNE_LR=0.0001`.
3. **Ver Etapa 7 abajo: intento de levantar datos ROS reales — bloqueado por infraestructura, no por el código de ML.**

## Etapa 7 — Intento de generar datos ROS reales (streaming) — bloqueado por infraestructura

Dado que el objetivo específico 4 de la tesis compromete "validar en un entorno de simulación robótica basado en ROS", y que hasta este punto solo se había probado el código con datos sintéticos generados por fuera del pipeline real, se intentó levantar la cadena completa real: `ros-gazebo` (publishers) → Kafka → Spark Structured Streaming → bronze → silver → gold → reentrenamiento/evaluación.

**Hallazgo importante sobre el propio diseño del proyecto**: el servicio `ros-gazebo` **no ejecuta una simulación física de Gazebo** — el `Dockerfile` (`infrastructure/ros/Dockerfile`) instala únicamente `ros:noetic-ros-base` (sin paquetes de Gazebo). Lo que sí es real es un nodo ROS (`ros_ws/src/ros_kafka_publisher.py`) que genera telemetría con un simulador paramétrico de degradación (tendencia base + random walk amortiguado + eventos de "shock" probabilísticos + ondulación periódica, con escalado numérico ajustado para parecerse a C-MAPSS) y la publica de verdad por un tópico ROS y por Kafka. Es decir: "ROS" en este proyecto se refiere al middleware de mensajería (nodos, tópicos, `rospy`), no a una simulación física 3D. Esto no es necesariamente un problema para la tesis —muchos trabajos usan "simulación basada en ROS" en ese sentido— pero es importante que quede explícito y consciente, no asumido.

Se identificaron tres problemas de infraestructura reales, independientes entre sí, todos preexistentes (no introducidos en esta sesión):

1. **Los publishers de ROS se cuelgan tras una ráfaga inicial.** `ros_kafka_publisher.py` usa `rospy.init_node()`, que requiere un `roscore`/master ROS activo. Ningún servicio del `docker-compose.yml` levanta un `roscore` dentro (o para) del contenedor `ros-gazebo`. Se observó una ráfaga inicial de ~76 000 mensajes publicados muy rápido (incompatible con el `PUBLISH_HZ=10` configurado, sugiriendo que `rospy.Rate.sleep()` no estaba limitando correctamente sin un clock ROS funcional) y luego los 3 procesos quedaron vivos (confirmado con `pgrep`) pero sin producir ningún mensaje más durante 20+ minutos.
2. **Retención de Kafka agresiva.** Para cuando se intentó leer esos ~76 000 mensajes en modo batch (`earliest`→`latest`), `GetOffsetShell` mostró que el offset más antiguo y el más nuevo del topic eran el mismo número — es decir, ya se habían purgado por completo. No se identificó qué configuración de retención está causando esto (no está fijada explícitamente en `docker-compose.yml`, así que corre con defaults de la imagen `confluentinc/cp-kafka:7.4.0`).
3. **`run_ros_batch.sh` apuntaba a rutas de archivo inexistentes** (`/apps/processing/build_ros_gold.py`, `/apps/models/evaluate_ros_from_gold.py`) en vez de las reales (`/src/features/build_ros_gold.py`, `/src/models/evaluate_ros_from_gold.py`). **Corregido en esta sesión.** Es evidencia adicional de que esta ruta del pipeline (bronze real → silver → gold vía batch script) nunca se había ejecutado de punta a punta antes.

**Ninguno de estos tres se resolvió** — requieren depuración iterativa de infraestructura (decidir cómo levantar un `roscore` mínimo o quitar la dependencia de `rospy.init_node()`, y ajustar la retención de Kafka), no son arreglos de una sola pasada. Se detuvieron los procesos de publisher colgados para liberar recursos; los contenedores (`kafka`, `ros-gazebo`) quedaron arriba por si se retoma la depuración.

**Conclusión de la Etapa 7**: el objetivo específico 4 (validación en entorno de simulación ROS) sigue sin evidencia empírica real — todo lo probado hasta ese punto usó datos sintéticos generados fuera del pipeline (`outputs/generate_synthetic_ros_silver.py`) solo para verificar que el código no falla. Antes de poder reportar resultados de validación ROS en la tesis, hacía falta resolver estos tres problemas de infraestructura.

## Etapa 8 — Datos ROS reales de punta a punta (Kafka real → bronze → silver → gold → fine-tuning → evaluación)

Continuación directa de la Etapa 7: se resolvieron los tres bloqueadores de infraestructura y se corrió la cadena completa con datos ROS reales por primera vez en el proyecto.

### Resolución de los 3 bloqueadores

1. **`roscore` faltante**: se confirmó con `pgrep -af roscore` que, en efecto, no había ningún proceso `roscore`/`rosmaster` corriendo en `ros-gazebo`, y `ROS_MASTER_URI` estaba vacío. Se levantó `roscore` manualmente en background dentro del contenedor (`roscore &`, puerto 11311). Con el master activo, `ros_kafka_publisher.py` publicó a la tasa correcta (10 Hz reales, confirmado con las marcas de tiempo de los logs: ~100ms entre mensajes, 1 ciclo efectivo/segundo) — el comportamiento errático anterior (ráfaga de ~76 000 mensajes sin control de tasa) se debía exactamente a la ausencia de `roscore`: sin un master ROS activo, `rospy.Rate.sleep()` no tiene una noción de tiempo válida y no limita nada.
2. **"Retención agresiva de Kafka" — diagnóstico corregido**: al inspeccionar los segmentos de log en disco del tópico (`/var/lib/kafka/data/ros_motor_telemetry-0/`), se encontró que el segmento activo (offset 75944) tenía fecha **14 de abril** y 0 bytes — es decir, los ~76 000 mensajes de la ráfaga anterior eran de una sesión mucho más antigua, y ya habían expirado legítimamente por el retention.ms por defecto (`log.retention.hours=168`, verificado directamente en `server.properties`). **No hay ninguna retención agresiva mal configurada** — la Etapa 7 diagnosticó mal la causa; el verdadero problema seguía siendo únicamente el `roscore` faltante (los publishers nunca llegaron a producir datos utilizables en el intento anterior porque no había limitación de tasa real, y para cuando se intentó leerlos ya habían expirado por antigüedad real, no por una purga rápida).
3. **Rutas de `run_ros_batch.sh`**: ya corregidas en la Etapa 7.

### Ejecución real

Con `roscore` activo, se lanzaron 3 publishers reales (motores 1/2/3, perfiles `medium`/`mild`/`aggressive`) durante ~9 minutos, acumulando ~12 187 mensajes Kafka reales (~400 ciclos efectivos por motor). Se corrió la cadena completa:

- `outputs/batch_kafka_to_bronze.py` (agregado el paquete `spark-sql-kafka-0-10` que faltaba en el comando `spark-submit`) → bronze real: **12 187 filas**.
- `run_ros_batch.sh silver` → silver: **12 187 filas** (0 pérdidas en limpieza).
- `run_ros_batch.sh gold` → gold: **1 219 filas**. La detección de desfase de distribución (`check_distribution_compatibility`) funcionó exactamente como se diseñó: detectó automáticamente que los sensores ROS están en una escala completamente distinta a NASA (diferencias de 9σ a 103σ) y usó la línea base ROS-específica en vez de la NASA, sin intervención manual.
- `retrain_ros_models.py` → **primera validación real (no solo lectura) del fine-tuning NASA→ROS**: los 3 autoencoders cargaron los pesos NASA, congelaron el encoder (`freeze_encoder`) y entrenaron solo el decoder con `FINETUNE_LR=0.0001` durante 20 épocas. El `val_loss` bajó de forma monótona y estable en los 3 modelos (LSTM: 0.863→0.845; TCN: 0.855→0.840; Transformer: 0.848→0.833), sin señales de divergencia — confirma que la estrategia de transferencia converge razonablemente con datos reales, cerrando el pendiente #2 de la Etapa 6.
- `run_ros_batch.sh evaluate` → escribió 532 predicciones y 3 filas de métricas reales en PostgreSQL (`eval_ros_predictions`, `eval_ros_metrics`).

Un problema de permisos incidental (`/data/models/artifacts` con archivos `root:root` de restauraciones previas vía `namenode`, pero `spark-submit` corriendo como usuario `spark`) causó un `PermissionError` en el paso `gold`; se corrigió con `chown spark:spark` sobre los artifacts.

### Hallazgo empírico real: el status salta a CRITICAL de inmediato, dominado por el salto de régimen — no por degradación

Al revisar `eval_ros_predictions`, el `final_status` es `CRITICAL` desde el primer ciclo evaluado (ciclo 230, justo después de warmup=29 + baseline=200) en **los 3 motores por igual**, sin importar el perfil de degradación configurado (`mild`/`medium`/`aggressive`) — una señal de que el disparo no está midiendo degradación real, sino otra cosa común a los 3 motores.

Se rastreó la causa con los propios artifacts:

- `score_calibration.json` (construido por `retrain_ros_models.py` sobre ventanas sanas ROS) espera scores LSTM en el rango **0.56–1.27** (escala MSE de reconstrucción típica).
- Los scores reales observados en `eval_ros_predictions` desde el ciclo 230 están en el rango **2 600–47 700** — 3 a 4 órdenes de magnitud por encima.
- Revisando `get_healthy_index()` en `build_ros_gold.py`: la línea base "sana" ROS se define como el primer `HEALTHY_RATIO=30%` de los ciclos de cada motor. Para un motor con ~400 ciclos totales, eso son los ciclos ~1–126.
- Revisando `compute_regime()` en `ros_kafka_publisher.py`: el simulador ROS cambia de régimen operativo de forma **discreta y determinística** en `cycle=180` y `cycle=360` (saltos de `load`/`slope`/`speed_cmd`), independientemente del perfil de degradación.

**Conclusión**: el 30% inicial usado como "sano" para ajustar la línea base de regresión ROS-específica (`fit_ros_regime_baseline`) y el detector de régimen (`fit_ros_regime_duration`) cae *enteramente* dentro del régimen 1 (ciclos 1–126, antes del primer salto en 180) — nunca ve los regímenes 2 y 3. Al llegar el ciclo 180, la regresión extrapola fuera de su dominio de ajuste y el residuo (y por lo tanto el error de reconstrucción de los autoencoders) se dispara, generando un `CRITICAL` uniforme que en realidad refleja "el motor entró a un régimen operativo nunca visto durante el ajuste de la línea base", no "el motor se está degradando". Es el mismo fenómeno diagnosticado para NASA FD002/FD004 en el Experimento 2 de la Etapa 1–4 (saltos de régimen operativo inflando el error de reconstrucción), pero aquí **no está mitigado**: la feature `cycles_since_regime_change` sí se calcula para ROS, pero la línea base de normalización de la que depende el residuo de entrada nunca vio ejemplos sanos de los regímenes 2 y 3, así que no puede generalizar a ellos.

Esto es un hallazgo metodológico honesto y accionable, no solo un dato negativo: confirma que la arquitectura de detección funciona (los autoencoders sí reaccionan fuertemente a algo real en los datos), pero expone que **la definición de "ciclos sanos" (`HEALTHY_RATIO` sobre los primeros ciclos) asume implícitamente un solo régimen operativo dominante al inicio de la vida del motor** — válida para NASA (donde el régimen del vuelo es más estable en las primeras corridas) pero no para este simulador ROS (que cambia de régimen por diseño en puntos fijos). Queda documentado como limitación conocida y como línea de trabajo futura antes de poder reportar resultados de detección ROS confiables en la tesis.

### Estado final del entorno tras esta etapa

- Los artifacts activos NASA (`regime_baseline.pkl`, `healthy_global_scaler.pkl`, `regime_duration_scaler.pkl`, `regime_detector_kmeans.pkl`, los 4 modelos `.pt`/`.pkl`, `model_metadata.json`, `score_calibration.json`, `selected_features.json`) fueron **restaurados desde el respaldo HDFS** (`nasa_train_20260816_231754` / `preprocessing_20260816_220101`), sobrescribiendo la versión ROS-adaptada que quedó activa tras `retrain_ros_models.py`. Se eliminaron `active_version.json`/`retrain_metadata.json` (describían el retrain ROS, ya no aplican al estado NASA restaurado).
- La versión ROS-adaptada (modelos con fine-tuning real + línea base ROS-específica) **no se perdió**: quedó respaldada íntegra en `hdfs://namenode:9000/user/root/nasa/artifacts/versions/ros_retrain_20260817_023602/`, disponible para retomarla si se continúa esta línea de trabajo.
- Los publishers ROS se detuvieron (`pkill -f ros_kafka_publisher.py`); `roscore` sigue corriendo en `ros-gazebo` en background por si se retoma la depuración sin tener que resolver el mismo bloqueador otra vez.
- Contenedores `kafka` y `ros-gazebo` quedaron arriba.

### Pendiente derivado de este hallazgo

Antes de reportar resultados de validación ROS en la tesis: redefinir `HEALTHY_RATIO`/`get_healthy_index()` para que la línea base sana cubra los regímenes operativos representativos (no solo los primeros N% de ciclos), o generar corridas ROS de calibración que empiecen ya en régimen estable, análogo a cómo NASA evita este problema por la naturaleza de sus trayectorias.

## Etapa 9 — Corrección de `get_healthy_index()`: baseline sano consciente de régimen

Corrección directa del hallazgo de la Etapa 8. Se modificó `get_healthy_index()` en `build_ros_gold.py`: en vez de tomar los primeros `HEALTHY_RATIO` (30%) ciclos cronológicos del motor (lo que en el simulador ROS cae enteramente antes del primer cambio de régimen en `cycle=180`), ahora se agrupa primero por régimen operativo (KMeans liviano sobre `op_settings`, propio de cada motor, sin depender de saber de antemano en qué ciclo cambia el régimen) y se toman los primeros 30% de ciclos **dentro de cada régimen detectado**. Así la línea base sana queda representada en los regímenes que el motor visita más tarde en su vida, no solo en el primero. Cambio acotado a ROS (`build_ros_gold.py`); no se tocó `run_preprocessing.py` (NASA) porque el diagnóstico de la Etapa 2 ya mostró que FD002/FD004 cambian de régimen en el 82.5% de los ciclos consecutivos — tan frecuente que el 30% inicial de cualquier motor NASA ya cubre prácticamente todos los regímenes por sí solo; el bug solo se manifiesta cuando los cambios de régimen son poco frecuentes, como en el simulador ROS (2 saltos en toda la corrida).

**Validación real**: se volvió a correr `gold` → `retrain_ros_models.py` → `evaluate` sobre los mismos datos ROS reales de la Etapa 8 (mismo bronze/silver, 12 187 filas). Resultado:

| | Antes (Etapa 8) | Después (Etapa 9) |
|---|---|---|
| Filas sanas usadas | 366 (solo régimen 1) | 375 (los 3 regímenes) |
| `val_loss` final Transformer (fine-tuning) | 0.833 | **0.445** (-47%) |
| Score LSTM en ciclo 230 (primer ciclo evaluado) | 2637 (CRITICAL, fuera de escala) | **0.49 (HEALTHY, en escala con la calibración)** |
| `first_critical_cycle` por motor | 230 / 230 / 230 (idéntico en los 3, sin importar el perfil) | 250 (medium) / 295 (mild) / 281 (aggressive) — **distinto por motor** |
| Score tras la evaluación (ciclo 230→400, motor 1) | ya en miles desde el inicio, sin tendencia visible | sube de forma gradual y monótona (0.49→12.5), consistente con acumulación real de degradación |

La caída a la mitad del `val_loss` de fine-tuning confirma que el modelo generaliza mucho mejor cuando su referencia "sana" incluye los 3 regímenes. El hecho de que el ciclo crítico ahora varíe por motor (en vez de ser el mismo para los 3 sin importar el perfil de degradación configurado) es la evidencia más directa de que el disparo dejó de estar dominado por el cambio de régimen y empezó a reflejar señal de degradación real. **El objetivo específico 4 queda con evidencia empírica real y metodológicamente defendible**, incluyendo el ciclo completo de diagnóstico → causa raíz → corrección → re-validación.

Artifacts NASA restaurados nuevamente desde HDFS tras este segundo `retrain_ros_models.py` (mismo procedimiento que la Etapa 8); la versión ROS de esta etapa quedó respaldada en `hdfs://namenode:9000/user/root/nasa/artifacts/versions/ros_retrain_20260817_025002/`.

**Limitación que queda abierta**: la señal todavía no distingue con claridad los 3 perfiles de degradación por severidad (el motor `aggressive` no fue el más temprano en llegar a CRITICAL) — puede deberse al tamaño pequeño de la corrida (~9 minutos, ~400 ciclos por motor) o a que el ruido/aleatoriedad del simulador domina sobre la diferencia entre perfiles en corridas cortas. Validar con corridas más largas queda fuera del alcance de esta sesión.

## Etapa 10 — Revisión de los scripts `.sh` (secuencia real de ejecución)

A pedido del usuario, se revisaron los 6 scripts de orquestación (`run_nasa_stage1.sh`, `run_nasa_stage2.sh`, `run_ros_publishers.sh`, `run_ros_live.sh`, `run_ros_batch.sh`, `run_ros_retrain.sh`) contra el código real (rutas de archivos, paquetes Spark, variables de entorno) y contra `README.txt`, para confirmar que la secuencia documentada efectivamente corre de punta a punta.

**Hallazgo principal — confirma y corrige de raíz el bug de la Etapa 8**: `run_ros_publishers.sh` nunca arrancaba `roscore`. Se verificó que ningún lugar del proyecto lo hace automáticamente (ni el `Dockerfile` de `ros-gazebo`, cuyo `CMD` es solo `["/bin/bash"]`, ni `docker-compose.yml`, ni ningún script) — es decir, el arreglo manual aplicado en la Etapa 8 (`docker exec -d ros-gazebo ... roscore`) nunca quedó incorporado a la automatización, y cualquier corrida futura de `run_ros_publishers.sh` habría vuelto a reproducir el mismo bug silencioso (ráfaga sin límite de tasa, luego publishers mudos). **Corregido**: `run_ros_publishers.sh` ahora arranca `roscore` de forma idempotente (verifica con `pgrep` si ya está corriendo antes de lanzarlo) antes de lanzar los motores, y cada motor exporta `ROS_MASTER_URI`/`ROS_HOSTNAME` explícitamente. Validado en vivo: 1 motor de prueba publicó a tasa estable (~10 mensajes/s reales, verificado con el offset de Kafka), sin ráfaga.

**Segundo hallazgo, documentado (no modificado — es una decisión de diseño, no un bug evidente)**: `ingest_ros_kafka_to_bronze.py` usa `startingOffsets=latest` por defecto. Esto significa que si los publishers ROS ya produjeron mensajes antes de que el job de streaming se suscriba al tópico, esos mensajes **se pierden en silencio** (sin error) — el consumer solo ve lo que se publica después de arrancar. Esto explica en retrospectiva el "stall tras 1 micro-batch" observado en la Etapa 7: probablemente el streaming se arrancó después de que los publishers ya habían emitido su ráfaga inicial (o, con el bug de roscore ya corregido, después de que emitieran sus primeros mensajes), y ese único micro-batch fue todo lo que alcanzó a capturar antes de quedar esperando mensajes nuevos que llegaban a un ritmo mucho menor al esperado. **El orden correcto es: arrancar `run_ros_live.sh bronze` ANTES que `run_ros_publishers.sh`**, no al revés.

**Tercer hallazgo, menor**: `README.txt` documentaba un orden de ejecución incorrecto (`run_ros_batch.sh all` listado antes de `run_ros_live.sh bronze`, cuando bronze debe existir primero) y no mencionaba `run_ros_publishers.sh` en absoluto (sin publishers no hay datos Kafka que consumir). **Corregido**: se reescribió `README.txt` con el orden real y explícito, incluyendo la advertencia sobre `startingOffsets=latest`.

**Hallazgo informativo, sin cambio de código**: `run_ros_retrain.sh` invoca `retrain_ros_models.py` con `python3` directo (no `spark-submit --master spark://spark-master:7077`), a diferencia del resto de los jobs Spark del proyecto. Esto es funcionalmente correcto (`build_spark()` no fija un master, por lo que cae a `local[*]` dentro del contenedor `spark-master`), pero no aprovecha los recursos dedicados de `spark-worker`. Para los volúmenes de datos ROS actuales (miles de filas) es irrelevante en la práctica; se deja documentado por si en el futuro los volúmenes crecen.

**Resto de los scripts revisados sin problemas**: todas las rutas de archivos referenciadas existen, los paquetes Spark (`--packages` para Postgres/Kafka) están presentes donde se necesitan, y las variables de entorno (`ARTIFACTS_DIR`, rutas HDFS) son consistentes entre scripts.

## Etapa 11 — Grafana y Jupyter levantados; tablas NASA recreadas; dashboards reales publicados

Los contenedores `grafana` y `jupyter` estaban detenidos desde hacía 2 semanas. Se levantaron ambos (`docker start grafana jupyter`). Jupyter corre sin token (`http://localhost:8888/lab`). Grafana (`http://localhost:3000`, `admin`/`admin123`) ya tenía un datasource de PostgreSQL configurado de una sesión previa (`grafana-postgresql-datasource`, apuntando a `postgres:5432/tesis`), verificado con conexión OK — pero **sin ningún dashboard creado**.

**Tablas NASA de detalle, recreadas**: `eval_interna_predictions` y `eval_externa_predictions` (borradas en la Etapa 6 por schema obsoleto) no se habían vuelto a poblar porque `train_and_evaluate.py`/`external_evaluation.py` no se habían vuelto a correr desde entonces — solo existían las tablas de métricas agregadas. Se corrió `run_nasa_stage2.sh all` (train + external) de nuevo. La ejecución tardó considerablemente más de lo habitual (~45 min vs. ~15 min de corridas anteriores en esta sesión) y el proceso quedó corriendo huérfano en segundo plano tras un timeout del lado del cliente (el contenedor `spark-master` tiene `mem_limit: 1g` y llegó a ~97% de uso sostenido durante el entrenamiento) — se dejó terminar en vez de reintentar, y terminó exitosamente sin necesidad de reiniciar nada. Resultado: métricas idénticas a las ya documentadas (IF spearman=0.822, LSTM=0.798, Transformer=0.695, TCN=0.693 — reproducibilidad confirmada), con `eval_interna_predictions`=139 798 filas y `eval_externa_predictions`=84 478 filas de detalle real por ventana/ciclo. Los artifacts NASA activos quedaron en la nueva corrida (`nasa_train_20260817_032109`) — válida y coherente, no requiere restauración (a diferencia de después de un retrain ROS).

**Dashboards reales publicados vía API de Grafana** (verificados con datos reales devueltos, no solo creados):
- `ROS - Detección de Anomalías (datos reales)` (`/d/ros-anomaly-real`): 4 paneles de series de tiempo (score LSTM/TCN/Transformer/IF por motor, eje x=ciclo) sobre la corrida corregida de la Etapa 9 (`ros_eval_20260817_025044`), más una tabla resumen de `first_warning_cycle`/`first_critical_cycle`/`final_status_last` por motor.
- `NASA CMAPSS - Comparación de Modelos` (`/d/nasa-model-comparison`): barras de spearman y monotonicidad por modelo (última corrida, vía subconsulta dinámica `ORDER BY run_id DESC LIMIT 1` — no requiere edición manual en corridas futuras), tabla de detalle de todas las corridas internas, y tabla de evaluación externa.
