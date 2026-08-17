Scripts incluidos:
- run_nasa_stage1.sh     -> ingesta NASA + silver + nasa_train_gold + artifacts de preprocessing
- run_nasa_stage2.sh     -> train/interna y externa NASA
- run_ros_publishers.sh  -> arranca roscore (si no está activo) + N motores ROS publicando a Kafka
- run_ros_live.sh        -> bronze live (Kafka->HDFS), inference live, alerts live
- run_ros_batch.sh       -> silver/gold/evaluate ROS offline (requiere bronze ya poblado)
- run_ros_retrain.sh     -> reentrenamiento ROS con guardado local + HDFS

Orden real de ejecución (NASA primero, ROS después):

1) NASA:
   bash run_nasa_stage1.sh
   bash run_nasa_stage2.sh all

2) ROS — IMPORTANTE: arrancar bronze (streaming) ANTES que los publishers.
   ingest_ros_kafka_to_bronze.py usa startingOffsets=latest por defecto: si los
   publishers ya produjeron mensajes antes de que el consumer se suscriba, esos
   mensajes se pierden en silencio (no hay error, el stream simplemente no los ve).
   bash run_ros_live.sh bronze &          # dejar corriendo en background/otra terminal
   bash run_ros_publishers.sh 3 medium 10 # arranca roscore si hace falta + 3 motores
   bash run_ros_batch.sh silver
   bash run_ros_batch.sh gold
   bash run_ros_retrain.sh
   bash run_ros_batch.sh evaluate
   bash run_ros_live.sh inference
   bash run_ros_live.sh alerts

Para detener los publishers ROS:
   docker exec ros-gazebo bash -lc "pkill -f ros_kafka_publisher.py || true"