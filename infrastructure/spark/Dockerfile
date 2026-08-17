FROM apache/spark:3.5.0

USER root

RUN if ! id -u spark >/dev/null 2>&1; then useradd -u 1001 -m spark; fi

ENV HOME=/home/spark
ENV IVY_HOME=/tmp/.ivy2
ENV SPARK_LOCAL_DIRS=/tmp/spark-temp

RUN mkdir -p /tmp/.ivy2/cache /tmp/.ivy2/local /tmp/spark-temp /opt/spark/work \
    && chmod -R 777 /tmp /opt/spark/work

RUN python3 -m pip install --no-cache-dir \
    numpy pandas pyarrow scikit-learn matplotlib seaborn

RUN python3 -m pip install --no-cache-dir "typing_extensions<4.13"

RUN python3 -m pip install --no-cache-dir \
    pyspark==3.5.0 \
    py4j==0.10.9.7

RUN python3 -m pip install --no-cache-dir \
    torch==2.4.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN python3 -m pip install --no-cache-dir kafka-python

USER spark