from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lit,
    concat_ws,
    split,
    regexp_replace,
    trim,
    size,
)
from pyspark.sql.types import DoubleType, IntegerType

# =========================================================
# CONFIG
# =========================================================
HDFS_BASE = "hdfs://namenode:9000/user/root/nasa"
HDFS_BRONZE_NASA = f"{HDFS_BASE}/bronze/nasa_raw"
HDFS_SILVER = f"{HDFS_BASE}/silver"

BASE_COLUMNS = (
    ["engine_num", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

EXPECTED_FIELD_COUNT = len(BASE_COLUMNS)


def build_spark(app_name: str) -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def leer_txt_nasa(spark: SparkSession, ruta_hdfs: str, dataset_id: str):
    df = spark.read.text(ruta_hdfs)
    df = df.withColumn("value", trim(col("value")))
    df = df.filter(col("value") != "")
    df = df.withColumn("value", regexp_replace(col("value"), r"\s+", ","))
    partes = split(col("value"), ",")

    df = df.filter(size(partes) >= EXPECTED_FIELD_COUNT)

    for i, nombre_col in enumerate(BASE_COLUMNS):
        df = df.withColumn(nombre_col, partes.getItem(i))

    df = df.drop("value")

    df = df.withColumn("engine_num", col("engine_num").cast(IntegerType()))
    df = df.withColumn("cycle", col("cycle").cast(IntegerType()))

    for c in [f"op_setting_{i}" for i in range(1, 4)]:
        df = df.withColumn(c, col(c).cast(DoubleType()))

    for c in [f"sensor_{i}" for i in range(1, 22)]:
        df = df.withColumn(c, col(c).cast(DoubleType()))

    df = df.dropna().dropDuplicates()
    df = df.withColumn("dataset_id", lit(dataset_id))
    df = df.withColumn("engine_id", concat_ws("_", col("dataset_id"), col("engine_num")))

    return df.select(
        "dataset_id",
        "engine_id",
        "engine_num",
        "cycle",
        "op_setting_1",
        "op_setting_2",
        "op_setting_3",
        *[f"sensor_{i}" for i in range(1, 22)],
    )


def leer_rul(spark: SparkSession, ruta_hdfs: str, dataset_id: str):
    rdd = (
        spark.sparkContext.textFile(ruta_hdfs)
        .map(lambda x: x.strip())
        .filter(lambda x: x != "")
    )

    rdd_idx = rdd.zipWithIndex().map(
        lambda x: (
            dataset_id,
            f"{dataset_id}_{int(x[1]) + 1}",
            int(x[1]) + 1,
            int(float(x[0])),
        )
    )

    return (
        spark.createDataFrame(rdd_idx, ["dataset_id", "engine_id", "engine_num", "rul"])
        .orderBy("dataset_id", "engine_num")
    )


def guardar_parquet(df, ruta_salida: str, descripcion: str) -> None:
    print(f"\n===== GUARDANDO {descripcion} =====")
    print(f"Ruta: {ruta_salida}")
    df.write.mode("overwrite").parquet(ruta_salida)
    print("✅ Guardado correcto")


def main() -> None:
    spark = build_spark("ProcessNASAFromBronzeToSilver")

    train_dfs = []
    for ds in ["FD001", "FD002", "FD003", "FD004"]:
        ruta = f"{HDFS_BRONZE_NASA}/train_{ds}.txt"
        df_train = leer_txt_nasa(spark, ruta, ds)
        train_dfs.append(df_train)

    df_train_all = train_dfs[0]
    for df_tmp in train_dfs[1:]:
        df_train_all = df_train_all.unionByName(df_tmp)

    df_train_all = df_train_all.orderBy("dataset_id", "engine_num", "cycle")
    guardar_parquet(df_train_all, f"{HDFS_SILVER}/train_all", "NASA TRAIN_ALL limpio")

    for ds in ["FD001", "FD002", "FD003", "FD004"]:
        ruta_test = f"{HDFS_BRONZE_NASA}/test_{ds}.txt"
        df_test = leer_txt_nasa(spark, ruta_test, ds).orderBy("dataset_id", "engine_num", "cycle")
        guardar_parquet(df_test, f"{HDFS_SILVER}/test_{ds}", f"NASA TEST {ds} limpio")

        ruta_rul = f"{HDFS_BRONZE_NASA}/RUL_{ds}.txt"
        df_rul = leer_rul(spark, ruta_rul, ds)
        guardar_parquet(df_rul, f"{HDFS_SILVER}/rul_{ds}", f"NASA RUL {ds} limpio")

    print("\n✅ NASA bronze -> silver completado")
    spark.stop()


if __name__ == "__main__":
    main()
