from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/src/models")
import importlib.util

spec = importlib.util.spec_from_file_location("tae", "/src/models/train_and_evaluate.py")
tae = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tae)

import joblib
import numpy as np
import pandas as pd
import torch
from pyspark.sql import SparkSession

ARTIFACTS_DIR = "/data/models/artifacts"
HDFS_GOLD_TRAIN = "hdfs://namenode:9000/user/root/nasa/gold/nasa_train_gold"

spark = (
    SparkSession.builder.appName("ScoreOnly4Models")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
gold = spark.read.parquet(HDFS_GOLD_TRAIN).toPandas()
spark.stop()

sensor_cols = tae.choose_sensor_cols(gold, ARTIFACTS_DIR)
model_input_cols = sensor_cols + [tae.DURATION_COL]

gold = gold.copy()
gold["dataset_id"] = gold["dataset_id"].astype(str)
gold["engine_id"] = gold["engine_id"].astype(str)
gold["engine_num"] = pd.to_numeric(gold["engine_num"], errors="coerce")
gold["cycle"] = pd.to_numeric(gold["cycle"], errors="coerce")
gold["is_healthy_train"] = pd.to_numeric(gold["is_healthy_train"], errors="coerce").fillna(0).astype(int)
for c in model_input_cols:
    gold[c] = pd.to_numeric(gold[c], errors="coerce")
gold = gold.dropna(subset=["engine_num", "cycle"] + model_input_cols).copy()
gold["engine_num"] = gold["engine_num"].astype(int)
gold["cycle"] = gold["cycle"].astype(int)
gold = gold.sort_values(["dataset_id", "engine_num", "cycle"]).reset_index(drop=True)

X_test, meta_test = tae.build_windows(gold, model_input_cols, split_filter="test", require_healthy=False)
X_train_healthy, _ = tae.build_windows(gold, model_input_cols, split_filter="train", require_healthy=True)
print(f"X_test={X_test.shape} X_train_healthy={X_train_healthy.shape}")

n_features = len(model_input_cols)
DEVICE = "cpu"

if_model = joblib.load(os.path.join(ARTIFACTS_DIR, "isolation_forest.pkl"))

lstm = tae.LSTMAutoencoder(n_features=n_features)
lstm.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "lstm_autoencoder.pt"), map_location=DEVICE))
lstm.eval()

tcn = tae.TCNAutoencoder(n_features=n_features)
tcn.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "tcn_autoencoder.pt"), map_location=DEVICE))
tcn.eval()

transformer = tae.TransformerAutoencoder(n_features=n_features)
transformer.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "transformer_autoencoder.pt"), map_location=DEVICE))
transformer.eval()

X_test_flat = X_test.reshape(len(X_test), -1)
X_healthy_ref_flat = X_train_healthy.reshape(len(X_train_healthy), -1)

if_test_raw = -if_model.decision_function(X_test_flat)
if_healthy_ref_raw = -if_model.decision_function(X_healthy_ref_flat)
if_test_score = tae.scale_with_reference(if_healthy_ref_raw, if_test_raw)

lstm_test_raw = tae.infer_ae_scores(lstm, X_test)
lstm_healthy_ref_raw = tae.infer_ae_scores(lstm, X_train_healthy)
lstm_test_score = tae.scale_with_reference(lstm_healthy_ref_raw, lstm_test_raw)

tcn_test_raw = tae.infer_ae_scores(tcn, X_test)
tcn_healthy_ref_raw = tae.infer_ae_scores(tcn, X_train_healthy)
tcn_test_score = tae.scale_with_reference(tcn_healthy_ref_raw, tcn_test_raw)

transformer_test_raw = tae.infer_ae_scores(transformer, X_test)
transformer_healthy_ref_raw = tae.infer_ae_scores(transformer, X_train_healthy)
transformer_test_score = tae.scale_with_reference(transformer_healthy_ref_raw, transformer_test_raw)

test_pred = meta_test.copy()
test_pred["if_score"] = if_test_score
test_pred["lstm_score"] = lstm_test_score
test_pred["tcn_score"] = tcn_test_score
test_pred["transformer_score"] = transformer_test_score

metrics_rows = []
for model_name, score_col in [
    ("IsolationForest", "if_score"),
    ("LSTM_AE", "lstm_score"),
    ("TCN_AE", "tcn_score"),
    ("Transformer_AE", "transformer_score"),
]:
    row = {"model": model_name}
    row.update(tae.group_internal_metrics(test_pred[["engine_id", "cycle", "true_rul", score_col]].copy(), score_col))
    metrics_rows.append(row)

metrics_df = pd.DataFrame(metrics_rows)
print("\n=== Métricas internas (4 modelos, desde artifacts ya guardados) ===")
print(metrics_df.to_string(index=False))

with open(os.path.join(ARTIFACTS_DIR, "model_metadata.json")) as f:
    meta = json.load(f)
print("\n=== val_loss de referencia (model_metadata.json) ===")
print(f"lstm_best_val_loss={meta.get('lstm_best_val_loss')}")
print(f"tcn_best_val_loss={meta.get('tcn_best_val_loss')}")
print(f"transformer_best_val_loss={meta.get('transformer_best_val_loss')}")
