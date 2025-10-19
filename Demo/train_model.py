#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_model.py — обучает модель с учётом времени суток и дистанции.

Ожидаемые колонки в train.csv:
- price_start_local (float/int)
- price_bid_local   (float/int)
- is_done           ('done' / 'cancel' или 1/0)
- order_timestamp   (строка, парсится pandas.to_datetime)
- tender_timestamp  (строка, парсится pandas.to_datetime)
- distance_in_meters (float/int) — дистанция маршрута от A до B

Сохраняет: model.joblib (dict: {'pipeline': Pipeline, 'feature_cols': [...]})
"""

import os, sys, numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

CSV_PATH = "train.csv"
MODEL_PATH = "model.joblib"

def _coerce_is_done(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower()
    mapping = {"done":1, "cancel":0, "canceled":0, "cancelled":0, "1":1, "0":0, "true":1, "false":0}
    out = s.map(mapping)
    if out.isna().any():
        bad = s[out.isna()].unique()[:10]
        raise ValueError(f"Не удалось распарсить is_done для: {bad}")
    return out.astype(int)

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[ERROR] Нет {path}", file=sys.stderr); sys.exit(1)
    df = pd.read_csv(path)

    need = [
        "price_start_local","price_bid_local","is_done",
        "order_timestamp","tender_timestamp","distance_in_meters"
    ]
    for c in need:
        if c not in df.columns:
            print(f"[ERROR] В train.csv нет колонки {c}", file=sys.stderr); sys.exit(1)

    df = df.dropna(subset=need)
    df = df[(df["price_start_local"]>0)&(df["price_bid_local"]>0)].copy()
    df["is_done"] = _coerce_is_done(df["is_done"])

    # время
    df["order_timestamp"]  = pd.to_datetime(df["order_timestamp"],  errors="coerce")
    df["tender_timestamp"] = pd.to_datetime(df["tender_timestamp"], errors="coerce")
    df = df.dropna(subset=["order_timestamp","tender_timestamp"])

    delay = (df["tender_timestamp"] - df["order_timestamp"]).dt.total_seconds().clip(lower=0)
    df["bid_delay_min"] = (delay/60.0).astype(float)

    df["order_hour"]  = df["order_timestamp"].dt.hour
    df["order_dow"]   = df["order_timestamp"].dt.dayofweek
    df["tender_hour"] = df["tender_timestamp"].dt.hour
    df["tender_dow"]  = df["tender_timestamp"].dt.dayofweek

    # циклические часы
    ang = 2*np.pi/24.0
    df["order_hour_sin"]  = np.sin(ang*df["order_hour"])
    df["order_hour_cos"]  = np.cos(ang*df["order_hour"])
    df["tender_hour_sin"] = np.sin(ang*df["tender_hour"])
    df["tender_hour_cos"] = np.cos(ang*df["tender_hour"])

    # ценовые
    df["bid_ratio"]  = df["price_bid_local"] / df["price_start_local"]
    df["log_ratio"]  = np.log(df["bid_ratio"].clip(lower=1e-12))
    df["uplift_rel"] = df["bid_ratio"] - 1.0
    df["uplift_abs"] = df["price_bid_local"] - df["price_start_local"]

    # дистанция
    df["distance_in_meters"] = pd.to_numeric(df["distance_in_meters"], errors="coerce")
    df["distance_in_meters"] = df["distance_in_meters"].fillna(df["distance_in_meters"].median()).clip(lower=0)
    df["log_distance_m"] = np.log(df["distance_in_meters"].clip(lower=1.0))

    return df

def train_and_save(df: pd.DataFrame):
    feature_cols = [
        # price features
        "price_start_local","price_bid_local","bid_ratio","log_ratio","uplift_rel","uplift_abs",
        # time features
        "order_hour","order_dow","tender_hour","tender_dow","bid_delay_min",
        "order_hour_sin","order_hour_cos","tender_hour_sin","tender_hour_cos",
        # distance features
        "distance_in_meters","log_distance_m",
    ]
    X = df[feature_cols].astype(float)
    y = df["is_done"].astype(int)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pre = ColumnTransformer([("num", StandardScaler(), X.columns)], remainder="drop")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_tr, y_tr)

    proba = pipe.predict_proba(X_va)[:,1]
    print(f"[INFO] AUC={roc_auc_score(y_va,proba):.3f}  Brier={brier_score_loss(y_va,proba):.3f}  LogLoss={log_loss(y_va,proba):.3f}")

    joblib.dump({"pipeline":pipe, "feature_cols":feature_cols}, MODEL_PATH)
    print(f"[OK] Сохранено: {MODEL_PATH} (фичей: {len(feature_cols)})")

if __name__=="__main__":
    df = load_data(CSV_PATH)
    train_and_save(df)