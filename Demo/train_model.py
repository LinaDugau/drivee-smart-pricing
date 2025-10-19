#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import inspect


MODEL_OUT = "predictions.joblib"
RANDOM_STATE = 42


def parse_args():
    ap = argparse.ArgumentParser(description="Train regression model to predict price_bid_local")
    ap.add_argument("--csv", type=str, default="train.csv", help="Path to CSV (new schema)")
    ap.add_argument("--model-out", type=str, default=MODEL_OUT, help="Where to save model artifact")
    return ap.parse_args()


def _parse_dt(s):
    """Безопасное преобразование даты/времени в pandas.Timestamp (без deprecated аргументов)."""
    return pd.to_datetime(s, errors="coerce", utc=False)


def load_and_featurize(path: str) -> pd.DataFrame:
    """
    Загружает CSV и строит признаки БЕЗ утечки таргета.
    Target: price_bid_local
    """
    df = pd.read_csv(path)

    required = [
        "order_timestamp", "tender_timestamp",
        "price_start_local", "price_bid_local",
        "distance_in_meters"
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Нет обязательной колонки: {col}")

    # Базовая чистка
    df["price_start_local"] = pd.to_numeric(df["price_start_local"], errors="coerce")
    df["price_bid_local"]   = pd.to_numeric(df["price_bid_local"], errors="coerce")
    df["distance_in_meters"]= pd.to_numeric(df["distance_in_meters"], errors="coerce")

    # Время
    df["order_timestamp"]  = _parse_dt(df["order_timestamp"])
    df["tender_timestamp"] = _parse_dt(df["tender_timestamp"])

    # Доп. (если есть)
    for c in ["duration_in_seconds","pickup_in_meters","pickup_in_seconds","driver_rating"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    if "driver_reg_date" in df.columns:
        df["driver_reg_date"] = _parse_dt(df["driver_reg_date"])
    else:
        df["driver_reg_date"] = pd.NaT

    # Удалим мусор
    df = df.dropna(subset=["order_timestamp","tender_timestamp","price_start_local","price_bid_local","distance_in_meters"])
    df = df[(df["price_start_local"]>0)&(df["price_bid_local"]>0)&(df["distance_in_meters"]>=0)].copy()

    # ===== Время
    df["order_hour"]  = df["order_timestamp"].dt.hour
    df["order_dow"]   = df["order_timestamp"].dt.dayofweek
    df["tender_hour"] = df["tender_timestamp"].dt.hour
    df["tender_dow"]  = df["tender_timestamp"].dt.dayofweek

    ang = 2*math.pi/24.0
    df["order_hour_sin"]  = np.sin(ang*df["order_hour"])
    df["order_hour_cos"]  = np.cos(ang*df["order_hour"])
    df["tender_hour_sin"] = np.sin(ang*df["tender_hour"])
    df["tender_hour_cos"] = np.cos(ang*df["tender_hour"])

    # Задержка бида
    df["bid_delay_min"] = (df["tender_timestamp"] - df["order_timestamp"]).dt.total_seconds()/60.0
    df["bid_delay_min"] = df["bid_delay_min"].clip(lower=0)

    # ===== Дистанции / скорости
    df["distance_in_meters"] = df["distance_in_meters"].clip(lower=0)
    df["log_distance_m"]     = np.log(df["distance_in_meters"].clip(lower=1.0))

    df["duration_in_seconds"] = df["duration_in_seconds"].clip(lower=0)
    df["avg_speed_kmh"] = np.where(
        df["duration_in_seconds"]>0,
        (df["distance_in_meters"]/df["duration_in_seconds"])*3.6,
        np.nan
    )
    df["avg_speed_kmh"] = df["avg_speed_kmh"].clip(lower=0, upper=150)

    df["pickup_in_meters"]  = df["pickup_in_meters"].clip(lower=0)
    df["pickup_in_seconds"] = df["pickup_in_seconds"].clip(lower=0)
    df["pickup_speed_kmh"] = np.where(
        df["pickup_in_seconds"]>0,
        (df["pickup_in_meters"]/df["pickup_in_seconds"])*3.6,
        np.nan
    )
    df["pickup_speed_kmh"] = df["pickup_speed_kmh"].clip(lower=0, upper=100)

    # Опыт водителя
    df["driver_experience_days"] = np.where(
        df["driver_reg_date"].notna(),
        (df["order_timestamp"] - df["driver_reg_date"]).dt.days,
        np.nan
    )
    df["driver_experience_days"] = df["driver_experience_days"].clip(lower=0)

    # Лог-цены пассажира
    df["log_price_start"] = np.log(df["price_start_local"].clip(lower=1.0))

    # Категориальные (если нет — подставим)
    for cat in ["platform","carmodel","carname"]:
        if cat not in df.columns:
            df[cat] = "unknown"
        df[cat] = df[cat].astype("string")

    return df


def build_pipeline(numeric_features, categorical_features):
    """Dense-пайплайн: OHE без sparse, чтобы совместимо с HistGBR."""
    num_transform = StandardScaler(with_mean=True, with_std=True)

    # Универсально под любую версию sklearn:
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        cat_transform = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        cat_transform = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre = ColumnTransformer(
        transformers=[
            ("num", num_transform, numeric_features),
            ("cat", cat_transform, categorical_features),
        ],
        remainder="drop",
        # возвращаем DENSE (т.к. и num, и cat — dense)
        sparse_threshold=1.0,
    )

    reg = HistGradientBoostingRegressor(
        max_depth=None,
        learning_rate=0.08,
        max_iter=400,
        l2_regularization=0.0,
        random_state=RANDOM_STATE,
    )

    pipe = Pipeline([
        ("pre", pre),
        ("reg", reg),
    ])
    return pipe


def main():
    args = parse_args()
    df = load_and_featurize(args.csv)

    # Базовые «желательные» списки
    numeric_wanted = [
        "price_start_local","log_price_start",
        "order_hour","order_dow","tender_hour","tender_dow",
        "order_hour_sin","order_hour_cos","tender_hour_sin","tender_hour_cos",
        "bid_delay_min","distance_in_meters","log_distance_m",
        "duration_in_seconds","avg_speed_kmh",
        "pickup_in_meters","pickup_in_seconds","pickup_speed_kmh",
        "driver_rating","driver_experience_days",
    ]
    categorical_wanted = ["platform","carmodel","carname"]

    # Если каких-то колонок нет — создаём безопасные дефолты
    defaults_num = {
        "duration_in_seconds": np.nan,
        "avg_speed_kmh": np.nan,
        "pickup_in_meters": np.nan,
        "pickup_in_seconds": np.nan,
        "pickup_speed_kmh": np.nan,
        "driver_rating": np.nan,
        "driver_experience_days": np.nan,
    }
    for col, val in defaults_num.items():
        if col not in df.columns:
            df[col] = val

    for col in categorical_wanted:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].astype("string")

    # Берём только те признаки, которые реально присутствуют
    numeric_features = [c for c in numeric_wanted if c in df.columns]
    categorical_features = [c for c in categorical_wanted if c in df.columns]

    feature_cols = numeric_features + categorical_features

    feature_cols = numeric_features + categorical_features
    target = "price_bid_local"

    X_train, X_valid, y_train, y_valid = train_test_split(
        df[feature_cols], df[target],
        test_size=0.2, random_state=RANDOM_STATE
    )

    pipe = build_pipeline(numeric_features, categorical_features)
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_valid)
    r2   = r2_score(y_valid, pred)
    mae  = mean_absolute_error(y_valid, pred)
    rmse = math.sqrt(mean_squared_error(y_valid, pred))

    print(f"[VAL] R² = {r2:.3f}   MAE = {mae:.2f} ₽   RMSE = {rmse:.2f} ₽")
    print(f"[VAL] n_train={len(X_train)}, n_valid={len(X_valid)}")

    meta = {
        "target": target,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "r2": r2, "mae": mae, "rmse": rmse,
        "csv_used": args.csv,
        "random_state": RANDOM_STATE,
        "note": "Dense OHE for HistGBR; RMSE via sqrt(MSE).",
    }
    artifact = {
        "pipeline": pipe,
        "feature_cols": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "meta": meta,
    }
    joblib.dump(artifact, args.model_out)
    print(f"[OK] Model saved to {args.model_out}")

     # === Финальный predict на test.csv ===
    TEST_PATH = "test.csv"
    OUTPUT_PATH = "XX_predict.csv"  

    try:
        df_test = load_and_featurize(TEST_PATH)

        # Используем те же признаки, что и при обучении
        X_test = df_test[feature_cols]

        # Предсказываем целевую переменную
        y_pred = pipe.predict(X_test)

        # Если нужно сделать предсказание 0/1 (например, принятие/отмена),
        # округляем к ближайшему целому:
        y_pred = np.round(y_pred).astype(int)

        # Сохраняем в CSV с одним столбцом is_done
        pd.DataFrame({"is_done": y_pred}).to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
        print(f"[OK] Predict saved to {OUTPUT_PATH} (rows={len(y_pred)})")

    except FileNotFoundError:
        print(f"[WARN] {TEST_PATH} not found, skipping prediction step.")
    except Exception as e:
        print(f"[ERROR] Failed to create predict file: {e}")


if __name__ == "__main__":
    main()