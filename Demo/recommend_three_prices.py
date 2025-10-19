#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recommend_three_prices.py — 4 цены с учётом времени (HH:MM) и дистанции (м):
1) Лучшая (макс. ожидаемый доход)
2) 90% от P(лучшей), справа
3) 60% от P(лучшей), справа
4) 20% от P(лучшей), справа
+ опционально: расчёт для своей цены (≥ стартовой)

Все цены округляются до ближайших 5 ₽.
"""

import os, sys, numpy as np, pandas as pd, joblib
from datetime import datetime, date

MODEL_PATH = "model.joblib"

FEATURE_COLS = [
    "price_start_local","price_bid_local","bid_ratio","log_ratio","uplift_rel","uplift_abs",
    "order_hour","order_dow","tender_hour","tender_dow","bid_delay_min",
    "order_hour_sin","order_hour_cos","tender_hour_sin","tender_hour_cos",
    "distance_in_meters","log_distance_m",
]

# === загрузка модели ===
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        print(f"[ERROR] Нет {path}. Сначала обучи модель: python train_model.py", file=sys.stderr); sys.exit(1)
    obj = joblib.load(path)
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj["pipeline"], obj.get("feature_cols", FEATURE_COLS)
    try: _ = obj.steps
    except Exception: print("[ERROR] model.joblib не Pipeline.", file=sys.stderr); sys.exit(1)
    return obj, FEATURE_COLS

# === парсинг времени HH:MM ===
def parse_hhmm(s: str) -> datetime:
    s = s.strip()
    for fmt in ["%H:%M","%H:%M:%S"]:
        try:
            t = datetime.strptime(s, fmt).time()
            return datetime.combine(date.today(), t)
        except Exception:
            pass
    raise ValueError("Введите время в формате HH:MM")

# === построение строки признаков ===
def make_row(start, bid, order_dt, tender_dt, distance_m):
    order_hour, tender_hour = order_dt.hour, tender_dt.hour
    order_dow = tender_dow = 0  # без даты фиксируем 0
    delay_min = max(0.0, (tender_dt - order_dt).total_seconds()/60.0)
    ang = 2*np.pi/24.0

    bid_ratio  = bid / start
    log_ratio  = np.log(max(bid_ratio, 1e-12))
    uplift_rel = bid_ratio - 1.0
    uplift_abs = bid - start

    distance_m = float(distance_m)
    log_distance_m = np.log(max(distance_m, 1.0))

    data = {
        "price_start_local": start,
        "price_bid_local": bid,
        "bid_ratio": bid_ratio,
        "log_ratio": log_ratio,
        "uplift_rel": uplift_rel,
        "uplift_abs": uplift_abs,
        "order_hour": order_hour,
        "order_dow": order_dow,
        "tender_hour": tender_hour,
        "tender_dow": tender_dow,
        "bid_delay_min": delay_min,
        "order_hour_sin": np.sin(ang*order_hour),
        "order_hour_cos": np.cos(ang*order_hour),
        "tender_hour_sin": np.sin(ang*tender_hour),
        "tender_hour_cos": np.cos(ang*tender_hour),
        "distance_in_meters": distance_m,
        "log_distance_m": log_distance_m,
    }
    return pd.DataFrame([[data[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)

def predict(pipe, start, bid, order_dt, tender_dt, distance_m) -> float:
    X = make_row(start, bid, order_dt, tender_dt, distance_m)
    return float(pipe.predict_proba(X)[:,1][0])

def build_curve(pipe, start, order_dt, tender_dt, distance_m, up_pct=0.60):
    prices = np.arange(int(round(start)), int(round(start*(1+up_pct)))+1, 1, dtype=int)
    probs = [predict(pipe, start, p, order_dt, tender_dt, distance_m) for p in prices]
    df = pd.DataFrame({"price":prices, "p_accept":probs})
    df["expected_revenue"] = df["price"]*df["p_accept"]
    return df

def round_price_5(x: float) -> int:
    return int(round(x/5)*5)

def pick_right(curve, idx_base, target_prob):
    """Первая точка справа от оптимума, где P <= target_prob. Если нет — ближайшая справа по |P-target|."""
    right = curve.iloc[idx_base+1:]
    if right.empty:
        return curve.iloc[-1]
    cand = right[right["p_accept"] <= target_prob]
    if len(cand):
        return cand.iloc[0]
    return right.loc[(right["p_accept"]-target_prob).abs().idxmin()]

# === main ===
if __name__=="__main__":
    pipe, feat = load_model()

    # вход
    try:
        start = float(input("Стартовая цена, ₽: ").strip().replace(",", "."))
        if start <= 0: raise ValueError
    except Exception:
        print("[ERROR] Неверная стартовая цена.", file=sys.stderr); sys.exit(1)

    try:
        order_s  = input("Время заказа (HH:MM): ").strip()
        tender_s = input("Время бида (ENTER = как у заказа): ").strip()
        order_dt  = parse_hhmm(order_s)
        tender_dt = parse_hhmm(tender_s) if tender_s else order_dt
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr); sys.exit(1)

    try:
        dist = float(input("Дистанция маршрута, м: ").strip().replace(",", "."))
        if dist < 0: raise ValueError
    except Exception:
        print("[ERROR] Дистанция должна быть неотрицательным числом.", file=sys.stderr); sys.exit(1)

    # кривая
    curve = build_curve(pipe, start, order_dt, tender_dt, dist, up_pct=0.60)

    # лучшая по ожиданию
    idx_opt = int(curve["expected_revenue"].idxmax())
    best = curve.loc[idx_opt]
    p_opt_raw = float(best["p_accept"])
    price_opt = round_price_5(best["price"])
    p_opt = predict(pipe, start, price_opt, order_dt, tender_dt, dist)
    e_opt = price_opt * p_opt

    # цели 90/60/20% от лучшей, справа
    target90, target60, target20 = 0.90*p_opt_raw, 0.60*p_opt_raw, 0.20*p_opt_raw
    pt90 = pick_right(curve, idx_opt, target90)
    pt60 = pick_right(curve, idx_opt, target60)
    pt20 = pick_right(curve, idx_opt, target20)

    # округлим цены до 5 ₽ и пересчитаем метрики с учётом округления
    def recompute(pt):
        price = round_price_5(pt["price"])
        pval  = predict(pipe, start, price, order_dt, tender_dt, dist)
        return price, pval, price*pval

    price_90, p90v, e90 = recompute(pt90)
    price_60, p60v, e60 = recompute(pt60)
    price_20, p20v, e20 = recompute(pt20)

    # вывод
    print("\nРекомендованные цены (учёт времени HH:MM и дистанции, цены округлены до 5 ₽):")
    print("-"*90)
    print(f"1) ЛУЧШАЯ (макс. E):          {price_opt:>4} ₽ | P≈{p_opt:.3f}  | E≈{e_opt:.1f} ₽")
    print(f"2) 90% от P(лучшей), справа: {price_90:>4} ₽ | P≈{p90v:.3f} | E≈{e90:.1f} ₽")
    print(f"3) 60% от P(лучшей), справа: {price_60:>4} ₽ | P≈{p60v:.3f} | E≈{e60:.1f} ₽")
    print(f"4) 20% от P(лучшей), справа: {price_20:>4} ₽ | P≈{p20v:.3f} | E≈{e20:.1f} ₽")
    print("-"*90)

    # своя цена
    if input("\nХотите ввести свою цену? (y/n): ").strip().lower() == "y":
        try:
            bid = float(input("Ваша цена (≥ стартовой), ₽: ").strip().replace(",", "."))
            if bid < start: raise ValueError
        except Exception:
            print("[ERROR] Неверная цена.", file=sys.stderr); sys.exit(1)
        bid = round_price_5(bid)
        pv = predict(pipe, start, bid, order_dt, tender_dt, dist)
        ev = bid * pv
        print(f"\nРезультат для {bid} ₽: P={pv:.3f}  E={bid:.0f}×{pv:.3f}={ev:.1f} ₽")