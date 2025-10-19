import os, math, requests, joblib
import numpy as np
import pandas as pd
from datetime import datetime, date
from flask import Flask, render_template, request, jsonify

MODEL_PATH = "predictions.joblib"   
ORS_API_KEY = os.getenv("ORS_API_KEY")
UP_PCT = 0.60 

FEATURE_COLS = [
    "price_start_local","price_bid_local","bid_ratio","log_ratio","uplift_rel","uplift_abs",
    "order_hour","order_dow","tender_hour","tender_dow","bid_delay_min",
    "order_hour_sin","order_hour_cos","tender_hour_sin","tender_hour_cos",
    "distance_in_meters","log_distance_m",
]

app = Flask(__name__)
MODEL = None
ors_session = requests.Session()

def parse_hhmm_to_dt(s: str) -> datetime:
    s = (s or "").strip()
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            t = datetime.strptime(s, fmt).time()
            return datetime.combine(date.today(), t)
        except Exception:
            pass
    raise ValueError("Время должно быть в формате HH:MM")

def round_price_5(x: float) -> int:
    return int(round(x/5)*5)

def make_row(start, bid, order_dt, tender_dt, distance_m, carname=None, carmodel=None, platform=None, driver_rating=None):
    order_hour  = int(order_dt.hour)
    tender_hour = int(tender_dt.hour)
    order_dow = tender_dow = 0  # если нет даты — фиксируем 0
    delay_min = max(0.0, (tender_dt - order_dt).total_seconds() / 60.0)
    ang = 2 * math.pi / 24.0

    distance_m = float(distance_m)
    log_distance_m = math.log(distance_m if distance_m > 1.0 else 1.0)
    log_price_start = math.log(start if start > 1.0 else 1.0)

    base = {
        "price_start_local": float(start),
        "log_price_start": log_price_start,
        "order_hour": order_hour,
        "order_dow": order_dow,
        "tender_hour": tender_hour,
        "tender_dow": tender_dow,
        "order_hour_sin": math.sin(ang * order_hour),
        "order_hour_cos": math.cos(ang * order_hour),
        "tender_hour_sin": math.sin(ang * tender_hour),
        "tender_hour_cos": math.cos(ang * tender_hour),
        "bid_delay_min": float(delay_min),
        "distance_in_meters": distance_m,
        "log_distance_m": log_distance_m,
        # опциональные числовые — по умолчанию NaN
        "duration_in_seconds": np.nan,
        "avg_speed_kmh": np.nan,
        "pickup_in_meters": np.nan,
        "pickup_in_seconds": np.nan,
        "pickup_speed_kmh": np.nan,
        "driver_rating": np.nan,
        "driver_experience_days": np.nan,
    }

    if "carname" in FEATURE_COLS:
        base["carname"] = (carname or "unknown")
    if "carmodel" in FEATURE_COLS:
        base["carmodel"] = (carmodel or "unknown")
    if "platform" in FEATURE_COLS:
        base["platform"] = (platform or "unknown")

    if "driver_rating" in FEATURE_COLS:
        try:
            base["driver_rating"] = float(driver_rating) if driver_rating is not None and str(driver_rating) != "" else np.nan
        except Exception:
            base["driver_rating"] = np.nan

    row = {}
    for col in FEATURE_COLS:
        row[col] = base.get(col, np.nan if col not in ("carname","carmodel","platform") else "unknown")
    return pd.DataFrame([row], columns=FEATURE_COLS)

def load_model():
    global MODEL, FEATURE_COLS, CATEGORICAL_FEATURES

    if MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"{MODEL_PATH} не найден. Сначала обучите модель.")
        obj = joblib.load(MODEL_PATH)
        if isinstance(obj, dict):
            MODEL = obj.get("pipeline", obj)
            FEATURE_COLS = obj.get("feature_cols", [])
            CATEGORICAL_FEATURES = obj.get("categorical_features", [])
        else:
            MODEL = obj
            FEATURE_COLS = []
            CATEGORICAL_FEATURES = []
        print(f"[INFO] Модель загружена: {MODEL_PATH}")
        print(f"[INFO] Признаков: {len(FEATURE_COLS)} | Категориальных: {len(CATEGORICAL_FEATURES)}")
    return MODEL

def predict_prob(pipe, start, bid, order_dt, tender_dt, distance_m, carname=None, carmodel=None, platform=None, driver_rating=None) -> float:
    X = make_row(start, bid, order_dt, tender_dt, distance_m, carname, carmodel, platform)
    if hasattr(pipe, "predict_proba"):
        return float(pipe.predict_proba(X)[:, 1][0])
    pred_bid = float(pipe.predict(X)[0])  
    m = bid / max(pred_bid, 1e-6)         
    k = 8.0                               
    p = 1.0 / (1.0 + math.exp(k * (m - 1.0)))
    return float(min(0.995, max(0.005, p)))

def build_curve(pipe, start, order_dt, tender_dt, distance_m, carname=None, carmodel=None, platform=None, driver_rating=None):
    p_min = int(round(start))
    p_max = int(round(start * (1.0 + UP_PCT)))
    grid = np.arange(p_min, max(p_min, p_max) + 1, 1, dtype=int)
    probs = [
        predict_prob(pipe, start, z, order_dt, tender_dt, distance_m, carname, carmodel, platform)
        for z in grid
    ]
    df = pd.DataFrame({"price": grid, "p": probs})
    df["e"] = df["price"] * df["p"]
    return df

def pick_right(curve, idx_base, target_prob):
    right = curve.iloc[idx_base+1:]
    if right.empty: return curve.iloc[-1]
    cand = right[right["p"] <= target_prob]
    if len(cand): return cand.iloc[0]
    return right.loc[(right["p"]-target_prob).abs().idxmin()]

def ors_distance_and_route(point_a, point_b):
    if not ORS_API_KEY or ORS_API_KEY == "PASTE_YOUR_ORS_KEY_HERE":
        return None, None, "Нет ORS_API_KEY (export/set ORS_API_KEY=...)"
    url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    body = {"coordinates": [[point_a[1], point_a[0]], [point_b[1], point_b[0]]], "units":"m"}
    try:
        r = ors_session.post(url, json=body, headers=headers, timeout=10.0)
        if r.status_code == 200:
            data = r.json()
            dist_m = float(data["features"][0]["properties"]["summary"]["distance"])
            coords = data["features"][0]["geometry"]["coordinates"]
            route = [[c[1], c[0]] for c in coords]
            return dist_m, route, None
        else:
            msg = "Ошибка ORS"
            try: msg = r.json().get("error",{}).get("message", msg)
            except Exception: pass
            if r.status_code == 401: msg = "Неверный ORS API-ключ"
            if r.status_code == 404: msg = "Маршрут не найден"
            if r.status_code == 429: msg = "Слишком много запросов (429)"
            return None, None, f"ORS HTTP {r.status_code}: {msg}"
    except requests.exceptions.Timeout:
        return None, None, "Таймаут ORS"
    except requests.exceptions.ConnectionError:
        return None, None, "Нет соединения с ORS"
    except Exception:
        return None, None, "Неожиданная ошибка ORS"

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/calc")
def api_calc():
    try:
        payload = request.get_json(force=True)
        carname = (payload.get("carname") or "").strip() or None
        carmodel = (payload.get("carmodel") or "").strip() or None
        platform = (payload.get("platform") or "").strip() or None

        dr_raw = payload.get("driver_rating", None)
        try:
            driver_rating = float(dr_raw) if dr_raw not in (None, "",) else None
        except Exception:
            driver_rating = None

        start = float(payload.get("start_price", 0))
        if start <= 0:
            return jsonify(error="Стартовая цена должна быть > 0"), 400

        order_dt = parse_hhmm_to_dt(payload.get("order_time",""))
        tender_s = (payload.get("tender_time") or "").strip()
        tender_dt = parse_hhmm_to_dt(tender_s) if tender_s else order_dt
        carname = (payload.get("carname") or "").strip() or None
        carmodel = (payload.get("carmodel") or "").strip() or None
        platform = (payload.get("platform") or "").strip() or None

        point_a = payload.get("point_a")
        point_b = payload.get("point_b")

        dist_m, route = None, None
        if point_a and point_b:
            if not (isinstance(point_a, list) and isinstance(point_b, list) and len(point_a)==2 and len(point_b)==2):
                return jsonify(error="Некорректные координаты A/B"), 400
            dist_m, route, err = ors_distance_and_route(point_a, point_b)
            if err: return jsonify(error=err), 400
        else:
            if "distance_meters" not in payload or str(payload["distance_meters"]).strip()=="":
                return jsonify(error="Задайте точки A и B на карте или дистанцию в метрах"), 400
            dist_m = float(payload["distance_meters"])
            if dist_m < 0: return jsonify(error="Дистанция должна быть ≥ 0"), 400

        pipe = load_model()

        curve = build_curve(pipe, start, order_dt, tender_dt, dist_m, carname, carmodel, platform, driver_rating)
        idx_opt = int(curve["e"].idxmax())
        best = curve.loc[idx_opt]
        p_opt_raw = float(best["p"])
        t90, t60, t20 = 0.90*p_opt_raw, 0.60*p_opt_raw, 0.20*p_opt_raw

        pt90 = pick_right(curve, idx_opt, t90)
        pt60 = pick_right(curve, idx_opt, t60)
        pt20 = pick_right(curve, idx_opt, t20)

        def pack(row, label):
            price = round_price_5(float(row["price"]))
            p = predict_prob(pipe, start, price, order_dt, tender_dt, dist_m,
                            carname, carmodel, platform, driver_rating)
            e = price * p
            return {"label": label, "price": int(price), "p": round(p,3), "e": round(e,1)}

        out = {
            "rows": [
                pack(best, "Лучшая (макс. ожидание)"),
                pack(pt90, "90% от P(лучшей), правее"),
                pack(pt60, "60% от P(лучшей), правее"),
                pack(pt20, "20% от P(лучшей), правее"),
            ],
            "distance_meters": round(dist_m, 1),
        }
        if route:
            out["route"] = route
        return jsonify(out)

    except FileNotFoundError as e:
        return jsonify(error=str(e)), 500
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        return jsonify(error=f"Внутренняя ошибка: {e}"), 500

@app.post("/api/custom")
def api_custom():
    try:
        payload = request.get_json(force=True)
        carname = (payload.get("carname") or "").strip() or None
        carmodel = (payload.get("carmodel") or "").strip() or None
        platform = (payload.get("platform") or "").strip() or None

        dr_raw = payload.get("driver_rating", None)
        try:
            driver_rating = float(dr_raw) if dr_raw not in (None, "",) else None
        except Exception:
            driver_rating = None

        start = float(payload.get("start_price", 0))
        bid   = float(payload.get("bid_price", 0))
        if start <= 0 or bid <= 0:
            return jsonify(error="Цены должны быть > 0"), 400
        if bid < start:
            return jsonify(error="Ваша цена должна быть не меньше стартовой"), 400

        order_dt = parse_hhmm_to_dt(payload.get("order_time",""))
        tender_s = (payload.get("tender_time") or "").strip()
        tender_dt = parse_hhmm_to_dt(tender_s) if tender_s else order_dt

        point_a = payload.get("point_a")
        point_b = payload.get("point_b")

        dist_m, route = None, None
        if point_a and point_b:
            if not (isinstance(point_a, list) and isinstance(point_b, list) and len(point_a)==2 and len(point_b)==2):
                return jsonify(error="Некорректные координаты A/B"), 400
            dist_m, route, err = ors_distance_and_route(point_a, point_b)
            if err: return jsonify(error=err), 400
        else:
            if "distance_meters" not in payload or str(payload["distance_meters"]).strip()=="":
                return jsonify(error="Задайте точки A и B на карте или дистанцию в метрах"), 400
            dist_m = float(payload["distance_meters"])
            if dist_m < 0: return jsonify(error="Дистанция должна быть ≥ 0"), 400

        pipe = load_model()
        price5 = round_price_5(bid)
        p = predict_prob(pipe, start, price5, order_dt, tender_dt, dist_m, carname, carmodel, platform, driver_rating)
        e = price5 * p

        out = {"price": int(price5), "p": round(p,3), "e": round(e,1), "distance_meters": round(dist_m,1)}
        if route:
            out["route"] = route
        return jsonify(out)

    except FileNotFoundError as e:
        return jsonify(error=str(e)), 500
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        return jsonify(error=f"Внутренняя ошибка: {e}"), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)