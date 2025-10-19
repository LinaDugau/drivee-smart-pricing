import os, math, requests, joblib
import numpy as np
import pandas as pd
from datetime import datetime, date
from flask import Flask, render_template, request, jsonify

MODEL_PATH = "model.joblib"
ORS_API_KEY = 'eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjA4MGQ3YjU4ZDkzNjRjN2U4NjkyODA1YWNlMzZjZjYxIiwiaCI6Im11cm11cjY0In0='
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

def make_row(start, bid, order_dt, tender_dt, distance_m):
    order_hour, tender_hour = order_dt.hour, tender_dt.hour
    order_dow = tender_dow = 0
    delay_min = max(0.0, (tender_dt - order_dt).total_seconds()/60.0)
    ang = 2*math.pi/24.0

    bid_ratio  = bid / start
    log_ratio  = math.log(bid_ratio if bid_ratio > 1e-12 else 1e-12)
    uplift_rel = bid_ratio - 1.0
    uplift_abs = bid - start

    distance_m = float(distance_m)
    log_distance_m = math.log(distance_m if distance_m > 1.0 else 1.0)

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
        "order_hour_sin": math.sin(ang*order_hour),
        "order_hour_cos": math.cos(ang*order_hour),
        "tender_hour_sin": math.sin(ang*tender_hour),
        "tender_hour_cos": math.cos(ang*tender_hour),
        "distance_in_meters": distance_m,
        "log_distance_m": log_distance_m,
    }
    return pd.DataFrame([[data[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)

def load_model():
    global MODEL
    if MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("model.joblib не найден. Обучите модель: python train_model.py")
        obj = joblib.load(MODEL_PATH)
        MODEL = obj["pipeline"] if isinstance(obj, dict) and "pipeline" in obj else obj
    return MODEL

def predict_prob(pipe, start, bid, order_dt, tender_dt, distance_m) -> float:
    X = make_row(start, bid, order_dt, tender_dt, distance_m)
    return float(pipe.predict_proba(X)[:,1][0])

def build_curve(pipe, start, order_dt, tender_dt, distance_m):
    p_min = int(round(start))
    p_max = int(round(start*(1.0+UP_PCT)))
    grid = np.arange(p_min, max(p_min,p_max)+1, 1, dtype=int)
    probs = [predict_prob(pipe, start, z, order_dt, tender_dt, distance_m) for z in grid]
    df = pd.DataFrame({"price":grid, "p":probs})
    df["e"] = df["price"]*df["p"]
    return df

def pick_right(curve, idx_base, target_prob):
    right = curve.iloc[idx_base+1:]
    if right.empty: return curve.iloc[-1]
    cand = right[right["p"] <= target_prob]
    if len(cand): return cand.iloc[0]
    return right.loc[(right["p"]-target_prob).abs().idxmin()]

def ors_distance_and_route(point_a, point_b):
    """
    point_* = [lat, lon]
    Возвращает: (distance_meters, routeLatLng [[lat,lon], ...], error)
    """
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
            coords = data["features"][0]["geometry"]["coordinates"]  # [lon,lat]
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
    """
    JSON:
      start_price: number (обязательно)
      order_time: "HH:MM" (обязательно)
      tender_time?: "HH:MM"
      point_a?: [lat, lon]
      point_b?: [lat, lon]
      distance_meters?: number  (если точек нет)
    """
    try:
        payload = request.get_json(force=True)
        start = float(payload.get("start_price", 0))
        if start <= 0:
            return jsonify(error="Стартовая цена должна быть > 0"), 400

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

        curve = build_curve(pipe, start, order_dt, tender_dt, dist_m)
        idx_opt = int(curve["e"].idxmax())
        best = curve.loc[idx_opt]
        p_opt_raw = float(best["p"])
        t90, t60, t20 = 0.90*p_opt_raw, 0.60*p_opt_raw, 0.20*p_opt_raw

        pt90 = pick_right(curve, idx_opt, t90)
        pt60 = pick_right(curve, idx_opt, t60)
        pt20 = pick_right(curve, idx_opt, t20)

        def pack(row, label):
            price = round_price_5(float(row["price"]))
            p = predict_prob(pipe, start, price, order_dt, tender_dt, dist_m)
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
    """
    JSON:
      start_price: number
      bid_price: number   
      order_time: "HH:MM"
      tender_time?: "HH:MM"
      point_a?: [lat, lon]
      point_b?: [lat, lon]
      distance_meters?: number (если точек нет)
    Ответ: { price: округлённая_до_5, p: вероятность, e: ожидание, distance_meters?, route? }
    """
    try:
        payload = request.get_json(force=True)
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
        p = predict_prob(pipe, start, price5, order_dt, tender_dt, dist_m)
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