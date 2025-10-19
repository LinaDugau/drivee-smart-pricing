# 🚖 Drivee Smart Pricing (Умная цена)

Веб-приложение рекомендует **оптимальную цену поездки такси**, прогнозируя вероятность принятия водителем и ожидаемый доход.

**На вход**: цена пассажира, время, маршрут (A→B) или дистанция  
**На выход**: лучшая цена + альтернативы (90% / 60% / 20% от максимальной вероятности) + расчёт «своей цены».

---

## 🚀 Развёртывание (локально)

### 1) Клонирование и окружение

```bash
git clone https://github.com/LinaDugau/drivee-smart-pricing.git
cd drivee-smart-pricing

python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

cd demo
```

### 2) Зависимости

```bash
pip install -r requirements.txt
```

### 3) Переменные окружения

Получите бесплатный ключ OpenRouteService: https://openrouteservice.org/dev/#/signup

```bash
# macOS/Linux:
export ORS_API_KEY="ВАШ_КЛЮЧ"

# Windows (PowerShell):
setx ORS_API_KEY "ВАШ_КЛЮЧ"
# или на текущую сессию:
set ORS_API_KEY=ВАШ_КЛЮЧ
```

### 4) Модель

Обучите модель на своем `train.csv` или используйте приложенную обученную модель `model.joblib` в корень проекта:

```bash
python train_model.py
```

### 5) Запуск

```bash
python app.py
```

Откройте: http://127.0.0.1:5000



## 📁 Структура репозитория

```
drivee-smart-pricing/
├── demo/
│   ├── app.py
│   ├── recommend_three_prices.py 
│   ├── model.joblib
│   ├── predictions.joblib 
│   ├── train_model.py
│   ├── requirements.txt
│   ├── train.csv
│   ├── test.csv
│   ├── static/
│   │   └── driver.jpg
│   └── templates/
│       └── index.html             
├── 07-ХХ.pptx             
├── Demo.txt
├── predictions.joblib               
├── Screencast.txt                           
└── README.md
```

## 🎯 Что внутри UI (index.html)

- **Интерактивная карта** (OSM/Leaflet): ставьте A и B (геокодинг Nominatim)
- **Маршрут по улицам** строится через ORS; дистанция передаётся в модель
- **Модальное окно «Своя цена»**: введите цену и нажмите Enter — сразу получите P(accept)% и E[доход]
