"""Forecast for tomorrow from the newest data. Used by the demo site (app.py) and the Telegram bot."""
import time

import lightgbm as lgb
import pandas as pd
import requests

from config import LATITUDE, LONGITUDE, MODELS_DIR, TIMEZONE
from features import feature_columns, load_hourly, make_features

WEATHER = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind",
    "surface_pressure": "pressure",
    "precipitation": "precipitation",
    "boundary_layer_height": "blh",
}


def recent_weather():
    # the archive is ~5 days behind, so the last days + tomorrow come from the forecast API
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(WEATHER),
        "past_days": 10,
        "forecast_days": 3,
        "timezone": TIMEZONE,
    }
    # open-meteo sometimes doesn't answer in time, so we try 3 times before giving up
    for attempt in range(3):
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=60)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(30)
    df = pd.DataFrame(r.json()["hourly"]).rename(columns=WEATHER)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def load_data():
    """Hourly table + daily features, with the archive weather filled up by the forecast."""
    hourly = load_hourly().set_index("time")
    fc = recent_weather()
    # full hour range, so a gap between the archive and the forecast doesn't break the days
    hourly = hourly.reindex(pd.date_range(hourly.index.min(), fc.index.max(), freq="h"))
    for col in WEATHER.values():
        hourly[col] = hourly[col].fillna(fc[col])
    hourly = hourly.reset_index().rename(columns={"index": "time"})
    daily = make_features(hourly)
    # the bot runs at 22:00, so the 23:00 value doesn't exist yet
    # before, it was empty, now we take the last measured hour
    pm = hourly.set_index("time").pm25.dropna()
    last_day = daily.date == pm.index.max().normalize()
    daily["pm25_lag1h"] = daily.pm25_lag1h.mask(last_day & daily.pm25_lag1h.isna(), pm.iloc[-1])
    return hourly, daily


def load_model(name="lgbm.txt"):
    return lgb.Booster(model_file=str(MODELS_DIR / name))


def predict_tomorrow(model, daily):
    # "today" = the last day where we have enough PM2.5 measurements
    today = daily.dropna(subset=["pm25_mean24h"]).iloc[-1]
    pred = model.predict(today[feature_columns(daily)].to_frame().T.astype(float))[0]
    return today, today.date + pd.Timedelta(days=1), pred


def recent_bias(model, daily, days=30):
    # the model learned the air of 2020-2025, but summer 2026 was cleaner and it kept
    # predicting about 3 too high. So the bot takes off its mean error of the last 30 days
    known = daily.dropna(subset=["pm25_mean24h", "target"]).tail(days)
    if len(known) < 10:
        return 0.0
    pred = model.predict(known[feature_columns(daily)].astype(float))
    return float((pred - known.target).mean())
