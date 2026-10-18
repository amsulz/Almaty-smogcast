"""Streamlit demo: last days of PM2.5 in Almaty + forecast for tomorrow.

Run:  streamlit run app.py
Before that: download data, build_dataset.py and train.py (it saves models/lgbm.txt)
"""
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))
from config import LATITUDE, LONGITUDE, MODELS_DIR, TIMEZONE  # noqa: E402
from evaluate import category  # noqa: E402
from features import feature_columns, load_hourly, make_features  # noqa: E402

COLORS = {"low": "#3a9b3a", "high": "#e08a1e", "very high": "#c0392b"}
WEATHER = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind",
    "surface_pressure": "pressure",
    "precipitation": "precipitation",
    "boundary_layer_height": "blh",
}


@st.cache_data(ttl=3600)
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
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"]).rename(columns=WEATHER)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


@st.cache_data
def load_data():
    hourly = load_hourly().set_index("time")
    try:
        fc = recent_weather()
        # full hour range, so a gap between the archive and the forecast doesn't break the days
        hourly = hourly.reindex(pd.date_range(hourly.index.min(), fc.index.max(), freq="h"))
        for col in WEATHER.values():
            hourly[col] = hourly[col].fillna(fc[col])
    except requests.RequestException:
        st.warning("Could not get the weather forecast, tomorrow's weather will be empty")
    hourly = hourly.reset_index().rename(columns={"index": "time"})
    return hourly, make_features(hourly)


st.set_page_config(page_title="Almaty smog")
plt.style.use("dark_background")
st.title("Almaty smog")
st.write("My forecast of tomorrow's PM2.5 in Almaty. Data from the US Embassy station, model is LightGBM.")

model_file = MODELS_DIR / "lgbm.txt"
if not model_file.exists():
    st.error("No model yet. Run `python src/train.py` first.")
    st.stop()

model = lgb.Booster(model_file=str(model_file))
hourly, daily = load_data()

# "today" = the last day where we have enough PM2.5 measurements
today = daily.dropna(subset=["pm25_mean24h"]).iloc[-1]
tomorrow = today.date + pd.Timedelta(days=1)
pred = model.predict(today[feature_columns(daily)].to_frame().T.astype(float))[0]
cat = category(pd.Series([pred]))[0]

st.metric(f"Forecast for {tomorrow:%d %b %Y}", f"{pred:.0f} µg/m³",
          f"{pred - today.pm25_mean24h:+.0f} vs today", delta_color="inverse")
st.markdown(f"Level: <span style='color:{COLORS[cat]}'>**{cat}**</span>", unsafe_allow_html=True)
if today.date < pd.Timestamp.now().normalize() - pd.Timedelta(days=2):
    st.info(f"Last PM2.5 data is from {today.date:%d %b %Y}. The embassy station stopped sending data.")

st.subheader("Last 14 days")
last = daily[(daily.date > today.date - pd.Timedelta(days=14)) & (daily.date <= today.date)]
bar_colors = [COLORS.get(c, "lightgray") for c in category(last.pm25).astype(str)]
fig, ax = plt.subplots(figsize=(8, 3.5))
fig.patch.set_alpha(0)
ax.set_facecolor("none")
ax.bar(last.date, last.pm25, color=bar_colors)
ax.bar(tomorrow, pred, color="none", edgecolor="white", hatch="//", label="forecast")
ax.axhline(35.4, color="gray", linestyle="--", linewidth=1)
ax.set_ylabel("daily mean PM2.5")
ax.legend()
fig.autofmt_xdate()
st.pyplot(fig)

st.subheader("Hourly, last 7 days")
week = hourly[(hourly.time > today.date - pd.Timedelta(days=6)) & (hourly.time < tomorrow)]
st.line_chart(week.set_index("time")[["pm25"]])

st.caption("Colors: low (0-35), high (35-125), very high (125+), "
           "roughly based on US EPA levels. WHO says the daily mean should be under 15.")
