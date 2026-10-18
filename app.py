"""Streamlit demo: last days of PM2.5 in Almaty + forecast for tomorrow.

Run:  streamlit run app.py
Before that: download data, build_dataset.py and train.py (it saves models/lgbm.txt)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))
from config import MODELS_DIR  # noqa: E402
from evaluate import category  # noqa: E402
from forecast import load_data, load_model, predict_tomorrow  # noqa: E402

COLORS = {"low": "#3a9b3a", "high": "#e08a1e", "very high": "#c0392b"}

st.set_page_config(page_title="Almaty smog")
plt.style.use("dark_background")
st.title("Almaty smog")
st.write("My forecast of tomorrow's PM2.5 in Almaty. Data from the US Embassy station until November 2025, "
         "then from AirGradient sensors near it. Model is LightGBM.")

if not (MODELS_DIR / "lgbm.txt").exists():
    st.error("No model yet. Run `python src/train.py` first.")
    st.stop()

try:
    # cache for an hour, so the weather forecast isn't downloaded on every click
    hourly, daily = st.cache_data(ttl=3600)(load_data)()
except requests.RequestException:
    st.error("Could not get the weather forecast from Open-Meteo, try again later")
    st.stop()

today, tomorrow, pred = predict_tomorrow(load_model(), daily)
cat = category(pd.Series([pred]))[0]

st.metric(f"Forecast for {tomorrow:%d %b %Y}", f"{pred:.0f} µg/m³",
          f"{pred - today.pm25_mean24h:+.0f} vs today", delta_color="inverse")
st.markdown(f"Level: <span style='color:{COLORS[cat]}'>**{cat}**</span>", unsafe_allow_html=True)
if today.date < pd.Timestamp.now().normalize() - pd.Timedelta(days=2):
    st.info(f"Last PM2.5 data is from {today.date:%d %b %Y}. Run download_airgradient.py and "
            "build_dataset.py to update.")

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
