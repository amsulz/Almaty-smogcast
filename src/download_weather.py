"""Download hourly historical weather for Almaty from the Open-Meteo archive (ERA5). No API key needed.

Also the temperature higher up (850 and 700 hPa), it is needed for the inversion features.
The archive doesn't have it, so it comes from the archive of old forecasts (starts 2021-03-23).
"""
from datetime import date, timedelta

import pandas as pd
import requests

from config import LATITUDE, LONGITUDE, RAW_DIR, START_DATE, TIMEZONE

URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
    "boundary_layer_height",  # low boundary layer = pollution gets trapped (inversions)
]

UPPER_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
UPPER_START = "2021-03-23"
# 850 hPa is about 700 m above the city, 700 hPa about 2 km
UPPER_VARIABLES = ["temperature_850hPa", "temperature_700hPa"]


def download_upper(end_date):
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": UPPER_START,
        "end_date": end_date,
        "hourly": ",".join(UPPER_VARIABLES),
        "timezone": TIMEZONE,
    }
    response = requests.get(UPPER_URL, params=params, timeout=120)
    response.raise_for_status()
    upper = pd.DataFrame(response.json()["hourly"])
    upper["time"] = pd.to_datetime(upper["time"])
    return upper


def main():
    # The archive lags a few days behind today
    end_date = (date.today() - timedelta(days=5)).isoformat()
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": end_date,
        "hourly": ",".join(VARIABLES),
        "timezone": TIMEZONE,
    }
    response = requests.get(URL, params=params, timeout=120)
    response.raise_for_status()

    df = pd.DataFrame(response.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.merge(download_upper(end_date), on="time", how="left")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "weather.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
