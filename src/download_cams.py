"""Download the CAMS global PM2.5 forecast for Almaty via Open-Meteo. my model has to be better than this

CAMS global data on Open-Meteo starts in August 2022. No API key needed.
"""
from datetime import date, timedelta

import pandas as pd
import requests

from config import LATITUDE, LONGITUDE, RAW_DIR, TIMEZONE

URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
CAMS_START = date(2022, 8, 1)


def main():
    chunks = []
    start = CAMS_START
    while start < date.today():
        # one year per request, otherwise the answer is too big
        end = min(start + timedelta(days=365), date.today())
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": "pm2_5,pm10",
            "timezone": TIMEZONE,
        }
        response = requests.get(URL, params=params, timeout=120)
        response.raise_for_status()
        chunks.append(pd.DataFrame(response.json()["hourly"]))
        print(f"Downloaded {start} .. {end}")
        start = end + timedelta(days=1)

    df = pd.concat(chunks).drop_duplicates("time")
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"pm2_5": "cams_pm25", "pm10": "cams_pm10"})

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "cams.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
