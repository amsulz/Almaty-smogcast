"""Download daily PM2.5 for Medeu district from the Almaty Air Initiative (AAI) API.

AAI corrects the raw AirGradient sensors. Their Medeu average is very close to the US embassy
station (2020-2025: 1.03 times the embassy, correlation 0.92), so after the embassy stopped
I use it as the level of PM2.5. The hourly shape still comes from download_airgradient.py.

Needs AAI_API_KEY in .env. Saves data/raw/pm25_aai_medeu.csv (date, pm25)
"""
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from config import EMBASSY_END, RAW_DIR

URL = "https://backend.air.org.kz/v1/pm25/calendar"
DISTRICT = "Medeu"  # the US consulate is in Medeu district


def main():
    load_dotenv()
    key = os.getenv("AAI_API_KEY")
    if not key:
        print("no AAI_API_KEY, skipping AAI data")
        return
    rows = []
    for year in range(pd.Timestamp(EMBASSY_END).year, pd.Timestamp.now().year + 1):
        r = requests.get(URL, params={"district": DISTRICT, "year": year},
                         headers={"X-API-Key": key}, timeout=60)
        r.raise_for_status()
        rows += r.json()
        time.sleep(1)
    df = pd.DataFrame(rows).rename(columns={"avg_pm25": "pm25"}).dropna()
    out = RAW_DIR / "pm25_aai_medeu.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} days to {out}, {df.date.min()} -> {df.date.max()}")


if __name__ == "__main__":
    main()
