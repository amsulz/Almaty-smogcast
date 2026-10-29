"""Download hourly PM2.5 measured at the US Embassy station in Almaty from OpenAQ v3.

Needs a free API key from https://explore.openaq.org/register
Put it in a .env file in the project root:  OPENAQ_API_KEY=your_key
"""
import os
import time
from datetime import date

import pandas as pd
import requests
from dotenv import load_dotenv

from config import OPENAQ_LOCATION_ID, RAW_DIR, START_DATE, UTC_OFFSET_HOURS

BASE = "https://api.openaq.org/v3"
PAGE_SIZE = 1000


def get(session, path, **params):
    errors = 0
    while True:
        response = session.get(f"{BASE}{path}", params=params, timeout=60)
        if response.status_code == 429:  # rate limit: wait and retry
            time.sleep(10)
            continue
        # their server sometimes just gives 500 in the middle of a year, retry helps
        if response.status_code >= 500 and errors < 5:
            errors += 1
            print(f"  server error {response.status_code}, retry {errors}")
            time.sleep(10 * errors)
            continue
        response.raise_for_status()
        return response.json()


def find_pm25_sensor(session):
    sensors = get(session, f"/locations/{OPENAQ_LOCATION_ID}/sensors")["results"]
    for sensor in sensors:
        if sensor["parameter"]["name"] == "pm25":
            return sensor["id"]
    raise RuntimeError(f"No PM2.5 sensor at location {OPENAQ_LOCATION_ID}")


def main():
    load_dotenv()
    key = os.getenv("OPENAQ_API_KEY")
    if not key:
        raise SystemExit("Set OPENAQ_API_KEY in .env (see .env.example)")

    session = requests.Session()
    session.headers["X-API-Key"] = key

    sensor_id = find_pm25_sensor(session)
    print(f"PM2.5 sensor id: {sensor_id}")

    rows = []
    # one year at a time, and inside a year page by page
    for year in range(int(START_DATE[:4]), date.today().year + 1):
        page = 1
        while True:
            data = get(
                session,
                f"/sensors/{sensor_id}/hours",
                datetime_from=f"{year}-01-01T00:00:00Z",
                datetime_to=f"{year + 1}-01-01T00:00:00Z",
                limit=PAGE_SIZE,
                page=page,
            )
            results = data["results"]
            rows += [
                {"time_utc": r["period"]["datetimeFrom"]["utc"], "pm25": r["value"]}
                for r in results
            ]
            if len(results) < PAGE_SIZE:
                break
            page += 1
            time.sleep(1)  # be polite to the free API
        print(f"{year}: total rows so far {len(rows)}")

    df = pd.DataFrame(rows)
    # Convert to the same local time as the weather data. Not tz_convert("Asia/Almaty"):
    # it gives +6 before March 2024 and then everything before that is 1 hour off from weather
    df["time"] = (
        pd.to_datetime(df["time_utc"], utc=True).dt.tz_localize(None)
        + pd.Timedelta(hours=UTC_OFFSET_HOURS)
    )
    df = df[["time", "pm25"]].drop_duplicates("time").sort_values("time")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "pm25_openaq.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
