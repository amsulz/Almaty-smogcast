"""Download hourly PM2.5 from AirGradient sensors near the US Embassy station (OpenAQ v3).

The embassy station stopped on 2025-11-14. AirGradient sensors are cheap and read higher
than the embassy, so first I compare one AirGradient sensor that worked at the same time
as the embassy, and scale everything down by that ratio.

Run download_openaq.py first (the embassy data is needed for the comparison).
Saves data/raw/pm25_airgradient.csv in the same format as pm25_openaq.csv
"""
import math
import os

import pandas as pd
import requests
from dotenv import load_dotenv

from config import (AIRGRADIENT_BRIDGE_LOCATION, AIRGRADIENT_RADIUS_KM, EMBASSY_LAT, EMBASSY_LON,
                    RAW_DIR, UTC_OFFSET_HOURS)
from download_openaq import PAGE_SIZE, get

# AirGradient appeared in Almaty in late 2024, nothing to download before that
FROM_DATE = "2024-10-01"
# a day needs this many hours to count in the comparison (same idea as MIN_HOURS in features.py)
MIN_HOURS = 18


def distance_km(lat, lon):
    # good enough for a few km: 1 degree of latitude is 111 km, of longitude at 43° about 81 km
    return math.hypot((lat - EMBASSY_LAT) * 111, (lon - EMBASSY_LON) * 81)


def pm25_sensor(location):
    for sensor in location["sensors"]:
        if sensor["parameter"]["name"] == "pm25":
            return sensor["id"]
    return None


def find_locations(session):
    data = get(session, "/locations", coordinates=f"{EMBASSY_LAT},{EMBASSY_LON}",
               radius=25000, limit=1000)
    return [loc for loc in data["results"]
            if (loc.get("provider") or {}).get("name") == "AirGradient" and pm25_sensor(loc)]


def download_hours(session, sensor_id):
    rows, page = [], 1
    while True:
        data = get(session, f"/sensors/{sensor_id}/hours", datetime_from=FROM_DATE,
                   limit=PAGE_SIZE, page=page)
        rows += [(r["period"]["datetimeFrom"]["utc"], r["value"]) for r in data["results"]]
        if len(data["results"]) < PAGE_SIZE:
            break
        page += 1
    df = pd.DataFrame(rows, columns=["time_utc", "pm25"])
    # same local time as everything else (see download_openaq.py)
    df["time"] = pd.to_datetime(df.time_utc, utc=True).dt.tz_localize(None) + pd.Timedelta(hours=UTC_OFFSET_HOURS)
    series = df.drop_duplicates("time").set_index("time").pm25
    return series[(series >= 0) & (series < 1000)]


def daily_mean(series):
    days = series.resample("D")
    return days.mean()[days.count() >= MIN_HOURS]


def main():
    load_dotenv()
    key = os.getenv("OPENAQ_API_KEY")
    if not key:
        raise SystemExit("Set OPENAQ_API_KEY in .env (see .env.example)")
    session = requests.Session()
    session.headers["X-API-Key"] = key

    locations = find_locations(session)
    print(f"AirGradient locations around Almaty: {len(locations)}")

    # 1. how much higher does AirGradient read? compare with the embassy on the same days
    bridge = next(loc for loc in locations if loc["id"] == AIRGRADIENT_BRIDGE_LOCATION)
    bridge_days = daily_mean(download_hours(session, pm25_sensor(bridge)))
    embassy = pd.read_csv(RAW_DIR / "pm25_openaq.csv", parse_dates=["time"]).set_index("time").pm25
    embassy_days = daily_mean(embassy[(embassy >= 0) & (embassy < 1000)])
    both = pd.concat([embassy_days.rename("embassy"), bridge_days.rename("ag")], axis=1, sort=True).dropna()
    ratio = both.embassy.sum() / both.ag.sum()
    print(f"same days with the embassy: {len(both)}, correlation {both.corr().iloc[0, 1]:.2f}")
    print(f"embassy = {ratio:.2f} x AirGradient")

    # 2. sensors close to the embassy, one hourly value = their median
    near = [loc for loc in locations
            if distance_km(loc["coordinates"]["latitude"], loc["coordinates"]["longitude"]) <= AIRGRADIENT_RADIUS_KM]
    print(f"sensors within {AIRGRADIENT_RADIUS_KM} km of the embassy: {len(near)}")
    sensors = {}
    for loc in near:
        sensors[loc["id"]] = download_hours(session, pm25_sensor(loc))
        print(f"  location {loc['id']}: {len(sensors[loc['id']])} hours")
    # median and not mean, so one broken sensor can't pull the value up
    hourly = pd.concat(sensors, axis=1, sort=True).median(axis=1) * ratio

    df = hourly.round(1).rename("pm25").rename_axis("time").reset_index()
    out = RAW_DIR / "pm25_airgradient.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}, {df.time.min()} -> {df.time.max()}")


if __name__ == "__main__":
    main()
