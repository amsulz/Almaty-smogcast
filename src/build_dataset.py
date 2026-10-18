"""Merge PM2.5, weather and CAMS into one hourly table: data/processed/dataset.csv"""
import pandas as pd

from config import EMBASSY_END, PROCESSED_DIR, RAW_DIR, START_DATE


def main():
    pm = pd.read_csv(RAW_DIR / "pm25_openaq.csv", parse_dates=["time"])
    # after the embassy stopped, PM2.5 comes from AirGradient sensors (download_airgradient.py)
    airgradient_file = RAW_DIR / "pm25_airgradient.csv"
    if airgradient_file.exists():
        ag = pd.read_csv(airgradient_file, parse_dates=["time"])
        pm = pd.concat([pm[pm.time < EMBASSY_END], ag[ag.time >= EMBASSY_END]])
    weather = pd.read_csv(RAW_DIR / "weather.csv", parse_dates=["time"])
    cams = pd.read_csv(RAW_DIR / "cams.csv", parse_dates=["time"])

    print("pm25:", pm.shape, pm.time.min(), "->", pm.time.max())
    print("weather:", weather.shape, weather.time.min(), "->", weather.time.max())
    print("cams:", cams.shape, cams.time.min(), "->", cams.time.max())

    # junk values. AirNow uses -999 for missing, and sometimes there are crazy single spikes
    bad = (pm.pm25 < 0) | (pm.pm25 > 1000)
    print(f"bad pm25 values: {bad.sum()}")
    pm.loc[bad, "pm25"] = None

    # full hourly index, so missing hours are real NaN rows and not just gaps
    end = max(pm.time.max(), weather.time.max())
    hours = pd.DataFrame({"time": pd.date_range(START_DATE, end, freq="h")})

    df = hours.merge(pm, on="time", how="left")
    df = df.merge(weather, on="time", how="left")
    df = df.merge(cams, on="time", how="left")

    df = df.rename(columns={
        "temperature_2m": "temp",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m": "wind",
        "wind_direction_10m": "wind_dir",
        "surface_pressure": "pressure",
        "boundary_layer_height": "blh",
    })

    print(f"\nrows: {len(df)}")
    print(f"pm25 missing: {df.pm25.isna().mean() * 100:.1f}%")
    print("missing by year (%):")
    print((df.pm25.isna().groupby(df.time.dt.year).mean() * 100).round(1))

    # sanity check for timezone. I expected the peak in the evening, but at the embassy it's
    # at 13-15. checked against Clarity sensors nearby (independent clocks): same cycle,
    # so the time is right, it's just how the air moves there
    winter = df[df.time.dt.month.isin([12, 1, 2])]
    by_hour = winter.groupby(winter.time.dt.hour).pm25.mean()
    print(f"\nwinter: dirtiest hour {by_hour.idxmax()}, cleanest hour {by_hour.idxmin()}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "dataset.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
