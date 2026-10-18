"""Turn the hourly dataset into one row per day.

Idea: we stand at the end of day D (23:00) and want to predict the mean PM2.5 of day D+1.
So every feature must be known by the end of day D. Only exception is weather for tomorrow,
which I take from the archive as if it was a perfect weather forecast (see README limitations).
"""
import pandas as pd

from config import EMBASSY_END, HEATING_END, HEATING_START, PROCESSED_DIR, TRAIN_END, VAL_END

# Kazakhstan public holidays with fixed dates (month, day).
# Kurban Ait moves every year so I skip it.
HOLIDAYS = [
    (1, 1), (1, 2), (1, 7), (3, 8), (3, 21), (3, 22), (3, 23),
    (5, 1), (5, 7), (5, 9), (7, 6), (8, 30), (10, 25), (12, 16),
]

WEATHER_COLS = ["temp", "humidity", "wind", "pressure", "precipitation", "blh"]

# a day needs at least this many hours of PM2.5, otherwise its mean is not trustworthy
MIN_HOURS = 18


def load_hourly():
    df = pd.read_csv(PROCESSED_DIR / "dataset.csv", parse_dates=["time"])
    return df


def is_heating(dates):
    md = dates.dt.month * 100 + dates.dt.day
    start = HEATING_START[0] * 100 + HEATING_START[1]
    end = HEATING_END[0] * 100 + HEATING_END[1]
    return ((md >= start) | (md < end)).astype(int)


def is_holiday(dates):
    md = list(zip(dates.dt.month, dates.dt.day))
    return pd.Series([int(x in HOLIDAYS) for x in md], index=dates.index)


def add_calendar(daily, prefix=""):
    # calendar for the date in column "date" (or for tomorrow if prefix="tmr_")
    d = daily["date"] if prefix == "" else daily["date"] + pd.Timedelta(days=1)
    daily[prefix + "dayofweek"] = d.dt.dayofweek
    daily[prefix + "weekend"] = (d.dt.dayofweek >= 5).astype(int)
    daily[prefix + "holiday"] = is_holiday(d)
    if prefix == "":
        daily["month"] = d.dt.month
        daily["dayofyear"] = d.dt.dayofyear
        daily["heating"] = is_heating(d)
    return daily


def make_daily(df):
    df = df.copy()
    df["date"] = df.time.dt.normalize()

    daily = df.groupby("date").agg(
        pm25=("pm25", "mean"),
        pm25_max=("pm25", "max"),
        pm25_hours=("pm25", "count"),
        temp=("temp", "mean"),
        temp_min=("temp", "min"),
        humidity=("humidity", "mean"),
        wind=("wind", "mean"),
        pressure=("pressure", "mean"),
        precipitation=("precipitation", "sum"),
        blh=("blh", "mean"),
        blh_min=("blh", "min"),
        cams_pm25=("cams_pm25", "mean"),
    ).reset_index()

    daily.loc[daily.pm25_hours < MIN_HOURS, "pm25"] = None

    # night is when inversions happen, so evening/night weather separately
    night = df[(df.time.dt.hour >= 18) | (df.time.dt.hour < 6)]
    night = night.groupby("date").agg(blh_night=("blh", "mean"), wind_night=("wind", "mean"))
    daily = daily.merge(night, on="date", how="left")
    return daily


def make_features(df=None):
    if df is None:
        df = load_hourly()
    daily = make_daily(df)

    # --- lags. "1 hour ago" = value at 23:00 today, the last thing we know at midnight
    pm = df.set_index("time").pm25
    last_hour = pm[pm.index.hour == 23]
    last_hour.index = last_hour.index.normalize()
    daily["pm25_lag1h"] = daily.date.map(last_hour)
    daily["pm25_lag24h"] = daily.date.map(last_hour.shift(1, freq="D"))
    daily["pm25_lag48h"] = daily.date.map(last_hour.shift(2, freq="D"))
    daily["pm25_mean24h"] = daily.pm25  # mean of today = mean of last 24 hours
    daily["pm25_yesterday"] = daily.pm25.shift(1)
    daily["pm25_mean7d"] = daily.pm25.rolling(7, min_periods=4).mean()

    daily = add_calendar(daily)
    daily = add_calendar(daily, prefix="tmr_")

    # --- weather tomorrow (as a "forecast")
    for col in WEATHER_COLS + ["temp_min", "blh_min", "blh_night", "wind_night"]:
        daily["tmr_" + col] = daily[col].shift(-1)

    # weather change: cold front coming tomorrow often cleans the air
    daily["tmr_temp_change"] = daily.tmr_temp - daily.temp
    daily["tmr_pressure_change"] = daily.tmr_pressure - daily.pressure

    # --- target: mean PM2.5 tomorrow. And CAMS for tomorrow as the thing to beat
    daily["target"] = daily.pm25.shift(-1)
    daily["cams_tomorrow"] = daily.cams_pm25.shift(-1)

    # shift(-1) only works if days go one after another without gaps
    assert daily.date.diff().dropna().eq(pd.Timedelta(days=1)).all()
    return daily


def feature_columns(daily):
    # everything except the answer, CAMS (benchmark, not a feature) and helper columns
    # pm25 is the same as pm25_mean24h, no need to have it twice
    drop = ["date", "target", "cams_tomorrow", "cams_pm25", "pm25_hours", "pm25"]
    return [c for c in daily.columns if c not in drop]


def split_by_time(daily):
    train = daily[daily.date < TRAIN_END]
    val = daily[(daily.date >= TRAIN_END) & (daily.date < VAL_END)]
    test = daily[(daily.date >= VAL_END) & (daily.date < EMBASSY_END)]
    return train, val, test


def airgradient_part(daily):
    # days after the embassy stopped, PM2.5 comes from AirGradient sensors (see download_airgradient.py)
    return daily[daily.date >= EMBASSY_END]


if __name__ == "__main__":
    daily = make_features()
    daily = daily.dropna(subset=["target"])
    cols = feature_columns(daily)
    print(daily.shape)
    print(len(cols), "features:", cols)

    train, val, test = split_by_time(daily)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"{name}: {len(part)} days, {part.date.min().date()} -> {part.date.max().date()}")

    # leakage check: today's features should not be suspiciously close to the target.
    # if some feature has corr ~1.0 with target, something from the future got in
    corr = daily[cols + ["target"]].corr(numeric_only=True)["target"].drop("target")
    print("\ntop correlations with target:")
    print(corr.abs().sort_values(ascending=False).head(8).round(3))
