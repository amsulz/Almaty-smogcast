"""Archive of the bot's forecasts: results/live_forecasts.csv

Every run adds the forecast for tomorrow. The real value (AAI, Medeu) is added later,
when that day is over, so the table shows how the bot does live.
"""
import pandas as pd

from config import RAW_DIR, RESULTS_DIR

ARCHIVE = RESULTS_DIR / "live_forecasts.csv"
COLUMNS = ["date", "forecast", "model", "bias", "actual", "error"]


def load():
    if ARCHIVE.exists():
        return pd.read_csv(ARCHIVE, dtype={"date": str})
    return pd.DataFrame(columns=COLUMNS)


def add_actual(df):
    aai = RAW_DIR / "pm25_aai_medeu.csv"
    if not aai.exists():
        return df
    real = pd.read_csv(aai, dtype={"date": str}).set_index("date").pm25
    for col in ["forecast", "model", "bias", "actual"]:
        df[col] = pd.to_numeric(df[col])
    # today is not over yet, its mean would be too early
    today = pd.Timestamp.now(tz="Asia/Almaty").strftime("%Y-%m-%d")
    done = df.date < today
    df.loc[done, "actual"] = df.loc[done, "date"].map(real).round(1)
    df["error"] = (df.forecast - df.actual).round(1)
    return df


def save(day, forecast, model, bias):
    df = load()
    # a second run for the same day (e.g. started by hand) replaces the first one
    df = df[df.date != f"{day:%Y-%m-%d}"]
    row = {"date": f"{day:%Y-%m-%d}", "forecast": round(forecast), "model": round(model, 1), "bias": round(bias, 1)}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = add_actual(df.sort_values("date"))
    df[COLUMNS].to_csv(ARCHIVE, index=False)
    known = df.dropna(subset=["actual"])
    if len(known):
        print(f"archive: {len(df)} days, MAE so far {(known.forecast - known.actual).abs().mean():.1f}")
