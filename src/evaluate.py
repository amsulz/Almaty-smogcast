"""Metrics and the results table. Every training script writes its line into results/metrics.csv"""
import numpy as np
import pandas as pd

from config import RESULTS_DIR

METRICS_FILE = RESULTS_DIR / "metrics.csv"

# US EPA breakpoints (2024 version) squeezed into 3 groups for the demo
CATEGORIES = ["low", "high", "very high"]


def category(pm25):
    return pd.cut(pm25, bins=[-1, 35.4, 125.4, 10000], labels=CATEGORIES)


def scores(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    acc = (category(pd.Series(y_true)) == category(pd.Series(y_pred))).mean()
    return {
        "mae": round(float(np.mean(np.abs(y_true - y_pred))), 2),
        "rmse": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 2),
        "category_acc": round(float(acc), 3),
        "n_days": int(ok.sum()),
    }


def save_metrics(model, split, y_true, y_pred, note=""):
    row = {"model": model, "split": split, **scores(y_true, y_pred), "note": note}
    print(row)

    if METRICS_FILE.exists():
        table = pd.read_csv(METRICS_FILE)
        # replace old result of the same model, so re-running doesn't make duplicates
        table = table[~((table.model == model) & (table.split == split))]
        table = pd.concat([table, pd.DataFrame([row])])
    else:
        table = pd.DataFrame([row])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(METRICS_FILE, index=False)


def save_predictions(dates, y_pred, name):
    """Keep test predictions of every model in one file, handy for plots later."""
    file = RESULTS_DIR / "predictions_test.csv"
    new = pd.DataFrame({"date": pd.to_datetime(dates).values, name: y_pred})
    if file.exists():
        old = pd.read_csv(file, parse_dates=["date"])
        old = old.drop(columns=[name], errors="ignore")
        new = old.merge(new, on="date", how="outer")
    new.to_csv(file, index=False)
