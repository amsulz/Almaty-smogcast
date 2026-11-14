"""Check the models on four years, one by one, and with several random seeds.

For every test year the model only learns from the days before it:
the last year before the test is validation (early stopping), everything earlier is train.
2026 is the AirGradient period, the other years are the embassy station.
Results go to results/backtest.csv

  python src/backtest.py        # 3 seeds, about a minute
  python src/backtest.py 30     # 30 seeds, slower
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import EMBASSY_END, RESULTS_DIR
from features import airgradient_part, feature_columns, make_features
from train import LGB_PARAMS, rarity_weights


def fit_predict(train, val, test, cols, log, weights, seed):
    # same steps as in train.py: early stopping on val, then train + val with that number of trees
    params = dict(LGB_PARAMS, seed=seed)
    y = np.log1p if log else (lambda s: s)
    w = rarity_weights(train.target) if weights else None
    model = lgb.train(params, lgb.Dataset(train[cols], y(train.target), weight=w), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(val[cols], y(val.target))],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    both = pd.concat([train, val])
    w = rarity_weights(both.target) if weights else None
    final = lgb.train(params, lgb.Dataset(both[cols], y(both.target), weight=w),
                      num_boost_round=model.best_iteration)
    pred = final.predict(test[cols])
    return np.expm1(pred) if log else pred


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    daily = make_features().dropna(subset=["target"])
    embassy = daily[daily.date < EMBASSY_END]
    old_cols = feature_columns(daily)
    new_cols = feature_columns(daily, inversion=True)
    # name: (columns, log target, AirCast weights)
    models = {
        "LightGBM": (old_cols, False, False),
        "LightGBM + AirCast weights": (old_cols, False, True),
        "LightGBM + inversion + log": (new_cols, True, False),
    }

    rows = []
    for year in [2023, 2024, 2025, 2026]:
        train = embassy[embassy.date < f"{year - 1}-01-01"]
        val = embassy[(embassy.date >= f"{year - 1}-01-01") & (embassy.date < f"{year}-01-01")]
        if year == 2026:
            test = airgradient_part(daily)
        else:
            test = embassy[(embassy.date >= f"{year}-01-01") & (embassy.date < f"{year + 1}-01-01")]
        real = test.target.values
        dirty = real >= np.quantile(real, 0.9)  # 10% most polluted days of this year

        known = ~np.isnan(test.pm25.values)  # "tomorrow = today" needs today's PM2.5
        rows.append({"year": year, "model": "baseline (tomorrow = today)", "days": len(test),
                     "mae": np.abs(real - test.pm25.values)[known].mean(),
                     "mae_dirty": np.abs(real - test.pm25.values)[known & dirty].mean()})
        for name, (cols, log, weights) in models.items():
            mae, mae_dirty = [], []
            for seed in range(seeds):
                err = np.abs(real - fit_predict(train, val, test, cols, log, weights, seed))
                mae.append(err.mean())
                mae_dirty.append(err[dirty].mean())
            # mean over the seeds, and the best and the worst run to see how much luck there is
            rows.append({"year": year, "model": name, "days": len(test), "mae": np.mean(mae),
                         "mae_min": np.min(mae), "mae_max": np.max(mae), "mae_dirty": np.mean(mae_dirty)})
        print(f"{year} done: train {len(train)}, val {len(val)}, test {len(test)} days")

    table = pd.DataFrame(rows).round(2)
    table["seeds"] = seeds
    table.to_csv(RESULTS_DIR / "backtest.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
