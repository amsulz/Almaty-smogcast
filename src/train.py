"""Baseline, CAMS, linear regression, LightGBM and LightGBM with AirCast weights. Results go to results/metrics.csv"""
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import MODELS_DIR
from evaluate import save_metrics, save_predictions
from features import airgradient_part, feature_columns, make_features, split_by_time

LGB_PARAMS = {
    "objective": "regression_l1",  # MAE, smog has big spikes and L2 chases them too much
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}


# Without weights the model pulls predictions toward common values and underpredicts heavy smog.
# This gives rare days a bigger penalty for errors, so the model can't ignore them.
def rarity_weights(target, bin_size=10, power=0.1):
    bins = (target // bin_size).astype(int)    # bin number, e.g. 23 // 10 = 2 (20-29 µg), 105 // 10 = 10 (100-109 µg)
    counts = bins.map(bins.value_counts())     # how often each bin happens: bin 2 is very common, bin 10 is very rare
    w = (1 / counts) ** power                  # rare bin = bigger penalty for errors, power 0.1 makes it softer
    return w / w.mean()                        # average weight = 1


def main():
    daily = make_features().dropna(subset=["target"])
    cols = feature_columns(daily)
    train, val, test = split_by_time(daily)
    print(f"train {len(train)}, val {len(val)}, test {len(test)} days, {len(cols)} features")

    save_predictions(test.date, test.target.values, "actual")

    # 1. baseline: tomorrow = today
    for split, part in [("val", val), ("test", test)]:
        save_metrics("baseline (tomorrow = today)", split, part.target, part.pm25)
    save_predictions(test.date, test.pm25.values, "baseline")

    # 2. CAMS global forecast
    for split, part in [("val", val), ("test", test)]:
        save_metrics("CAMS", split, part.target, part.cams_tomorrow)
    save_predictions(test.date, test.cams_tomorrow.values, "cams")

    # 3. linear regression. it can't handle NaN, so fill with the median from train
    linear = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
    linear.fit(train[cols], train.target)
    save_metrics("linear regression", "val", val.target, linear.predict(val[cols]))

    # for the test, retrain on train + val, same as LightGBM below, so the comparison is fair
    train_val = daily[daily.date < test.date.min()]
    linear.fit(train_val[cols], train_val.target)
    pred_linear = linear.predict(test[cols])
    save_metrics("linear regression", "test", test.target, pred_linear, note="trained on train+val")
    save_predictions(test.date, pred_linear, "linear")

    # 4. LightGBM. early stopping on val to pick the number of trees
    model = lgb.train(
        LGB_PARAMS,
        lgb.Dataset(train[cols], train.target),
        num_boost_round=3000,
        valid_sets=[lgb.Dataset(val[cols], val.target)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    best = model.best_iteration
    print(f"LightGBM best iteration: {best}")
    save_metrics("LightGBM", "val", val.target, model.predict(val[cols], num_iteration=best))

    # for the test, retrain on train + val with the same number of trees.
    # 2024 is the closest year to 2025, it would be a waste not to use it
    final = lgb.train(LGB_PARAMS, lgb.Dataset(train_val[cols], train_val.target), num_boost_round=best)
    pred = final.predict(test[cols])
    save_metrics("LightGBM", "test", test.target, pred, note="trained on train+val")
    save_predictions(test.date, pred, "lightgbm")

    MODELS_DIR.mkdir(exist_ok=True)
    final.save_model(str(MODELS_DIR / "lgbm.txt"))
    print(f"saved model to {MODELS_DIR / 'lgbm.txt'}")

    # 5. LightGBM + AirCast weights. The Telegram bot uses this one
    weighted = lgb.train(
        LGB_PARAMS,
        lgb.Dataset(train[cols], train.target, weight=rarity_weights(train.target)),
        num_boost_round=3000,
        valid_sets=[lgb.Dataset(val[cols], val.target)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    best_w = weighted.best_iteration
    save_metrics("LightGBM + AirCast weights", "val", val.target, weighted.predict(val[cols], num_iteration=best_w))
    final_w = lgb.train(LGB_PARAMS, lgb.Dataset(train_val[cols], train_val.target,
                                                weight=rarity_weights(train_val.target)),
                        num_boost_round=best_w)
    pred_w = final_w.predict(test[cols])
    save_metrics("LightGBM + AirCast weights", "test", test.target, pred_w, note="trained on train+val")
    save_predictions(test.date, pred_w, "lightgbm_weighted")
    final_w.save_model(str(MODELS_DIR / "lgbm_weighted.txt"))

    # 6. one more check on days after the embassy stopped. The model never saw them,
    # but the "real" PM2.5 here is from cheap AirGradient sensors, so it's less exact
    new = airgradient_part(daily)
    print(f"\nAirGradient days: {len(new)}")
    if new.empty:
        # no pm25_airgradient.csv yet (run download_airgradient.py), e.g. in the GitHub bot run
        return
    note = "AirGradient sensors, scaled to the embassy"
    save_metrics("baseline (tomorrow = today)", "airgradient", new.target, new.pm25, note=note)
    save_metrics("CAMS", "airgradient", new.target, new.cams_tomorrow, note=note)
    save_metrics("linear regression", "airgradient", new.target, linear.predict(new[cols]), note=note)
    save_metrics("LightGBM", "airgradient", new.target, final.predict(new[cols]), note=note)
    save_metrics("LightGBM + AirCast weights", "airgradient", new.target, final_w.predict(new[cols]), note=note)


if __name__ == "__main__":
    main()
