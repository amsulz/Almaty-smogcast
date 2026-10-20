"""Idea from AirCast (MBZUAI, 2025): give more weight to rare, very polluted days."""
import lightgbm as lgb
import pandas as pd

from features import feature_columns, make_features, split_by_time
from train import LGB_PARAMS, rarity_weights


def errors(part, pred):
    err = (part.target - pred).abs()
    dirty = part.target >= part.target.quantile(0.9)
    return err.mean(), err[dirty].mean()


def main():
    daily = make_features().dropna(subset=["target"])
    cols = feature_columns(daily)
    train, val, test = split_by_time(daily)
    train_val = daily[daily.date < test.date.min()]

    rows = []
    for power in [0, 0.1, 0.2, 0.3, 0.5]:
        w = None if power == 0 else rarity_weights(train.target, power=power)
        model = lgb.train(
            LGB_PARAMS,
            lgb.Dataset(train[cols], train.target, weight=w),
            num_boost_round=3000,
            valid_sets=[lgb.Dataset(val[cols], val.target)],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        best = model.best_iteration
        val_all, val_dirty = errors(val, model.predict(val[cols], num_iteration=best))

        w = None if power == 0 else rarity_weights(train_val.target, power=power)
        final = lgb.train(LGB_PARAMS, lgb.Dataset(train_val[cols], train_val.target, weight=w),
                          num_boost_round=best)
        test_all, test_dirty = errors(test, final.predict(test[cols]))
        rows.append({"power": power, "trees": best, "val all": val_all, "val 10% dirtiest": val_dirty,
                     "test all": test_all, "test 10% dirtiest": test_dirty})

    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
