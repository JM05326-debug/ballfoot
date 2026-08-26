"""
嚴格時間序切分（禁止把 random train/test split 當作主要驗證方式）。

- chronological_season_split(): 依「完整賽季」切成 Train / Validation / Test。
  Test = 最近一個已經完整結束的賽季，Validation = 再前一個賽季，
  Train = 更早的全部賽季。目前賽季（仍在進行中）永遠不會被放進這三者，
  保留給 predict.py 對「還沒開踢」的比賽做真正的預測，這樣才不會有任何
  「用當下已知結果去驗證同一個賽季」的疑慮。
- walk_forward_season_splits(): 供第十二階段 Backtesting 使用的 rolling /
  walk-forward 產生器——每次往前推進一個賽季，train 永遠只用「該賽季開踢前」
  的所有資料，完全模擬「如果我站在過去某一天，只用當時已知資料」的情境。

哪個賽季屬於 Train/Validation/Test 是依「目前日期」動態算出來的（見 config.py），
不會寫死特定年份，賽季往前推進時，切分點會自動跟著往前移動。
"""

from __future__ import annotations

import pandas as pd

from . import config


def completed_seasons(dataset: pd.DataFrame) -> list[int]:
    """回傳資料集中，所有「已經結束」的賽季起始年（不含目前進行中的賽季）。"""
    current = config.current_season_start_year()
    seasons = sorted(int(s) for s in dataset["SeasonStartYear"].unique())
    return [s for s in seasons if s < current]


def chronological_season_split(
    dataset: pd.DataFrame,
    n_val_seasons: int = 1,
    n_test_seasons: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    completed = completed_seasons(dataset)
    min_needed = n_val_seasons + n_test_seasons + 1
    if len(completed) < min_needed:
        raise ValueError(
            f"已結束的賽季只有 {len(completed)} 個（{completed}），"
            f"至少需要 {min_needed} 個賽季才能切出 Train/Validation/Test"
        )

    test_seasons = completed[-n_test_seasons:]
    val_seasons = completed[-(n_test_seasons + n_val_seasons):-n_test_seasons]
    train_seasons = completed[: -(n_test_seasons + n_val_seasons)]

    train = dataset[dataset["SeasonStartYear"].isin(train_seasons)].sort_values("Date").reset_index(drop=True)
    val = dataset[dataset["SeasonStartYear"].isin(val_seasons)].sort_values("Date").reset_index(drop=True)
    test = dataset[dataset["SeasonStartYear"].isin(test_seasons)].sort_values("Date").reset_index(drop=True)

    excluded_current = sorted(set(dataset["SeasonStartYear"].unique()) - set(completed))

    info = {
        "train_seasons": [config.season_label(y) for y in train_seasons],
        "val_seasons": [config.season_label(y) for y in val_seasons],
        "test_seasons": [config.season_label(y) for y in test_seasons],
        "excluded_current_season": [config.season_label(y) for y in excluded_current],
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
    }
    return train, val, test, info


def walk_forward_season_splits(dataset: pd.DataFrame, min_train_seasons: int = 3):
    """逐賽季往前推進的 walk-forward 切分產生器，供 Backtesting 使用。

    每一次 yield (train_df, test_df, info)，train 只包含 test 賽季開踢之前的
    所有已結束賽季資料，完全不會看到 test 賽季（甚至更晚）的任何資訊。
    """
    completed = completed_seasons(dataset)
    for i in range(min_train_seasons, len(completed)):
        train_seasons = completed[:i]
        test_season = completed[i]

        train = dataset[dataset["SeasonStartYear"].isin(train_seasons)].sort_values("Date").reset_index(drop=True)
        test = dataset[dataset["SeasonStartYear"] == test_season].sort_values("Date").reset_index(drop=True)

        info = {
            "train_seasons": [config.season_label(y) for y in train_seasons],
            "test_season": config.season_label(test_season),
            "train_rows": len(train),
            "test_rows": len(test),
        }
        yield train, test, info
