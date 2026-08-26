"""
Backtesting：模擬「如果我站在過去某一天，只用當時已知資料，模型會怎麼預測？」

用 time_split.walk_forward_season_splits() 逐賽季往前滾動：每一輪只用「這個賽季開踢前」
的所有歷史資料重新訓練 ML 模型 + Poisson，預測該賽季全部比賽，然後訓練窗往前推一個賽季。

Ensemble 權重沿用 train.py（第四階段）用 2024-25 Validation 資料算出來的固定權重，
**不在每一輪 backtest fold 裡重新配適權重**——這裡驗證的是「一套已經定案的方法論
（模型組合 + 固定權重 + 固定校準器）」在其他賽季上是否穩定，而不是每個賽季重新調一次
參數（那樣每一輪都變成用該賽季自己的結果去挑最好的權重，會失去 backtest 的意義）。

Fuzzy 的隸屬函數/規則庫從頭到尾都是固定的常數，本來就不需要、也不會重新訓練。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import ensemble as ensemble_mod
from . import evaluation, ml_models, poisson_model, time_split


def run_walk_forward_backtest(
    dataset: pd.DataFrame,
    matches_clean: pd.DataFrame,
    feature_cols: list[str],
    ensemble_weights: dict[str, float],
    calibrator=None,
    min_train_seasons: int | None = None,
    max_test_seasons: int = 3,
) -> list[dict]:
    completed = time_split.completed_seasons(dataset)
    if min_train_seasons is None:
        min_train_seasons = max(3, len(completed) - max_test_seasons)

    fold_results = []
    for train_df, test_df, info in time_split.walk_forward_season_splits(dataset, min_train_seasons=min_train_seasons):
        X_train, y_train = ml_models.prepare_xy(train_df, feature_cols)
        X_test, y_test = ml_models.prepare_xy(test_df, feature_cols)

        proba_store: dict[str, np.ndarray] = {}
        registry = ml_models.build_model_registry()
        for name, model in registry.items():
            model.fit(X_train, y_train)
            proba_store[name] = ml_models.predict_proba_aligned(model, X_test)

        train_years = [int(s.split("-")[0]) for s in info["train_seasons"]]
        train_matches = matches_clean[matches_clean["SeasonStartYear"].isin(train_years)]
        dc_model = poisson_model.fit_dixon_coles(train_matches)
        poisson_preds = poisson_model.predict_matches_df(dc_model, test_df[["MatchID", "HomeTeam", "AwayTeam"]])
        merged = test_df[["MatchID"]].merge(poisson_preds, on="MatchID", how="left")
        proba_store["Poisson"] = merged[["Poisson_P_HomeWin", "Poisson_P_Draw", "Poisson_P_AwayWin"]].to_numpy(dtype=float)

        proba_store["Fuzzy"] = test_df[["Fuzzy_P_HomeWin", "Fuzzy_P_Draw", "Fuzzy_P_AwayWin"]].to_numpy(dtype=float)

        members = [n for n in ensemble_weights if n in proba_store]
        w_sum = sum(ensemble_weights[n] for n in members)
        w = {n: ensemble_weights[n] / w_sum for n in members}
        ensemble_proba = ensemble_mod.weighted_ensemble_proba({n: proba_store[n] for n in members}, w)

        if calibrator is not None:
            from . import calibration
            ensemble_proba = calibration.apply_calibrator(calibrator, ensemble_proba)

        report = evaluation.full_report(y_test, ensemble_proba)
        fold_results.append({
            "test_season": info["test_season"],
            "train_seasons": info["train_seasons"],
            "n_matches": len(test_df),
            "y_true_idx": y_test,
            "proba": ensemble_proba,
            "report": report,
        })
        print(f"  完成 fold: train={info['train_seasons']} -> test={info['test_season']} "
              f"({len(test_df)} 場) accuracy={report['accuracy']:.3f} log_loss={report['log_loss']:.3f}")

    return fold_results


def aggregate_last_n_seasons(fold_results: list[dict], n: int) -> dict:
    """把最近 n 個賽季（依 fold_results 順序，最後 n 筆）的比賽『合併』後重新計算指標，
    而不是單純平均每個賽季的指標——賽季間比賽數可能因延賽而略有差異，合併後計算更準確。"""
    recent = fold_results[-n:]
    y_true = np.concatenate([f["y_true_idx"] for f in recent])
    proba = np.concatenate([f["proba"] for f in recent], axis=0)
    report = evaluation.full_report(y_true, proba)
    report["seasons_included"] = [f["test_season"] for f in recent]
    return report
