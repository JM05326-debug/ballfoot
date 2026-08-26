#!/usr/bin/env python
"""
第五階段主要入口：Backtesting——用 walk-forward 驗證「這套方法論」在最近
1 個賽季 / 2 個賽季 / 3 個賽季的穩定表現。

用法：
    python evaluate.py

沿用最新一個 models/model_vNNN/ 版本的 Ensemble 權重與 Platt 校準器（不重新配適），
只重新訓練 ML 模型 + Poisson（因為每個 walk-forward fold 的訓練窗都不一樣，
底層模型一定要重新訓練；但「方法論」——要用哪些模型、權重怎麼分配、怎麼校準——維持不變）。

輸出：
  - 每個 fold（每個賽季）的 Accuracy / Log Loss / Brier / Calibration / 各類別 Accuracy
  - 合併最近 1 / 2 / 3 個賽季後的整體指標
  - 存成 models/model_vNNN/backtest_report.json
"""

from __future__ import annotations

import json
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import pandas as pd

from src import backtest, config


def find_latest_model_dir():
    existing = sorted(config.MODELS_DIR.glob("model_v*"))
    if not existing:
        raise FileNotFoundError("找不到任何 models/model_vNNN/，請先執行 python train.py")
    return existing[-1]


def main():
    model_dir = find_latest_model_dir()
    print(f"使用模型版本: {model_dir.name}")

    ensemble_weights = json.loads((model_dir / "ensemble_weights.json").read_text(encoding="utf-8"))
    feature_cols = json.loads((model_dir / "feature_columns.json").read_text(encoding="utf-8"))
    calibrator_path = model_dir / "calibrator_platt.joblib"
    calibrator = joblib.load(calibrator_path) if calibrator_path.exists() else None

    dataset_path = config.PROCESSED_DIR / "training_dataset.csv"
    fuzzy_path = config.PROCESSED_DIR / "fuzzy_outputs.csv"
    matches_path = config.PROCESSED_DIR / "matches_clean.csv"
    dataset = pd.read_csv(dataset_path, parse_dates=["Date"])
    fuzzy = pd.read_csv(fuzzy_path)
    matches_clean = pd.read_csv(matches_path, parse_dates=["Date"])
    dataset = dataset.merge(fuzzy, on="MatchID", how="left")

    print("\n" + "=" * 70)
    print("Walk-forward Backtest（逐賽季往前滾動，每輪只用該賽季開踢前的資料訓練）")
    print("=" * 70)
    fold_results = backtest.run_walk_forward_backtest(
        dataset, matches_clean, feature_cols, ensemble_weights, calibrator=calibrator, max_test_seasons=3
    )

    print("\n" + "=" * 70)
    print("彙總結果：最近 1 / 2 / 3 個賽季（誠實回報所有指標，不是只給一個勝率）")
    print("=" * 70)
    summary = {}
    for n in (1, 2, 3):
        if n > len(fold_results):
            print(f"  最近 {n} 個賽季：資料不足（只有 {len(fold_results)} 個 fold 可用），跳過")
            continue
        agg = backtest.aggregate_last_n_seasons(fold_results, n)
        summary[f"last_{n}_season(s)"] = agg
        print(f"\n  最近 {n} 個賽季 ({agg['seasons_included']})：")
        print(f"    Accuracy      : {agg['accuracy']:.4f}")
        print(f"    Log Loss      : {agg['log_loss']:.4f}")
        print(f"    Brier Score   : {agg['brier_score']:.4f}")
        print(f"    ROC-AUC(ovr)  : {agg['roc_auc_ovr_macro']}")
        print(f"    ECE(主勝校準) : {agg['ece_home_win']:.4f}")
        print(f"    主勝 Accuracy : {agg['per_class_accuracy']['H']:.4f}")
        print(f"    和局 Accuracy : {agg['per_class_accuracy']['D']:.4f}")
        print(f"    客勝 Accuracy : {agg['per_class_accuracy']['A']:.4f}")
        print(f"    樣本數        : {agg['n_samples']}")

    per_fold_summary = [
        {
            "test_season": f["test_season"],
            "train_seasons": f["train_seasons"],
            "n_matches": f["n_matches"],
            "report": f["report"],
        }
        for f in fold_results
    ]
    out = {"per_fold": per_fold_summary, "aggregated": summary}
    out_path = model_dir / "backtest_report.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已儲存: {out_path}")


if __name__ == "__main__":
    main()
