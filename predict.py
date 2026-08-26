#!/usr/bin/env python
"""
第五階段主要入口：對「下一輪」還沒開踢的比賽產生完整預測。

用法：
    python predict.py                  # 預測目前賽季下一個還沒開踢的輪次
    python predict.py --round 5        # 指定輪次
    python predict.py --season 2026    # 指定賽季起始年（預設=目前賽季）

一次執行即可完成：載入最新已定案的方法論（Ensemble 權重 + 校準器）-> 用「目前所有
已知真實資料」重新訓練正式上線模型 -> 對下一輪比賽產生預測 -> 存檔 -> 結束。
不依賴背景常駐程式或排程，電腦關機重開後隨時可以手動重新執行。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd

from src import config, feature_engineering, prediction


def find_latest_model_dir():
    existing = sorted(config.MODELS_DIR.glob("model_v*"))
    if not existing:
        raise FileNotFoundError("找不到任何 models/model_vNNN/，請先執行 python train.py")
    return existing[-1]


def main():
    parser = argparse.ArgumentParser(description="對下一輪還沒開踢的英超比賽產生完整預測")
    parser.add_argument("--season", type=int, default=None, help="賽季起始年，預設為目前賽季")
    parser.add_argument("--round", type=int, default=None, help="指定輪次，預設為下一個還沒開踢的輪次")
    args = parser.parse_args()

    season_start_year = args.season or config.current_season_start_year()

    model_dir = find_latest_model_dir()
    print(f"使用方法論版本: {model_dir.name}（Ensemble 權重 + 校準器沿用此版本，模型本身會用最新資料重新訓練）")

    ensemble_weights = json.loads((model_dir / "ensemble_weights.json").read_text(encoding="utf-8"))
    feature_cols = json.loads((model_dir / "feature_columns.json").read_text(encoding="utf-8"))
    calibrator = joblib.load(model_dir / "calibrator_platt.joblib")

    matches_path = config.PROCESSED_DIR / "matches_clean.csv"
    fixtures_path = config.PROCESSED_DIR / "fixtures_full.csv"
    dataset_path = config.PROCESSED_DIR / "training_dataset.csv"
    fuzzy_path = config.PROCESSED_DIR / "fuzzy_outputs.csv"
    elo_path = config.PROCESSED_DIR / "elo_ratings_latest.json"
    for p in (matches_path, fixtures_path, dataset_path, fuzzy_path, elo_path):
        if not p.exists():
            print(f"DATA SOURCE ERROR: 找不到 {p}，請先執行 update_data.py 以及 python -m src.fuzzy_model", file=sys.stderr)
            sys.exit(1)

    matches_clean = pd.read_csv(matches_path, parse_dates=["Date"])
    fixtures_full = pd.read_csv(fixtures_path, parse_dates=["Date"])
    dataset = pd.read_csv(dataset_path, parse_dates=["Date"])
    fuzzy = pd.read_csv(fuzzy_path)
    dataset = dataset.merge(fuzzy, on="MatchID", how="left")
    elo_ratings = json.loads(elo_path.read_text(encoding="utf-8"))["ratings"]

    round_number = args.round or prediction.get_next_unplayed_round(fixtures_full, season_start_year)
    if round_number is None:
        print(f"賽季 {config.season_label(season_start_year)} 目前沒有還沒開踢的場次，無需預測。")
        return

    upcoming = prediction.build_upcoming_matches(fixtures_full, season_start_year, round_number)
    if upcoming.empty:
        print(f"賽季 {config.season_label(season_start_year)} 第 {round_number} 輪找不到賽程資料。")
        return

    print(f"\n預測目標: {config.season_label(season_start_year)} 第 {round_number} 輪，共 {len(upcoming)} 場比賽")

    print("\n" + "=" * 70)
    print("用「目前所有已知真實資料」重新訓練正式上線模型")
    print("=" * 70)
    fitted_models = prediction.fit_production_models(dataset, feature_cols)
    print(f"  已訓練: {list(fitted_models.keys())}")
    dc_model = prediction.fit_production_poisson(matches_clean)
    print(f"  Poisson(Dixon-Coles) 已用 {dc_model.n_matches_used} 場歷史比賽重新估計（截至 {dc_model.as_of_date}）")

    rf_model = fitted_models.get("RandomForest")
    if rf_model is not None:
        importances = rf_model.named_steps["clf"].feature_importances_
        rf_importances = dict(zip(feature_cols, importances))
    else:
        rf_importances = {}
    feature_std = dataset[feature_cols].std().to_dict()

    print("\n" + "=" * 70)
    print("計算賽前特徵（只用目前為止已經真實發生的比賽，無洩漏）")
    print("=" * 70)
    upcoming_features = feature_engineering.build_features_for_upcoming(upcoming, matches_clean, elo_ratings)

    predictions = []
    for _, match_row in upcoming.iterrows():
        feature_row = upcoming_features[upcoming_features["MatchID"] == match_row["MatchID"]].iloc[0]
        pred = prediction.predict_one_match(
            match_row, feature_row, feature_cols, fitted_models, dc_model,
            ensemble_weights, calibrator, rf_importances, feature_std,
        )
        predictions.append(pred)

    print("\n" + "=" * 70)
    print(f"{config.season_label(season_start_year)} 第 {round_number} 輪 預測結果")
    print("=" * 70)
    for pred in predictions:
        print()
        print(prediction.format_match_report(pred))

    out_dir = config.PREDICTIONS_DIR
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    season_label_str = config.season_label(season_start_year)
    json_path = out_dir / f"predictions_{season_label_str}_R{round_number}_{timestamp}.json"
    csv_path = out_dir / f"predictions_{season_label_str}_R{round_number}_{timestamp}.csv"

    json_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")

    flat_rows = []
    for p in predictions:
        flat_rows.append({
            "date": p["date"], "round": p["round"],
            "home_team": p["home_team"], "away_team": p["away_team"],
            "p_home_win": p["p_home_win"], "p_draw": p["p_draw"], "p_away_win": p["p_away_win"],
            "predicted_score": p["predicted_score"],
            "expected_home_goals": p["expected_home_goals"], "expected_away_goals": p["expected_away_goals"],
            "p_over_2_5": p["p_over_2_5"], "p_under_2_5": p["p_under_2_5"],
            "p_btts_yes": p["p_btts_yes"], "p_btts_no": p["p_btts_no"],
            "confidence": p["confidence"],
        })
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)

    print(f"\n已儲存: {json_path.name}, {csv_path.name}（於 {out_dir}）")


if __name__ == "__main__":
    main()
