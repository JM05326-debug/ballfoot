#!/usr/bin/env python
"""
比賽結束後回收預測結果：把之前 predict.py 存的預測，跟 update_data.py 抓到的
真實比分做比對，計算每場的 Brier Score / Log Loss / 有沒有猜對勝負，
累積寫進 data/predictions/prediction_history.csv。

對應規格第十三節的迴圈：
    Prediction -> Actual Result -> Error -> Save -> Update Dataset -> Retrain
本程式負責「Prediction -> Actual Result -> Error -> Save」這一段；
「Update Dataset -> Retrain」由 update_data.py + train.py（GitHub Actions 每週執行）負責。

用法：
    python collect_results.py

規則：
  - 掃描 data/predictions/predictions_*.json（每一輪常常因為一天四次排程被
    重複預測很多次；同一場比賽只取「最後一次、最接近賽前」的那次預測來計分，
    因為那個版本用到的近期資料最新）
  - 跟 data/processed/matches_clean.csv 的真實結果比對（用主隊/客隊/日期比對，
    不看時間，避免賽程小幅調整時間造成配對失敗）
  - 已經記錄過的比賽不會重複寫入，可以放心每天重複執行
  - 沒有真實結果可以比對的比賽（還沒開踢）直接跳過，不會假裝有結果
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from src import config, evaluation

HISTORY_COLUMNS = [
    "prediction_time", "match_date", "home_team", "away_team",
    "predicted_home", "predicted_draw", "predicted_away", "predicted_result",
    "predicted_score", "actual_home", "actual_away", "actual_result", "correct",
    "brier_score", "log_loss", "model_version",
]

FILENAME_RE = re.compile(r"predictions_.+_R\d+_(\d{8}_\d{6})\.json$")


def _parse_prediction_time(path) -> str | None:
    m = FILENAME_RE.search(path.name)
    if not m:
        return None
    dt = _dt.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    return dt.isoformat(timespec="seconds")


def load_latest_predictions_per_match() -> dict[tuple[str, str, str], dict]:
    """掃描所有時間戳記的預測檔，同一場比賽（主隊/客隊/日期）只留最後一次預測。"""
    latest: dict[tuple[str, str, str], dict] = {}
    files = sorted(config.PREDICTIONS_DIR.glob("predictions_*.json"))
    for f in files:
        pred_time = _parse_prediction_time(f)
        if pred_time is None:
            continue
        try:
            records = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"警告：{f.name} 不是合法的 JSON，略過", file=sys.stderr)
            continue

        for rec in records:
            date_only = str(rec["date"]).split(" ")[0]
            key = (rec["home_team"], rec["away_team"], date_only)
            rec_with_time = {**rec, "_prediction_time": pred_time, "_match_date": date_only}
            existing = latest.get(key)
            if existing is None or rec_with_time["_prediction_time"] > existing["_prediction_time"]:
                latest[key] = rec_with_time
    return latest


def load_actual_results() -> dict[tuple[str, str, str], dict]:
    matches_path = config.PROCESSED_DIR / "matches_clean.csv"
    if not matches_path.exists():
        raise FileNotFoundError(f"找不到 {matches_path}，請先執行 update_data.py")
    matches = pd.read_csv(matches_path, parse_dates=["Date"])

    actuals = {}
    for _, row in matches.iterrows():
        date_only = row["Date"].strftime("%Y-%m-%d")
        key = (row["HomeTeam"], row["AwayTeam"], date_only)
        actuals[key] = {"FTHG": int(row["FTHG"]), "FTAG": int(row["FTAG"]), "FTR": row["FTR"]}
    return actuals


def load_existing_history() -> pd.DataFrame:
    path = config.PREDICTIONS_DIR / "prediction_history.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def compute_row(pred: dict, actual: dict) -> dict:
    label_idx = {"H": 0, "D": 1, "A": 2}
    y_true_idx = np.array([label_idx[actual["FTR"]]])
    proba = np.array([[pred["p_home_win"], pred["p_draw"], pred["p_away_win"]]])

    predicted_idx = int(np.argmax(proba[0]))
    predicted_result = ["H", "D", "A"][predicted_idx]

    return {
        "prediction_time": pred["_prediction_time"],
        "match_date": pred["_match_date"],
        "home_team": pred["home_team"],
        "away_team": pred["away_team"],
        "predicted_home": round(pred["p_home_win"], 4),
        "predicted_draw": round(pred["p_draw"], 4),
        "predicted_away": round(pred["p_away_win"], 4),
        "predicted_result": predicted_result,
        "predicted_score": pred.get("predicted_score"),
        "actual_home": actual["FTHG"],
        "actual_away": actual["FTAG"],
        "actual_result": actual["FTR"],
        "correct": predicted_result == actual["FTR"],
        "brier_score": round(evaluation.brier_score_multiclass(y_true_idx, proba), 4),
        "log_loss": round(evaluation.multiclass_log_loss(y_true_idx, proba), 4),
        "model_version": pred.get("model_version") or "unknown",
    }


def main():
    predictions = load_latest_predictions_per_match()
    actuals = load_actual_results()
    history = load_existing_history()

    already_recorded = set(zip(history["home_team"], history["away_team"], history["match_date"])) if len(history) else set()

    new_rows = []
    still_pending = 0
    for key, pred in predictions.items():
        if key in already_recorded:
            continue
        actual = actuals.get(key)
        if actual is None:
            still_pending += 1
            continue
        new_rows.append(compute_row(pred, actual))

    if new_rows:
        new_df = pd.DataFrame(new_rows)[HISTORY_COLUMNS]
        combined = pd.concat([history, new_df], ignore_index=True) if len(history) else new_df
        out_path = config.PREDICTIONS_DIR / "prediction_history.csv"
        combined.to_csv(out_path, index=False)
        print(f"新增 {len(new_rows)} 場比賽的回收結果，累積共 {len(combined)} 場，已存至 {out_path}")
        acc = new_df["correct"].mean()
        print(f"  本次新增比賽的命中率: {acc:.3f}")
        print(f"  本次新增比賽的平均 Brier Score: {new_df['brier_score'].mean():.4f}")
        print(f"  本次新增比賽的平均 Log Loss: {new_df['log_loss'].mean():.4f}")
    else:
        print("沒有新的比賽結果可以回收（可能都還沒開踢，或早就記錄過了）。")

    if still_pending:
        print(f"另有 {still_pending} 場已預測但尚未開踢/尚未有真實結果的比賽，等下次真實結果出來後再回收。")


if __name__ == "__main__":
    main()
