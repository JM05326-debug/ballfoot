"""
Elo 積分模型。

重點（防止資料洩漏）：
每場比賽會先「記錄賽前 (pre-match) Elo」，再用該場實際比分去更新 Elo。
下游的 feature/training dataset 只能使用 *_EloPre 欄位，絕對不能使用比賽當下
或賽後才更新完成的 Elo，否則等於用了未來資訊。

模型設計（World Football Elo 慣用做法）：
  E_home = 1 / (1 + 10^(-(elo_home + home_advantage - elo_away) / 400))
  S_home = 1 (主勝) / 0.5 (和局) / 0 (主敗)
  multiplier (依淨勝球數放大/縮小更新幅度):
      goal_diff <= 1  -> 1.0
      goal_diff == 2  -> 1.5
      goal_diff >= 3  -> (11 + goal_diff) / 8
  new_elo_home = elo_home + K * multiplier * (S_home - E_home)
  new_elo_away = elo_away + K * multiplier * (S_away - E_away)   # S_away = 1 - S_home, E_away = 1 - E_home

新升班/資料集中首次出現的球隊一律給予中性初始值 NEW_TEAM_INITIAL_RATING，
不假設其實力強弱（無法證實的假設 = 造假資料），這是已知限制：
升班馬在賽季初期幾場的 Elo 會偏不準，需要幾場比賽後才會收斂到合理水準。
"""

from __future__ import annotations

import json

import pandas as pd

from . import config

DEFAULT_K = 20.0
DEFAULT_HOME_ADVANTAGE = 100.0
NEW_TEAM_INITIAL_RATING = 1500.0


def _expected_home_score(elo_home: float, elo_away: float, home_advantage: float) -> float:
    return 1.0 / (1.0 + 10 ** (-((elo_home + home_advantage) - elo_away) / 400.0))


def _mov_multiplier(goal_diff: int) -> float:
    goal_diff = abs(goal_diff)
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11 + goal_diff) / 8.0


def compute_elo_features(
    matches: pd.DataFrame,
    k: float = DEFAULT_K,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
) -> tuple[pd.DataFrame, dict]:
    """依時間順序逐場計算 Elo，回傳附加 HomeEloPre/AwayEloPre/EloDiffPre 欄位的 DataFrame。

    matches 必須已依 Date 由舊到新排序（若未排序，本函式會先排序再計算，
    但呼叫端仍應自行確保排序，避免不同呼叫情境下順序不一致）。
    """
    df = matches.sort_values("Date").reset_index(drop=True).copy()

    ratings: dict[str, float] = {}
    home_pre, away_pre, diff_pre = [], [], []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        elo_home = ratings.get(home, NEW_TEAM_INITIAL_RATING)
        elo_away = ratings.get(away, NEW_TEAM_INITIAL_RATING)

        home_pre.append(elo_home)
        away_pre.append(elo_away)
        diff_pre.append(elo_home - elo_away)

        exp_home = _expected_home_score(elo_home, elo_away, home_advantage)
        if row["FTR"] == "H":
            s_home = 1.0
        elif row["FTR"] == "D":
            s_home = 0.5
        else:
            s_home = 0.0

        gd = int(row["FTHG"]) - int(row["FTAG"])
        mult = _mov_multiplier(gd)

        new_elo_home = elo_home + k * mult * (s_home - exp_home)
        new_elo_away = elo_away + k * mult * ((1.0 - s_home) - (1.0 - exp_home))

        ratings[home] = new_elo_home
        ratings[away] = new_elo_away

    df["HomeEloPre"] = home_pre
    df["AwayEloPre"] = away_pre
    df["EloDiffPre"] = diff_pre

    return df, ratings


def save_latest_ratings(ratings: dict, as_of_date: str) -> None:
    path = config.PROCESSED_DIR / "elo_ratings_latest.json"
    payload = {
        "as_of_date": as_of_date,
        "home_advantage": DEFAULT_HOME_ADVANTAGE,
        "k_factor": DEFAULT_K,
        "ratings": {team: round(r, 1) for team, r in sorted(ratings.items(), key=lambda kv: -kv[1])},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
