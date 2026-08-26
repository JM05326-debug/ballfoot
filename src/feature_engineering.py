"""
特徵工程層（第一階段：Elo + 近期狀態 + 主客場拆分 + 賽程休息）。

防止資料洩漏的核心規則：
  對於「這一場比賽」的任何球隊特徵，只能使用該球隊在「此日期之前」的比賽資料。
  本模組一律先建立「球隊比賽長表 (team match log)」，每支球隊每場比賽一列，
  依球隊、日期排序後，用 shift 的方式只取「前 N 場」計算加權平均，
  嚴禁把本場比賽自己的賽後數據 (HS/AS/HST/... /FTHG/FTAG 本身) 當作特徵，
  這些欄位只會保留在輸出資料集中作為訓練標籤 (label)，不會被拿來當作特徵。

輸出：data/processed/training_dataset.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import config, elo_model

FORM_WINDOWS = [3, 5, 10]
VENUE_SPLIT_WINDOW = 5
CONGESTION_DAYS = 14
RECENCY_DECAY = 0.85  # 越接近 1 表示新舊比賽權重差異越小；0.85 代表越近的比賽權重明顯較高

# 球隊比賽長表要追蹤的統計欄位：(長表欄位名, 主隊來源欄, 客隊來源欄)
LOG_METRIC_SOURCE = {
    "GF": ("FTHG", "FTAG"),
    "GA": ("FTAG", "FTHG"),
    "ShotsF": ("HS", "AS"),
    "ShotsA": ("AS", "HS"),
    "SOTF": ("HST", "AST"),
    "SOTA": ("AST", "HST"),
    "CornersF": ("HC", "AC"),
    "CornersA": ("AC", "HC"),
    "xGF": ("HxG", "AxG"),
    "xGA": ("AxG", "HxG"),
}

FORM_METRICS = ["Result", "GF", "GA", "ShotsF", "SOTF", "xGF", "xGA"]


def build_team_match_log(matches: pd.DataFrame) -> pd.DataFrame:
    """把「一場比賽一列」轉成「一支球隊一場比賽一列」的長表，主客場各一列。"""
    rows = []
    for _, m in matches.iterrows():
        for venue, team, opp in (("H", m["HomeTeam"], m["AwayTeam"]), ("A", m["AwayTeam"], m["HomeTeam"])):
            if venue == "H":
                result = 1.0 if m["FTR"] == "H" else (0.5 if m["FTR"] == "D" else 0.0)
            else:
                result = 1.0 if m["FTR"] == "A" else (0.5 if m["FTR"] == "D" else 0.0)

            rec = {
                "MatchID": m["MatchID"],
                "Date": m["Date"],
                "Season": m["Season"],
                "SeasonStartYear": m["SeasonStartYear"],
                "Team": team,
                "Opponent": opp,
                "Venue": venue,
                "Result": result,
            }
            for log_col, (home_src, away_src) in LOG_METRIC_SOURCE.items():
                src_col = home_src if venue == "H" else away_src
                rec[log_col] = m[src_col] if src_col in matches.columns else np.nan
            rows.append(rec)

    log = pd.DataFrame(rows)
    log = log.sort_values(["Team", "Date"]).reset_index(drop=True)
    return log


def _weighted_avg(values: np.ndarray, decay: float = RECENCY_DECAY) -> float:
    n = len(values)
    if n == 0:
        return np.nan
    finite_mask = ~np.isnan(values)
    if finite_mask.sum() == 0:
        return np.nan
    idx = np.arange(n)
    weights = decay ** (n - 1 - idx)
    weights = weights[finite_mask]
    vals = values[finite_mask]
    return float(np.sum(weights * vals) / np.sum(weights))


def add_rolling_form_features(log: pd.DataFrame) -> pd.DataFrame:
    """在長表上，針對每個 window 計算「此場之前」的加權平均值。"""
    log = log.sort_values(["Team", "Date"]).reset_index(drop=True)

    for metric in FORM_METRICS:
        for w in FORM_WINDOWS:
            log[f"{metric}_L{w}"] = np.nan

    log["MatchesPlayedPrior"] = 0
    log["RestDays"] = np.nan
    log["MatchesLast14Days"] = 0

    for team, idx in log.groupby("Team").groups.items():
        idx = list(idx)
        sub = log.loc[idx].sort_values("Date")
        order = sub.index.tolist()
        dates = sub["Date"].tolist()
        metric_arrays = {m: sub[m].to_numpy(dtype=float) for m in FORM_METRICS}

        for pos, row_idx in enumerate(order):
            log.at[row_idx, "MatchesPlayedPrior"] = pos

            if pos > 0:
                rest = (dates[pos] - dates[pos - 1]).days
                log.at[row_idx, "RestDays"] = rest
                window_start = dates[pos] - pd.Timedelta(days=CONGESTION_DAYS)
                count_recent = sum(1 for d in dates[:pos] if d > window_start)
                log.at[row_idx, "MatchesLast14Days"] = count_recent

            for w in FORM_WINDOWS:
                start = max(0, pos - w)
                if start == pos:
                    continue
                for metric in FORM_METRICS:
                    vals = metric_arrays[metric][start:pos]
                    log.at[row_idx, f"{metric}_L{w}"] = _weighted_avg(vals)

    return log


def add_venue_split_features(log: pd.DataFrame, window: int = VENUE_SPLIT_WINDOW) -> pd.DataFrame:
    """主場專屬 / 客場專屬 的近況（只用同樣主客場身份的過去比賽）。"""
    log = log.sort_values(["Team", "Venue", "Date"]).reset_index(drop=True)
    log[f"VenueGF_L{window}"] = np.nan
    log[f"VenueGA_L{window}"] = np.nan
    log[f"VenueResult_L{window}"] = np.nan

    for (team, venue), idx in log.groupby(["Team", "Venue"]).groups.items():
        idx = list(idx)
        sub = log.loc[idx].sort_values("Date")
        order = sub.index.tolist()
        gf_arr = sub["GF"].to_numpy(dtype=float)
        ga_arr = sub["GA"].to_numpy(dtype=float)
        res_arr = sub["Result"].to_numpy(dtype=float)

        for pos, row_idx in enumerate(order):
            start = max(0, pos - window)
            if start == pos:
                continue
            log.at[row_idx, f"VenueGF_L{window}"] = _weighted_avg(gf_arr[start:pos])
            log.at[row_idx, f"VenueGA_L{window}"] = _weighted_avg(ga_arr[start:pos])
            log.at[row_idx, f"VenueResult_L{window}"] = _weighted_avg(res_arr[start:pos])

    return log


_RAW_CURRENT_MATCH_COLUMNS = set(LOG_METRIC_SOURCE.keys()) | {"Result"}
_NON_FEATURE_COLUMNS = {"MatchID", "Date", "Season", "SeasonStartYear", "Team", "Opponent", "Venue"}


def _merge_side(matches: pd.DataFrame, log: pd.DataFrame, venue: str, prefix: str) -> pd.DataFrame:
    """把長表併回比賽層級資料。

    重要：長表裡的 Result/GF/GA/ShotsF/... 等「原始欄位」代表的是這場比賽本身的
    賽後結果，只能拿來當作『計算未來場次』滾動特徵的原料，絕對不能直接併回這場
    比賽自己的特徵欄位（否則等於用比賽結果去預測比賽結果）。因此這裡只保留
    _L3/_L5/_L10 等滾動視窗欄位，以及 MatchesPlayedPrior/RestDays/... 等賽前已知資訊。
    """
    side_log = log[log["Venue"] == venue].copy()
    feature_cols = [
        c for c in side_log.columns
        if c not in _NON_FEATURE_COLUMNS and c not in _RAW_CURRENT_MATCH_COLUMNS
    ]
    rename_map = {c: f"{prefix}_{c}" for c in feature_cols}
    side_log = side_log[["MatchID"] + feature_cols].rename(columns=rename_map)
    return matches.merge(side_log, on="MatchID", how="left")


def build_training_dataset(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.sort_values("Date").reset_index(drop=True)

    log = build_team_match_log(matches)
    log = add_rolling_form_features(log)
    log = add_venue_split_features(log)

    dataset = matches.copy()
    dataset = _merge_side(dataset, log, "H", "Home")
    dataset = _merge_side(dataset, log, "A", "Away")

    elo_df, final_ratings = elo_model.compute_elo_features(matches)
    elo_cols = elo_df[["MatchID", "HomeEloPre", "AwayEloPre", "EloDiffPre"]]
    dataset = dataset.merge(elo_cols, on="MatchID", how="left")
    elo_model.save_latest_ratings(final_ratings, as_of_date=str(matches["Date"].max().date()))

    label_cols = ["FTHG", "FTAG", "FTR"]
    stat_cols_to_drop = [c for c in ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR", "HxG", "AxG", "HTHG", "HTAG", "HTR"] if c in dataset.columns]
    dataset = dataset.drop(columns=stat_cols_to_drop)

    front_cols = ["MatchID", "Season", "SeasonStartYear", "Date", "HomeTeam", "AwayTeam"] + label_cols
    other_cols = [c for c in dataset.columns if c not in front_cols]
    dataset = dataset[front_cols + other_cols]

    return dataset


def team_snapshot(team: str, as_of_date: pd.Timestamp, log: pd.DataFrame, venue_filter: str | None = None) -> dict:
    """算出「現在」(as_of_date 之前) 某支球隊的滾動特徵快照，供還沒開踢的比賽做預測。

    跟 add_rolling_form_features() 的邏輯完全一樣（一樣是「只用這個時間點之前的資料」、
    一樣的加權公式），差別只在於這裡不是針對歷史上「某一場比賽的當下」計算，
    而是針對「現在這個時間點」計算，用於還沒發生的未來比賽。
    """
    hist = log[(log["Team"] == team) & (log["Date"] < as_of_date)]
    unfiltered_hist = hist.sort_values("Date")
    if venue_filter is not None:
        hist = hist[hist["Venue"] == venue_filter]
    hist = hist.sort_values("Date")

    n_any = len(unfiltered_hist)
    out: dict = {"MatchesPlayedPrior": n_any}

    if n_any == 0:
        out["RestDays"] = np.nan
        out["MatchesLast14Days"] = 0
    else:
        last_date = unfiltered_hist["Date"].iloc[-1]
        out["RestDays"] = (as_of_date - last_date).days
        window_start = as_of_date - pd.Timedelta(days=CONGESTION_DAYS)
        out["MatchesLast14Days"] = int((unfiltered_hist["Date"] > window_start).sum())

    for metric in FORM_METRICS:
        vals_all = unfiltered_hist[metric].to_numpy(dtype=float)
        for w in FORM_WINDOWS:
            vals = vals_all[-w:] if len(vals_all) > 0 else vals_all
            out[f"{metric}_L{w}"] = _weighted_avg(vals) if len(vals) > 0 else np.nan

    if venue_filter is not None:
        w = VENUE_SPLIT_WINDOW
        gf_all = hist["GF"].to_numpy(dtype=float)[-w:]
        ga_all = hist["GA"].to_numpy(dtype=float)[-w:]
        res_all = hist["Result"].to_numpy(dtype=float)[-w:]
        out[f"VenueGF_L{w}"] = _weighted_avg(gf_all) if len(gf_all) > 0 else np.nan
        out[f"VenueGA_L{w}"] = _weighted_avg(ga_all) if len(ga_all) > 0 else np.nan
        out[f"VenueResult_L{w}"] = _weighted_avg(res_all) if len(res_all) > 0 else np.nan

    return out


def build_features_for_upcoming(upcoming: pd.DataFrame, historical_matches: pd.DataFrame, elo_ratings: dict) -> pd.DataFrame:
    """對還沒開踢的比賽（只需要 MatchID/Date/HomeTeam/AwayTeam）套用跟訓練資料集
    完全相同的特徵定義，用「目前為止全部已知的真實比賽結果」當作歷史依據。

    重要限制（誠實記錄，不隱藏）：如果一次預測「未來好幾輪」比賽，第 2 輪、第 3 輪的
    「近期狀態」特徵會缺少第 1 輪（同樣還沒開踢）的結果，因為那場比賽當下也還沒發生。
    這是任何預測系統本質上的限制，不是 bug——要拿到最準確的「近期狀態」，應該在
    每一輪比賽結束、資料更新後，再預測下一輪，而不是一次預測好幾輪之後的比賽。
    """
    historical_matches = historical_matches.sort_values("Date").reset_index(drop=True)
    log = build_team_match_log(historical_matches)

    records = []
    for _, m in upcoming.iterrows():
        as_of_date = m["Date"]
        home_snap = team_snapshot(m["HomeTeam"], as_of_date, log, venue_filter=None)
        away_snap = team_snapshot(m["AwayTeam"], as_of_date, log, venue_filter=None)
        home_venue_snap = team_snapshot(m["HomeTeam"], as_of_date, log, venue_filter="H")
        away_venue_snap = team_snapshot(m["AwayTeam"], as_of_date, log, venue_filter="A")

        rec = {
            "MatchID": m["MatchID"],
            "RoundNumber": m.get("RoundNumber"),
            "Date": as_of_date,
            "HomeTeam": m["HomeTeam"],
            "AwayTeam": m["AwayTeam"],
        }
        for k, v in home_snap.items():
            rec[f"Home_{k}"] = v
        for k, v in away_snap.items():
            rec[f"Away_{k}"] = v
        for k in (f"VenueGF_L{VENUE_SPLIT_WINDOW}", f"VenueGA_L{VENUE_SPLIT_WINDOW}", f"VenueResult_L{VENUE_SPLIT_WINDOW}"):
            rec[f"Home_{k}"] = home_venue_snap.get(k, np.nan)
            rec[f"Away_{k}"] = away_venue_snap.get(k, np.nan)

        home_elo = elo_ratings.get(m["HomeTeam"], elo_model.NEW_TEAM_INITIAL_RATING)
        away_elo = elo_ratings.get(m["AwayTeam"], elo_model.NEW_TEAM_INITIAL_RATING)
        rec["HomeEloPre"] = home_elo
        rec["AwayEloPre"] = away_elo
        rec["EloDiffPre"] = home_elo - away_elo

        records.append(rec)

    return pd.DataFrame(records)


def run() -> pd.DataFrame:
    matches_path = config.PROCESSED_DIR / "matches_clean.csv"
    if not matches_path.exists():
        raise FileNotFoundError(
            f"找不到 {matches_path}，請先執行 data_cleaner.py（或直接執行 update_data.py）"
        )
    matches = pd.read_csv(matches_path, parse_dates=["Date"])
    dataset = build_training_dataset(matches)

    out_path = config.PROCESSED_DIR / "training_dataset.csv"
    dataset.to_csv(out_path, index=False)
    return dataset


def _main():
    dataset = run()
    print(f"特徵工程完成，訓練資料集: {len(dataset)} 場比賽, {dataset.shape[1]} 個欄位")
    print(f"輸出: {config.PROCESSED_DIR / 'training_dataset.csv'}")

    no_history = dataset["Home_MatchesPlayedPrior"].fillna(0) == 0
    print(f"主隊完全沒有歷史資料可用的比賽數（Elo/Form 特徵會是 NaN）: {int(no_history.sum())}")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    _main()
