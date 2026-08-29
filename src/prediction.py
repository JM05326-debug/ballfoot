"""
第五階段：對「還沒開踢」的比賽產生完整預測（規格第九節要求的完整格式）。

流程：
  1. 用「目前為止全部已知的真實比賽結果」（含目前賽季已踢的場次）重新訓練一套
     「正式上線版」模型——這跟 train.py／evaluate.py 用的「只用 Train 賽季」版本不同：
     train.py/evaluate.py 的版本是為了誠實評估方法論好不好，正式要拿去預測未來時，
     應該用到目前所有能用的真實資料，不該為了評估而浪費掉 Validation/Test 那些資料。
  2. Ensemble 權重、Platt 校準器，沿用 train.py 用 Validation 資料選出來的固定版本
     ——這兩個東西代表的是「方法論」，不會因為換一批訓練資料就重新調整。
  3. 1X2 機率 = 校準後的 Ensemble；Predicted Score / Expected Goals / Over-Under 2.5 / BTTS
     一律來自獨立的 Poisson/Dixon-Coles 模型（規格明確要求不能讓分類模型直接猜比分）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, ensemble as ensemble_mod, evaluation, feature_engineering, fuzzy_model, ml_models, poisson_model

CONFIDENCE_HIGH_GAP = 0.30
CONFIDENCE_MEDIUM_GAP = 0.15

FEATURE_LABELS = {
    "Result_L5": "近期戰績（近 5 場加權勝率）",
    "Result_L10": "近期戰績（近 10 場加權勝率）",
    "GF_L10": "近期進攻火力（近 10 場場均進球）",
    "GA_L10": "近期防守穩定度（近 10 場場均失球，越低越穩）",
    "ShotsF_L5": "近期射門量（近 5 場）",
    "SOTF_L5": "近期射正量（近 5 場）",
    "VenueResult_L5": "主/客場近況（近 5 個主場或客場）",
    "RestDays": "休息天數",
    "MatchesLast14Days": "近期賽程密度（14 天內比賽數）",
}
ELO_LABEL = "整體實力差距（Elo）"


def fit_production_models(dataset: pd.DataFrame, feature_cols: list[str]):
    """用全部目前已知的真實資料（含目前賽季已踢場次）重新訓練 ML 模型 + Poisson。"""
    X_all, y_all = ml_models.prepare_xy(dataset, feature_cols)
    registry = ml_models.build_model_registry()
    fitted = {}
    for name, model in registry.items():
        model.fit(X_all, y_all)
        fitted[name] = model
    return fitted


def fit_production_poisson(matches_clean: pd.DataFrame) -> poisson_model.DixonColesModel:
    return poisson_model.fit_dixon_coles(matches_clean)


def get_next_unplayed_round(fixtures_full: pd.DataFrame, season_start_year: int) -> int | None:
    season_fx = fixtures_full[fixtures_full["SeasonStartYear"] == season_start_year]
    unplayed = season_fx[season_fx["Played"] == False]  # noqa: E712
    if unplayed.empty:
        return None
    return int(unplayed["RoundNumber"].min())


def build_upcoming_matches(fixtures_full: pd.DataFrame, season_start_year: int, round_number: int) -> pd.DataFrame:
    fx = fixtures_full[
        (fixtures_full["SeasonStartYear"] == season_start_year) & (fixtures_full["RoundNumber"] == round_number)
    ].copy()
    fx["Date"] = pd.to_datetime(fx["Date"])
    fx["MatchID"] = (
        fx["Date"].dt.strftime("%Y%m%d") + "_" + fx["HomeTeam"].str.replace(" ", "") + "_" + fx["AwayTeam"].str.replace(" ", "")
    )
    return fx[["MatchID", "RoundNumber", "Date", "HomeTeam", "AwayTeam"]].reset_index(drop=True)


def confidence_level(proba: np.ndarray) -> str:
    sorted_p = np.sort(proba)[::-1]
    gap = sorted_p[0] - sorted_p[1]
    if gap >= CONFIDENCE_HIGH_GAP:
        return "High"
    if gap >= CONFIDENCE_MEDIUM_GAP:
        return "Medium"
    return "Low"


def top_influencing_factors(
    feature_row: pd.Series,
    feature_cols: list[str],
    feature_importances: dict[str, float],
    feature_std: dict[str, float],
    n: int = 5,
) -> list[dict]:
    """簡化版的「賽事影響因素」排名：RandomForest 全域特徵重要性 x 這場比賽主客差距的標準化程度。

    這不是 SHAP 那種嚴謹的逐場歸因方法，是刻意選擇的輕量作法（誠實說明，不假裝是嚴謹的
    可解釋性分析）：用全域重要性當作「這個因素整體而言有多關鍵」，乘上「這場比賽主客雙方
    在這個因素上差距多大（標準化後）」，近似出「這個因素在這場比賽裡有多突出」。
    """
    candidates = []

    for base_metric, label in FEATURE_LABELS.items():
        home_col, away_col = f"Home_{base_metric}", f"Away_{base_metric}"
        if home_col not in feature_cols or away_col not in feature_cols:
            continue
        home_val = feature_row.get(home_col, np.nan)
        away_val = feature_row.get(away_col, np.nan)
        if pd.isna(home_val) or pd.isna(away_val):
            continue
        diff = home_val - away_val
        if base_metric == "GA_L10":  # 失球數：數值低代表防守好，方向要反過來解讀
            diff = -diff
        std = feature_std.get(home_col, 0) or 1.0
        importance = (feature_importances.get(home_col, 0) + feature_importances.get(away_col, 0)) / 2
        score = importance * (diff / std)
        favors = "Home" if diff > 0 else "Away"
        candidates.append({"label": label, "score": float(score), "favors": favors})

    if "EloDiffPre" in feature_cols:
        diff = feature_row.get("EloDiffPre", np.nan)
        if pd.notna(diff):
            std = feature_std.get("EloDiffPre", 0) or 1.0
            importance = feature_importances.get("EloDiffPre", 0)
            score = importance * (diff / std)
            candidates.append({"label": ELO_LABEL, "score": float(score), "favors": "Home" if diff > 0 else "Away"})

    candidates.sort(key=lambda c: -abs(c["score"]))
    return candidates[:n]


def predict_one_match(
    match_row: pd.Series,
    feature_row: pd.Series,
    feature_cols: list[str],
    fitted_models: dict,
    dc_model: poisson_model.DixonColesModel,
    ensemble_weights: dict[str, float],
    calibrator,
    rf_importances: dict[str, float],
    feature_std: dict[str, float],
    model_version: str | None = None,
) -> dict:
    X = feature_row[feature_cols].to_numpy(dtype=float).reshape(1, -1)

    proba_store = {}
    for name, model in fitted_models.items():
        proba_store[name] = ml_models.predict_proba_aligned(model, X)[0]

    poisson_pred = poisson_model.predict_match(dc_model, match_row["HomeTeam"], match_row["AwayTeam"])
    proba_store["Poisson"] = np.array([poisson_pred["p_home_win"], poisson_pred["p_draw"], poisson_pred["p_away_win"]])

    fuzzy_input = feature_row.to_dict()
    fuzzy_out = fuzzy_model.evaluate_match(fuzzy_input)
    proba_store["Fuzzy"] = np.array([fuzzy_out["Fuzzy_P_HomeWin"], fuzzy_out["Fuzzy_P_Draw"], fuzzy_out["Fuzzy_P_AwayWin"]])

    members = [n for n in ensemble_weights if n in proba_store]
    w_sum = sum(ensemble_weights[n] for n in members)
    w = {n: ensemble_weights[n] / w_sum for n in members}
    ensemble_proba = ensemble_mod.weighted_ensemble_proba(
        {n: proba_store[n].reshape(1, -1) for n in members}, w
    )[0]

    calibrated_proba = calibration_apply_single(calibrator, ensemble_proba)

    factors = top_influencing_factors(feature_row, feature_cols, rf_importances, feature_std)

    return {
        "match_id": match_row.get("MatchID"),
        "home_team": match_row["HomeTeam"],
        "away_team": match_row["AwayTeam"],
        "date": str(match_row["Date"]),
        "round": int(match_row.get("RoundNumber")) if pd.notna(match_row.get("RoundNumber")) else None,
        "model_version": model_version,
        "p_home_win": float(calibrated_proba[0]),
        "p_draw": float(calibrated_proba[1]),
        "p_away_win": float(calibrated_proba[2]),
        "predicted_score": poisson_pred["top_scorelines"][0]["score"],
        "top_scorelines": poisson_pred["top_scorelines"],
        "expected_home_goals": poisson_pred["expected_home_goals"],
        "expected_away_goals": poisson_pred["expected_away_goals"],
        "p_over_2_5": poisson_pred["p_over_2_5"],
        "p_under_2_5": poisson_pred["p_under_2_5"],
        "p_btts_yes": poisson_pred["p_btts_yes"],
        "p_btts_no": poisson_pred["p_btts_no"],
        "confidence": confidence_level(calibrated_proba),
        "top_influencing_factors": factors,
        "per_model_raw_proba": {n: proba_store[n].tolist() for n in members},
        "ensemble_weights_used": w,
    }


def calibration_apply_single(calibrator, proba_row: np.ndarray) -> np.ndarray:
    from . import calibration
    return calibration.apply_calibrator(calibrator, proba_row.reshape(1, -1))[0]


def format_match_report(pred: dict) -> str:
    lines = []
    lines.append(f"Match: {pred['home_team']} vs {pred['away_team']}  ({pred['date']}, Round {pred['round']})")
    lines.append("Prediction:")
    lines.append(f"  Home Win: {pred['p_home_win']*100:.1f}%")
    lines.append(f"  Draw:     {pred['p_draw']*100:.1f}%")
    lines.append(f"  Away Win: {pred['p_away_win']*100:.1f}%")
    lines.append(f"Predicted Score: {pred['predicted_score']}")
    lines.append(f"Expected Goals: Home {pred['expected_home_goals']:.2f} - Away {pred['expected_away_goals']:.2f}")
    lines.append(f"Over 2.5:  {pred['p_over_2_5']*100:.1f}%")
    lines.append(f"Under 2.5: {pred['p_under_2_5']*100:.1f}%")
    lines.append(f"BTTS Yes: {pred['p_btts_yes']*100:.1f}%")
    lines.append(f"BTTS No:  {pred['p_btts_no']*100:.1f}%")
    lines.append(f"Confidence: {pred['confidence']}")
    lines.append("Top 5 influencing factors:")
    for i, f in enumerate(pred["top_influencing_factors"], 1):
        side = pred["home_team"] if f["favors"] == "Home" else pred["away_team"]
        lines.append(f"  {i}. {f['label']} -> 有利於 {side}")
    return "\n".join(lines)
