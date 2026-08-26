"""
獨立比分模型：Dixon-Coles（帶低比分修正的雙變量 Poisson）。

不讓分類模型直接「猜比分」——本模組獨立估計每隊的攻擊力/防守力參數，
算出主客隊「預期進球」(lambda)，再組出完整比分機率矩陣，由矩陣本身
推導 1X2、Over/Under 2.5、BTTS、以及最可能的 Top-N 比分。

模型設定（Dixon & Coles, 1997）：
  log(lambda_home) = c + home_adv + attack[home] - defense[away]
  log(lambda_away) = c + attack[away] - defense[home]
  tau(x,y) 對 (0,0)/(1,0)/(0,1)/(1,1) 這幾個低比分做修正，修正真實足球比賽中
  低比分（尤其 0-0、1-1）發生率與獨立 Poisson 假設之間的落差。

防止資料洩漏：本模組只會用「切分好的 Train 期間」比賽結果去估計球隊參數，
估好之後參數是凍結的，拿去預測 Validation/Test（或未來）比賽時完全不會
再用到那些比賽自己的結果。新升班/資料中沒出現過的球隊，attack/defense
一律退回聯盟平均值（0），不假設其強弱。

時間加權：離「估計基準日」越久的比賽，權重呈指數衰減（half_life_days 控制半衰期），
呼應第一階段近期狀態特徵「越近的比賽權重越高」的設計精神。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from . import config

RIDGE_LAMBDA = 0.01
SHRINKAGE_K = 3.0  # 小樣本球隊的收縮常數，見 fit_dixon_coles() 內的說明
DEFAULT_HALF_LIFE_DAYS = 260.0
MAX_GOALS_FOR_MATRIX = 10
RHO_BOUNDS = (-0.3, 0.3)
ATTACK_DEFENSE_BOUNDS = (-3.0, 3.0)
HOME_ADV_BOUNDS = (-0.2, 1.0)
INTERCEPT_BOUNDS = (-1.0, 2.0)


@dataclass
class DixonColesModel:
    teams: list[str]
    attack: dict[str, float]
    defense: dict[str, float]
    home_adv: float
    intercept: float
    rho: float
    as_of_date: str
    n_matches_used: int

    def get_attack(self, team: str) -> float:
        return self.attack.get(team, 0.0)

    def get_defense(self, team: str) -> float:
        return self.defense.get(team, 0.0)

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        log_lh = self.intercept + self.home_adv + self.get_attack(home_team) - self.get_defense(away_team)
        log_la = self.intercept + self.get_attack(away_team) - self.get_defense(home_team)
        return float(np.exp(log_lh)), float(np.exp(log_la))

    def to_dict(self) -> dict:
        return {
            "teams": self.teams,
            "attack": self.attack,
            "defense": self.defense,
            "home_adv": self.home_adv,
            "intercept": self.intercept,
            "rho": self.rho,
            "as_of_date": self.as_of_date,
            "n_matches_used": self.n_matches_used,
        }


def _tau_vectorized(home_goals: np.ndarray, away_goals: np.ndarray, lam_home: np.ndarray, lam_away: np.ndarray, rho: float) -> np.ndarray:
    tau = np.ones_like(lam_home)
    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)
    tau[m00] = 1 - lam_home[m00] * lam_away[m00] * rho
    tau[m01] = 1 + lam_home[m01] * rho
    tau[m10] = 1 + lam_away[m10] * rho
    tau[m11] = 1 - rho
    return np.clip(tau, 1e-6, None)


def _negative_log_likelihood(params: np.ndarray, home_idx, away_idx, home_goals, away_goals, weights, n_teams: int) -> float:
    attack = params[:n_teams]
    defense = params[n_teams : 2 * n_teams]
    home_adv = params[2 * n_teams]
    intercept = params[2 * n_teams + 1]
    rho = params[2 * n_teams + 2]

    log_lh = intercept + home_adv + attack[home_idx] - defense[away_idx]
    log_la = intercept + attack[away_idx] - defense[home_idx]
    lam_home = np.exp(log_lh)
    lam_away = np.exp(log_la)

    logpmf_home = stats.poisson.logpmf(home_goals, lam_home)
    logpmf_away = stats.poisson.logpmf(away_goals, lam_away)
    tau = _tau_vectorized(home_goals, away_goals, lam_home, lam_away, rho)

    log_lik = np.sum(weights * (np.log(tau) + logpmf_home + logpmf_away))
    ridge = RIDGE_LAMBDA * float(np.sum(attack**2) + np.sum(defense**2))
    return -log_lik + ridge


def fit_dixon_coles(
    matches: pd.DataFrame,
    as_of_date: pd.Timestamp | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> DixonColesModel:
    """用 matches（需含 Date/HomeTeam/AwayTeam/FTHG/FTAG）估計 Dixon-Coles 參數。

    as_of_date 預設為 matches 中最新的日期，作為時間衰減權重的基準點。
    """
    matches = matches.sort_values("Date").reset_index(drop=True)
    if as_of_date is None:
        as_of_date = matches["Date"].max()

    teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
    team_to_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    home_idx = matches["HomeTeam"].map(team_to_idx).to_numpy()
    away_idx = matches["AwayTeam"].map(team_to_idx).to_numpy()
    home_goals = matches["FTHG"].to_numpy(dtype=float)
    away_goals = matches["FTAG"].to_numpy(dtype=float)

    days_before = (as_of_date - matches["Date"]).dt.days.to_numpy().astype(float)
    days_before = np.clip(days_before, 0, None)
    xi = np.log(2) / half_life_days
    weights = np.exp(-xi * days_before)

    x0 = np.zeros(2 * n_teams + 3)
    x0[2 * n_teams] = 0.25  # home_adv 初始值
    x0[2 * n_teams + 1] = float(np.log(max(matches["FTHG"].mean(), 0.1)))  # intercept 初始值
    x0[2 * n_teams + 2] = 0.0  # rho 初始值

    bounds = (
        [ATTACK_DEFENSE_BOUNDS] * n_teams
        + [ATTACK_DEFENSE_BOUNDS] * n_teams
        + [HOME_ADV_BOUNDS, INTERCEPT_BOUNDS, RHO_BOUNDS]
    )

    result = minimize(
        _negative_log_likelihood,
        x0,
        args=(home_idx, away_idx, home_goals, away_goals, weights, n_teams),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500},
    )

    params = result.x
    attack = params[:n_teams]
    defense = params[n_teams : 2 * n_teams]
    home_adv = float(params[2 * n_teams])
    intercept = float(params[2 * n_teams + 1])
    rho = float(params[2 * n_teams + 2])

    # 重新置中，讓 mean(attack)=mean(defense)=0，方便解讀「攻擊力高於/低於聯盟平均多少」，
    # 同時把偏移量吸收進 intercept，數學上完全不影響任何一場比賽的預測結果。
    attack_mean = float(attack.mean())
    defense_mean = float(defense.mean())
    attack = attack - attack_mean
    defense = defense - defense_mean
    intercept = intercept + attack_mean - defense_mean

    # 小樣本收縮 (James-Stein 風格)：只有 1~2 場比賽紀錄的球隊（通常是剛升班、
    # 資料集裡第一次出現），MLE 很容易被單一極端比分拉到邊界值（例如只輸一場
    # 0-3，attack 參數會被推到接近下界 -3，換算成預期進球幾乎是 0，這在實務上
    # 不合理——一場比賽不足以判斷一支英超球隊「幾乎不會進球」）。
    # 因此比賽場次越少，attack/defense 就越往聯盟平均（0）收縮：
    #   shrink_factor = n / (n + SHRINKAGE_K)
    # 場次夠多時 shrink_factor -> 1，幾乎不影響原本的估計值。
    n_matches_per_team = pd.concat([matches["HomeTeam"], matches["AwayTeam"]]).value_counts()
    for i, team in enumerate(teams):
        n = int(n_matches_per_team.get(team, 0))
        shrink_factor = n / (n + SHRINKAGE_K)
        attack[i] *= shrink_factor
        defense[i] *= shrink_factor

    return DixonColesModel(
        teams=teams,
        attack={t: float(attack[team_to_idx[t]]) for t in teams},
        defense={t: float(defense[team_to_idx[t]]) for t in teams},
        home_adv=home_adv,
        intercept=intercept,
        rho=rho,
        as_of_date=str(pd.Timestamp(as_of_date).date()),
        n_matches_used=len(matches),
    )


def score_matrix(model: DixonColesModel, home_team: str, away_team: str, max_goals: int = MAX_GOALS_FOR_MATRIX) -> np.ndarray:
    lam_home, lam_away = model.expected_goals(home_team, away_team)
    goals = np.arange(0, max_goals + 1)
    p_home = stats.poisson.pmf(goals, lam_home)
    p_away = stats.poisson.pmf(goals, lam_away)
    matrix = np.outer(p_home, p_away)

    for x in (0, 1):
        for y in (0, 1):
            tau = _tau_vectorized(
                np.array([x], dtype=float), np.array([y], dtype=float),
                np.array([lam_home]), np.array([lam_away]), model.rho,
            )[0]
            matrix[x, y] *= tau

    matrix = matrix / matrix.sum()
    return matrix


def predict_match(model: DixonColesModel, home_team: str, away_team: str, max_goals: int = MAX_GOALS_FOR_MATRIX, top_n: int = 5) -> dict:
    lam_home, lam_away = model.expected_goals(home_team, away_team)
    matrix = score_matrix(model, home_team, away_team, max_goals)

    goals = np.arange(0, max_goals + 1)
    home_grid, away_grid = np.meshgrid(goals, goals, indexing="ij")

    p_home_win = float(matrix[home_grid > away_grid].sum())
    p_draw = float(matrix[home_grid == away_grid].sum())
    p_away_win = float(matrix[home_grid < away_grid].sum())

    total_goals_grid = home_grid + away_grid
    p_over25 = float(matrix[total_goals_grid > 2.5].sum())
    p_under25 = 1.0 - p_over25

    btts_grid = (home_grid >= 1) & (away_grid >= 1)
    p_btts_yes = float(matrix[btts_grid].sum())
    p_btts_no = 1.0 - p_btts_yes

    flat_idx = np.dstack(np.unravel_index(np.argsort(-matrix, axis=None), matrix.shape))[0]
    top_scorelines = []
    for i, j in flat_idx[:top_n]:
        top_scorelines.append({"score": f"{i}-{j}", "probability": float(matrix[i, j])})

    return {
        "home_team": home_team,
        "away_team": away_team,
        "expected_home_goals": lam_home,
        "expected_away_goals": lam_away,
        "p_home_win": p_home_win,
        "p_draw": p_draw,
        "p_away_win": p_away_win,
        "p_over_2_5": p_over25,
        "p_under_2_5": p_under25,
        "p_btts_yes": p_btts_yes,
        "p_btts_no": p_btts_no,
        "top_scorelines": top_scorelines,
    }


def predict_matches_df(model: DixonColesModel, matches: pd.DataFrame) -> pd.DataFrame:
    """對一整批比賽（需含 HomeTeam/AwayTeam 欄位）套用模型，回傳含機率欄位的 DataFrame。"""
    records = []
    for _, row in matches.iterrows():
        pred = predict_match(model, row["HomeTeam"], row["AwayTeam"])
        records.append(
            {
                "MatchID": row.get("MatchID"),
                "Poisson_ExpHomeGoals": pred["expected_home_goals"],
                "Poisson_ExpAwayGoals": pred["expected_away_goals"],
                "Poisson_P_HomeWin": pred["p_home_win"],
                "Poisson_P_Draw": pred["p_draw"],
                "Poisson_P_AwayWin": pred["p_away_win"],
                "Poisson_P_Over25": pred["p_over_2_5"],
                "Poisson_P_Under25": pred["p_under_2_5"],
                "Poisson_P_BTTS_Yes": pred["p_btts_yes"],
                "Poisson_P_BTTS_No": pred["p_btts_no"],
            }
        )
    return pd.DataFrame(records)


def save_model(model: DixonColesModel, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model.to_dict(), f, ensure_ascii=False, indent=2)


def load_model(path) -> DixonColesModel:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return DixonColesModel(**d)
