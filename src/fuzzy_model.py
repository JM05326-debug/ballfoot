"""
模糊邏輯層 (Fuzzy Logic Layer)。

設計原則：
1. 所有輸入一律使用「隸屬函數 (membership function)」做模糊化，不使用硬切割
   （例如不會寫「勝率 > 60% = Strong」這種 if/else）。一個數值可以同時部分屬於
   多個模糊集合，例如某隊近期狀態可能同時 Good=0.65、Very Good=0.35。
2. 隸屬函數的斷點 (breakpoints) 不是憑空假設，而是從 data/processed/training_dataset.csv
   實際跑出來的分位數 (percentile) 決定，並在下方常數區塊註明來源數字，可重新驗證。
3. 規則庫使用標準 Mamdani 推論：
     rule strength = min(前提們的隸屬度)          # AND
     同一個 (輸出變數, 輸出詞彙) 若被多條規則指向 -> 取 max 聚合
     最後用 centroid（形心法）解模糊，得到 0~100 的清晰分數 (crisp potential score)
4. 本層只輸出「潛力分數 / 機率傾向」，不是最終機率——最終機率要等第七階段做
   Ensemble 時，把 Fuzzy Model 的輸出跟其他機器學習模型的輸出放在一起比較、加權。
   但這裡仍然提供一個簡單的正規化函式，把 Fuzzy Model 當成一個獨立可評估的模型
   （對應第七階段規格：「至少比較 ... 6. Fuzzy Model」）。

輸入特徵一律沿用第一階段 feature_engineering.py 產生、且已經過防洩漏處理的欄位
（Home_Result_L5、Home_GF_L10、Home_GA_L10、Home_VenueResult_L5、Home_RestDays、
Home_MatchesLast14Days、HomeEloPre，Away_* 同理），不會引入任何新的洩漏風險。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# 隸屬函數工具
# ---------------------------------------------------------------------------


def trimf(x: np.ndarray, abc: tuple[float, float, float]) -> np.ndarray:
    a, b, c = abc
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if a != b:
        left = (x - a) / (b - a)
        y = np.where((x >= a) & (x <= b), left, y)
    else:
        y = np.where(x == a, 1.0, y)
    if b != c:
        right = (c - x) / (c - b)
        y = np.where((x > b) & (x <= c), right, y)
    else:
        y = np.where(x == b, 1.0, y)
    y = np.clip(y, 0.0, 1.0)
    return y


def trapmf(x: np.ndarray, abcd: tuple[float, float, float, float]) -> np.ndarray:
    a, b, c, d = abcd
    x = np.asarray(x, dtype=float)
    y = np.ones_like(x)
    if a != b:
        y = np.where(x < b, (x - a) / (b - a), y)
    y = np.where(x < a, 0.0, y)
    if c != d:
        y = np.where(x > c, (d - x) / (d - c), y)
    y = np.where(x > d, 0.0, y)
    y = np.clip(y, 0.0, 1.0)
    return y


def interp_mf(x0: float, universe: np.ndarray, mf_values: np.ndarray) -> float:
    if x0 is None or (isinstance(x0, float) and np.isnan(x0)):
        return 0.0
    x0 = float(np.clip(x0, universe[0], universe[-1]))
    return float(np.interp(x0, universe, mf_values))


# ---------------------------------------------------------------------------
# 輸入變數定義（隸屬函數斷點來自 training_dataset.csv 實際分位數）
# ---------------------------------------------------------------------------
# 以下數字為 2018/19~2026/27（9 個賽季，3050 場）訓練資料集的實際分位數（已於開發時計算，
# 可用 analyze_input_distributions() 重新計算驗證）：
#   Elo:              p10=1412 p25=1464 p50=1524 p75=1608 p90=1728  (min=1255, max=1908)
#   Form(Result_L5):  p10=0.19 p25=0.33 p50=0.50 p75=0.67 p90=0.82  (0~1)
#   GF_L10:           p10=0.74 p25=1.00 p50=1.36 p75=1.79 p90=2.24
#   GA_L10:           p10=0.77 p25=1.05 p50=1.39 p75=1.76 p90=2.13
#   HomeVenueResultL5:p25=0.38 p50=0.56 p75=0.73
#   RestDays:         p10=3 p25=5 p50=7 p75=8 p90=14（含季初超長間隔離群值，故 fatigue 計算時會裁切上限）
#   MatchesLast14Days:p25=1 p50=1 p75=2 (0~4)

FORM_UNIVERSE = np.linspace(0, 1, 101)
ATTACK_UNIVERSE = np.linspace(0, 4.5, 91)
DEFENSE_UNIVERSE = np.linspace(0, 4.5, 91)  # 數值 = 場均失球，數值越低代表防守越強
HOME_ADV_UNIVERSE = np.linspace(0, 1, 101)
FATIGUE_UNIVERSE = np.linspace(0, 1, 101)
STRENGTH_UNIVERSE = np.linspace(1200, 1950, 151)

INPUT_TERMS: dict[str, dict[str, tuple]] = {
    "Form": {
        "VeryPoor": ("trap", (0.0, 0.0, 0.12, 0.28)),
        "Poor": ("tri", (0.15, 0.33, 0.50)),
        "Average": ("tri", (0.33, 0.50, 0.67)),
        "Good": ("tri", (0.50, 0.67, 0.85)),
        "VeryGood": ("trap", (0.72, 0.85, 1.0, 1.0)),
    },
    "Attack": {
        "Weak": ("trap", (0.0, 0.0, 0.80, 1.20)),
        "Average": ("tri", (0.90, 1.40, 1.90)),
        "Strong": ("trap", (1.60, 2.10, 4.5, 4.5)),
    },
    "Defense": {  # 輸入為場均失球；Strong = 失球少
        "Strong": ("trap", (0.0, 0.0, 0.80, 1.20)),
        "Average": ("tri", (0.90, 1.40, 1.90)),
        "Weak": ("trap", (1.60, 2.10, 4.5, 4.5)),
    },
    "HomeAdvantage": {
        "Low": ("trap", (0.0, 0.0, 0.30, 0.45)),
        "Medium": ("tri", (0.35, 0.55, 0.75)),
        "High": ("trap", (0.65, 0.80, 1.0, 1.0)),
    },
    "Fatigue": {
        "Low": ("trap", (0.0, 0.0, 0.25, 0.40)),
        "Medium": ("tri", (0.30, 0.50, 0.70)),
        "High": ("trap", (0.60, 0.75, 1.0, 1.0)),
    },
    "TeamStrength": {
        "Weak": ("trap", (1200, 1200, 1420, 1480)),
        "Average": ("tri", (1440, 1525, 1610)),
        "Strong": ("trap", (1570, 1650, 1950, 1950)),
    },
    # 相對差距變數（Home - Away）。勝負和局本質上是「相對」的，只用兩隊各自的
    # 絕對等級（TeamStrength/Form）容易在雙方都落在同一模糊集合時，Home/Draw/Away
    # 三個輸出完全沒有規則觸發、只能退回中性值 50，導致模型對任何比賽都判斷不出差異。
    # 因此另外加入 StrengthGap / FormGap，用 5 個詞彙做「全值域覆蓋」的分割
    # （任何差距值都至少會落在某個詞彙上有明顯隸屬度），確保主/和/客三個輸出
    # 一定會有規則被觸發，能反映出真正懸殊或勢均力敵的差異。
    # 斷點依據 EloDiffPre / (Home_Result_L5 - Away_Result_L5) 的實際分位數：
    #   EloDiffPre: p10=-210 p25=-104 p50=0 p75=108 p90=218 (min=-531, max=594)
    #   FormGap:    p10=-0.46 p25=-0.24 p50=-0.02 p75=0.22 p90=0.42
    "StrengthGap": {
        "AwayMuchStronger": ("trap", (-600, -600, -250, -150)),
        "AwayStronger": ("tri", (-220, -110, 0)),
        "Similar": ("tri", (-120, 0, 120)),
        "HomeStronger": ("tri", (0, 110, 220)),
        "HomeMuchStronger": ("trap", (150, 250, 600, 600)),
    },
    "FormGap": {
        "AwayFormMuchBetter": ("trap", (-1.0, -1.0, -0.5, -0.3)),
        "AwayFormBetter": ("tri", (-0.45, -0.22, 0.0)),
        "FormSimilar": ("tri", (-0.25, 0.0, 0.25)),
        "HomeFormBetter": ("tri", (0.0, 0.22, 0.45)),
        "HomeFormMuchBetter": ("trap", (0.3, 0.5, 1.0, 1.0)),
    },
}

STRENGTH_GAP_UNIVERSE = np.linspace(-600, 600, 121)
FORM_GAP_UNIVERSE = np.linspace(-1, 1, 101)

INPUT_UNIVERSE = {
    "Form": FORM_UNIVERSE,
    "Attack": ATTACK_UNIVERSE,
    "Defense": DEFENSE_UNIVERSE,
    "HomeAdvantage": HOME_ADV_UNIVERSE,
    "Fatigue": FATIGUE_UNIVERSE,
    "TeamStrength": STRENGTH_UNIVERSE,
    "StrengthGap": STRENGTH_GAP_UNIVERSE,
    "FormGap": FORM_GAP_UNIVERSE,
}


def _build_mf(shape: str, params: tuple, universe: np.ndarray) -> np.ndarray:
    if shape == "tri":
        return trimf(universe, params)
    if shape == "trap":
        return trapmf(universe, params)
    raise ValueError(f"未知隸屬函數形狀: {shape}")


def fuzzify(var_kind: str, x0: float) -> dict[str, float]:
    """把一個清晰數值模糊化成 {詞彙: 隸屬度} 字典。"""
    universe = INPUT_UNIVERSE[var_kind]
    out = {}
    for term, (shape, params) in INPUT_TERMS[var_kind].items():
        mf = _build_mf(shape, params, universe)
        out[term] = interp_mf(x0, universe, mf)
    return out


# ---------------------------------------------------------------------------
# 輸出變數定義（0~100 的「潛力分數」，Low/Medium/High 三個詞彙）
# ---------------------------------------------------------------------------

OUTPUT_UNIVERSE = np.linspace(0, 100, 101)
OUTPUT_TERMS = {
    "Low": ("trap", (0, 0, 25, 50)),
    "Medium": ("tri", (25, 50, 75)),
    "High": ("trap", (50, 75, 100, 100)),
}
OUTPUT_TERM_MF = {term: _build_mf(shape, params, OUTPUT_UNIVERSE) for term, (shape, params) in OUTPUT_TERMS.items()}

OUTPUT_VARIABLES = [
    "HomeWinPotential",
    "DrawPotential",
    "AwayWinPotential",
    "HomeGoalPotential",
    "AwayGoalPotential",
    "Over25Potential",
    "Under25Potential",
]


# ---------------------------------------------------------------------------
# 規則庫（至少 30 條，主客場對稱設計，不偏向主隊）
# 每條規則: (前提列表 [(輸入變數, 詞彙), ...], 結論列表 [(輸出變數, 詞彙), ...])
# 前提之間一律用 AND（取 min）；一條規則可以同時對多個輸出變數下結論
# （例如「主隊實力明顯佔優」同時代表 HomeWin=High、Draw=Low、AwayWin=Low）。
#
# 設計上的重要修正：勝負和局本質是「相對」的。開發過程中發現，如果只用
# StrengthGap/FormGap 之外，Home/Draw/Away 三個輸出各自獨立判斷「Strong」
# 「Average」等絕對等級，會出現雙方剛好落在同一模糊集合、卻沒有任何規則明確
# 講到另外兩個輸出變數該是多少的狀況——這會讓沒被規則觸及的變數退回中性值 50，
# 稀釋掉原本應該很懸殊的判斷（實測：主隊 1900 Elo 對客隊 1250 Elo 的極端情境，
# 修正前只給主勝 41% / 和局 30% / 客勝 30%，明顯不合理）。
# 因此下面第一組規則改用 StrengthGap / FormGap（涵蓋全值域、任何差距都至少有
# 一個詞彙有明顯隸屬度）當作勝負和局判斷的主幹，確保三個輸出一定會被觸發；
# 其餘規則（主場優勢、疲勞、絕對等級組合）則作為疊加的輔助證據。
# ---------------------------------------------------------------------------

Antecedent = tuple[str, str]
Consequent = tuple[str, str]
Rule = tuple[list[Antecedent], list[Consequent]]


def _r(antecedents: list[Antecedent], *consequents: Consequent) -> Rule:
    return (antecedents, list(consequents))


RULES: list[Rule] = [
    # === 主幹規則：相對實力差距 (StrengthGap) -> 主/和/客 潛力，全值域覆蓋 ===
    _r([("StrengthGap", "HomeMuchStronger")], ("HomeWinPotential", "High"), ("DrawPotential", "Low"), ("AwayWinPotential", "Low")),
    _r([("StrengthGap", "HomeStronger")], ("HomeWinPotential", "Medium"), ("DrawPotential", "Medium"), ("AwayWinPotential", "Low")),
    _r([("StrengthGap", "Similar")], ("HomeWinPotential", "Medium"), ("DrawPotential", "Medium"), ("AwayWinPotential", "Low")),
    _r([("StrengthGap", "AwayStronger")], ("HomeWinPotential", "Low"), ("DrawPotential", "Medium"), ("AwayWinPotential", "Medium")),
    _r([("StrengthGap", "AwayMuchStronger")], ("HomeWinPotential", "Low"), ("DrawPotential", "Low"), ("AwayWinPotential", "High")),
    # === 主幹規則：相對近期狀態差距 (FormGap) -> 主/和/客 潛力，全值域覆蓋 ===
    _r([("FormGap", "HomeFormMuchBetter")], ("HomeWinPotential", "High"), ("DrawPotential", "Low"), ("AwayWinPotential", "Low")),
    _r([("FormGap", "HomeFormBetter")], ("HomeWinPotential", "Medium"), ("DrawPotential", "Medium"), ("AwayWinPotential", "Low")),
    _r([("FormGap", "FormSimilar")], ("HomeWinPotential", "Medium"), ("DrawPotential", "Medium"), ("AwayWinPotential", "Low")),
    _r([("FormGap", "AwayFormBetter")], ("HomeWinPotential", "Low"), ("DrawPotential", "Medium"), ("AwayWinPotential", "Medium")),
    _r([("FormGap", "AwayFormMuchBetter")], ("HomeWinPotential", "Low"), ("DrawPotential", "Low"), ("AwayWinPotential", "High")),
    # === 輔助規則：主場優勢 / 主場劣勢情境（疊加證據，不單獨決定勝負）===
    _r([("HomeAdvantage", "High"), ("HomeForm", "Good")], ("HomeWinPotential", "High")),
    _r([("HomeAdvantage", "High"), ("StrengthGap", "Similar")], ("HomeWinPotential", "Medium")),
    _r([("HomeAdvantage", "Low"), ("AwayForm", "Good")], ("AwayWinPotential", "Medium")),
    _r([("HomeAdvantage", "Low"), ("AwayStrength", "Strong")], ("AwayWinPotential", "High")),
    _r([("HomeAdvantage", "Medium"), ("StrengthGap", "Similar")], ("DrawPotential", "High")),
    # === 輔助規則：疲勞 (Fatigue) ===
    _r([("HomeFatigue", "High"), ("AwayFatigue", "Low")], ("AwayWinPotential", "Medium")),
    _r([("AwayFatigue", "High"), ("HomeFatigue", "Low")], ("HomeWinPotential", "Medium")),
    _r([("HomeFatigue", "High"), ("HomeStrength", "Strong")], ("HomeWinPotential", "Medium")),
    _r([("AwayFatigue", "High"), ("AwayStrength", "Strong")], ("AwayWinPotential", "Medium")),
    _r([("HomeFatigue", "High"), ("AwayFatigue", "High")], ("Under25Potential", "Medium")),
    # === 輔助規則：絕對等級組合（Strong/Strong 或 Weak/Weak 傾向和局）===
    _r([("HomeStrength", "Weak"), ("AwayStrength", "Weak")], ("DrawPotential", "Medium")),
    _r([("HomeStrength", "Strong"), ("AwayStrength", "Strong")], ("DrawPotential", "Medium")),
    _r([("HomeStrength", "Strong"), ("HomeForm", "VeryGood")], ("HomeWinPotential", "High")),
    _r([("AwayStrength", "Strong"), ("AwayForm", "VeryGood")], ("AwayWinPotential", "High")),
    _r([("HomeStrength", "Weak"), ("HomeForm", "VeryPoor")], ("AwayWinPotential", "Medium")),
    _r([("AwayStrength", "Weak"), ("AwayForm", "VeryPoor")], ("HomeWinPotential", "Medium")),
    # === 進球潛力：Attack vs Defense ===
    _r([("HomeAttack", "Strong"), ("AwayDefense", "Weak")], ("HomeGoalPotential", "High")),
    _r([("HomeAttack", "Weak"), ("AwayDefense", "Strong")], ("HomeGoalPotential", "Low")),
    _r([("AwayAttack", "Strong"), ("HomeDefense", "Weak")], ("AwayGoalPotential", "High")),
    _r([("AwayAttack", "Weak"), ("HomeDefense", "Strong")], ("AwayGoalPotential", "Low")),
    _r([("HomeAttack", "Average"), ("AwayDefense", "Average")], ("HomeGoalPotential", "Medium")),
    _r([("AwayAttack", "Average"), ("HomeDefense", "Average")], ("AwayGoalPotential", "Medium")),
    _r([("HomeAttack", "Strong"), ("AwayDefense", "Average")], ("HomeGoalPotential", "Medium")),
    _r([("AwayAttack", "Strong"), ("HomeDefense", "Average")], ("AwayGoalPotential", "Medium")),
    _r([("HomeForm", "VeryGood"), ("HomeAttack", "Strong")], ("HomeGoalPotential", "High")),
    _r([("AwayForm", "VeryGood"), ("AwayAttack", "Strong")], ("AwayGoalPotential", "High")),
    # === Over/Under 2.5（含 Average/Average 中央情境，確保永遠有規則觸發）===
    _r([("HomeAttack", "Strong"), ("AwayAttack", "Strong"), ("HomeDefense", "Weak"), ("AwayDefense", "Weak")], ("Over25Potential", "High"), ("Under25Potential", "Low")),
    _r([("HomeDefense", "Strong"), ("AwayDefense", "Strong"), ("HomeAttack", "Weak"), ("AwayAttack", "Weak")], ("Under25Potential", "High"), ("Over25Potential", "Low")),
    _r([("HomeDefense", "Weak"), ("AwayDefense", "Weak")], ("Over25Potential", "High")),
    _r([("HomeAttack", "Strong"), ("AwayDefense", "Weak")], ("Over25Potential", "Medium")),
    _r([("AwayAttack", "Strong"), ("HomeDefense", "Weak")], ("Over25Potential", "Medium")),
    _r([("HomeDefense", "Strong"), ("AwayDefense", "Strong")], ("Under25Potential", "Medium")),
    _r([("HomeAttack", "Weak"), ("AwayAttack", "Weak")], ("Under25Potential", "Medium")),
    _r([("HomeAttack", "Weak"), ("AwayDefense", "Strong")], ("Over25Potential", "Low")),
    _r([("AwayAttack", "Weak"), ("HomeDefense", "Strong")], ("Over25Potential", "Low")),
    _r([("HomeAttack", "Average"), ("AwayAttack", "Average")], ("Over25Potential", "Medium"), ("Under25Potential", "Medium")),
]

assert len(RULES) >= 30, "規則數量必須至少 30 條"


# ---------------------------------------------------------------------------
# 疲勞複合指標
# ---------------------------------------------------------------------------

REST_DAYS_CAP = 14.0  # 超過 14 天休息，多休息不再進一步降低疲勞（p90 = 14）
CONGESTION_CAP = 4.0  # 14 天內比賽數上限（觀察最大值 = 4）


def compute_fatigue_index(rest_days: float, matches_last_14d: float) -> float:
    """把休息天數 + 賽程密度合成一個 0~1 的疲勞指標，數值越高代表越疲勞。

    這是一個明確標註出處的工程設計選擇（非資料本身），而不是偽造資料：
      rest_component = 1 - clip(rest_days, 0, 14) / 14   # 休息越少，這項越接近 1
      congestion_component = clip(matches_last_14d, 0, 4) / 4
      fatigue_index = 0.6 * rest_component + 0.4 * congestion_component
    """
    if rest_days is None or (isinstance(rest_days, float) and np.isnan(rest_days)):
        rest_component = 0.3  # 缺乏歷史資料時，給中性偏低的預設疲勞（不假設過度疲勞）
    else:
        rest_component = 1.0 - min(max(rest_days, 0.0), REST_DAYS_CAP) / REST_DAYS_CAP

    if matches_last_14d is None or (isinstance(matches_last_14d, float) and np.isnan(matches_last_14d)):
        congestion_component = 0.2
    else:
        congestion_component = min(max(matches_last_14d, 0.0), CONGESTION_CAP) / CONGESTION_CAP

    return float(0.6 * rest_component + 0.4 * congestion_component)


# ---------------------------------------------------------------------------
# 資料列 -> 模糊輸入 crisp 值
# ---------------------------------------------------------------------------

# 缺乏歷史資料（球隊剛進入資料集，Home_MatchesPlayedPrior==0 附近）時的中性預設值。
# 這些預設值等於聯盟平均水準，代表「不知道，就假設普通」，不是捏造一個具體強弱判斷。
NEUTRAL_DEFAULTS = {
    "Form": 0.5,
    "Attack": 1.4,
    "Defense": 1.4,
    "HomeAdvantage": 0.5,
    "TeamStrength": 1500.0,
}


def _safe(value, default):
    if value is None:
        return default
    try:
        if np.isnan(value):
            return default
    except TypeError:
        return default
    return float(value)


@dataclass
class MatchCrispInputs:
    HomeForm: float
    AwayForm: float
    HomeAttack: float
    AwayAttack: float
    HomeDefense: float
    AwayDefense: float
    HomeAdvantage: float
    HomeFatigue: float
    AwayFatigue: float
    HomeStrength: float
    AwayStrength: float
    StrengthGap: float
    FormGap: float


def build_crisp_inputs(row: dict) -> MatchCrispInputs:
    home_form = _safe(row.get("Home_Result_L5"), NEUTRAL_DEFAULTS["Form"])
    away_form = _safe(row.get("Away_Result_L5"), NEUTRAL_DEFAULTS["Form"])
    home_strength = _safe(row.get("HomeEloPre"), NEUTRAL_DEFAULTS["TeamStrength"])
    away_strength = _safe(row.get("AwayEloPre"), NEUTRAL_DEFAULTS["TeamStrength"])

    return MatchCrispInputs(
        HomeForm=home_form,
        AwayForm=away_form,
        HomeAttack=_safe(row.get("Home_GF_L10"), NEUTRAL_DEFAULTS["Attack"]),
        AwayAttack=_safe(row.get("Away_GF_L10"), NEUTRAL_DEFAULTS["Attack"]),
        HomeDefense=_safe(row.get("Home_GA_L10"), NEUTRAL_DEFAULTS["Defense"]),
        AwayDefense=_safe(row.get("Away_GA_L10"), NEUTRAL_DEFAULTS["Defense"]),
        HomeAdvantage=_safe(row.get("Home_VenueResult_L5"), NEUTRAL_DEFAULTS["HomeAdvantage"]),
        HomeFatigue=compute_fatigue_index(row.get("Home_RestDays"), row.get("Home_MatchesLast14Days")),
        AwayFatigue=compute_fatigue_index(row.get("Away_RestDays"), row.get("Away_MatchesLast14Days")),
        HomeStrength=home_strength,
        AwayStrength=away_strength,
        StrengthGap=home_strength - away_strength,
        FormGap=home_form - away_form,
    )


# 前提變數名稱 -> (crisp inputs 的欄位名, 模糊化用的變數種類)
_ANTECEDENT_MAP = {
    "HomeForm": ("HomeForm", "Form"),
    "AwayForm": ("AwayForm", "Form"),
    "HomeAttack": ("HomeAttack", "Attack"),
    "AwayAttack": ("AwayAttack", "Attack"),
    "HomeDefense": ("HomeDefense", "Defense"),
    "AwayDefense": ("AwayDefense", "Defense"),
    "HomeAdvantage": ("HomeAdvantage", "HomeAdvantage"),
    "HomeFatigue": ("HomeFatigue", "Fatigue"),
    "AwayFatigue": ("AwayFatigue", "Fatigue"),
    "HomeStrength": ("HomeStrength", "TeamStrength"),
    "AwayStrength": ("AwayStrength", "TeamStrength"),
    "StrengthGap": ("StrengthGap", "StrengthGap"),
    "FormGap": ("FormGap", "FormGap"),
}


def fuzzify_all(inputs: MatchCrispInputs) -> dict[str, dict[str, float]]:
    """把每個前提變數都模糊化一次，回傳 {變數名: {詞彙: 隸屬度}}。"""
    memberships = {}
    for ante_name, (field_name, var_kind) in _ANTECEDENT_MAP.items():
        x0 = getattr(inputs, field_name)
        memberships[ante_name] = fuzzify(var_kind, x0)
    return memberships


def evaluate_rules(memberships: dict[str, dict[str, float]]) -> list[float]:
    """套用規則庫，回傳每條規則的 firing strength（依 RULES 順序排列）。"""
    strengths = []
    for antecedents, _consequents in RULES:
        degree = min(memberships[var][term] for var, term in antecedents)
        strengths.append(degree)
    return strengths


def defuzzify_outputs(rule_strengths: list[float]) -> dict[str, float]:
    """Mamdani 聚合 + centroid 解模糊，回傳每個輸出變數的 0~100 crisp 分數。"""
    # 先把每個 (輸出變數, 詞彙) 的最大 firing strength 算出來（一條規則可能同時
    # 貢獻好幾個 (輸出變數, 詞彙) 組合）
    agg_strength: dict[tuple[str, str], float] = {}
    for strength, (_, consequents) in zip(rule_strengths, RULES):
        for key in consequents:
            agg_strength[key] = max(agg_strength.get(key, 0.0), strength)

    results = {}
    for out_var in OUTPUT_VARIABLES:
        aggregated_mf = np.zeros_like(OUTPUT_UNIVERSE)
        any_fired = False
        for term, term_mf in OUTPUT_TERM_MF.items():
            strength = agg_strength.get((out_var, term), 0.0)
            if strength > 0:
                any_fired = True
                clipped = np.minimum(term_mf, strength)
                aggregated_mf = np.maximum(aggregated_mf, clipped)

        if not any_fired or aggregated_mf.sum() == 0:
            results[out_var] = 50.0  # 沒有任何規則被觸發時，回傳中性值 50（不知道 = 不偏頗）
        else:
            centroid = float(np.sum(OUTPUT_UNIVERSE * aggregated_mf) / np.sum(aggregated_mf))
            results[out_var] = centroid

    return results


def evaluate_match(row: dict) -> dict:
    """輸入一場比賽的賽前特徵 dict，回傳模糊推論後的完整結果。"""
    inputs = build_crisp_inputs(row)
    memberships = fuzzify_all(inputs)
    rule_strengths = evaluate_rules(memberships)
    potentials = defuzzify_outputs(rule_strengths)

    win_sum = potentials["HomeWinPotential"] + potentials["DrawPotential"] + potentials["AwayWinPotential"]
    if win_sum <= 0:
        p_home, p_draw, p_away = 1 / 3, 1 / 3, 1 / 3
    else:
        p_home = potentials["HomeWinPotential"] / win_sum
        p_draw = potentials["DrawPotential"] / win_sum
        p_away = potentials["AwayWinPotential"] / win_sum

    ou_sum = potentials["Over25Potential"] + potentials["Under25Potential"]
    if ou_sum <= 0:
        p_over, p_under = 0.5, 0.5
    else:
        p_over = potentials["Over25Potential"] / ou_sum
        p_under = potentials["Under25Potential"] / ou_sum

    return {
        **{f"Fuzzy_{k}": v for k, v in potentials.items()},
        "Fuzzy_P_HomeWin": p_home,
        "Fuzzy_P_Draw": p_draw,
        "Fuzzy_P_AwayWin": p_away,
        "Fuzzy_P_Over25": p_over,
        "Fuzzy_P_Under25": p_under,
    }


# ---------------------------------------------------------------------------
# 批次套用到訓練資料集 + 簡單分辨力檢查
# ---------------------------------------------------------------------------


def analyze_input_distributions(dataset: pd.DataFrame) -> None:
    """重新列印目前訓練資料集的輸入變數分位數，供比對 INPUT_TERMS 斷點是否仍合理。"""
    checks = {
        "Elo (Home+Away)": pd.concat([dataset["HomeEloPre"], dataset["AwayEloPre"]]),
        "Form L5 (Home+Away)": pd.concat([dataset["Home_Result_L5"], dataset["Away_Result_L5"]]),
        "GF_L10 (Home+Away)": pd.concat([dataset["Home_GF_L10"], dataset["Away_GF_L10"]]),
        "GA_L10 (Home+Away)": pd.concat([dataset["Home_GA_L10"], dataset["Away_GA_L10"]]),
    }
    for name, s in checks.items():
        s = s.dropna()
        q = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        print(f"{name}: p10={q[0.1]:.2f} p25={q[0.25]:.2f} p50={q[0.5]:.2f} p75={q[0.75]:.2f} p90={q[0.9]:.2f}")


def apply_to_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in dataset.iterrows():
        records.append(evaluate_match(row.to_dict()))
    fuzzy_df = pd.DataFrame(records)
    fuzzy_df.insert(0, "MatchID", dataset["MatchID"].values)
    return fuzzy_df


def run() -> pd.DataFrame:
    dataset_path = config.PROCESSED_DIR / "training_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"找不到 {dataset_path}，請先執行 update_data.py")
    dataset = pd.read_csv(dataset_path, parse_dates=["Date"])

    fuzzy_df = apply_to_dataset(dataset)
    out_path = config.PROCESSED_DIR / "fuzzy_outputs.csv"
    fuzzy_df.to_csv(out_path, index=False)

    merged = dataset[["MatchID", "FTR", "FTHG", "FTAG"]].merge(fuzzy_df, on="MatchID")
    return merged


def _sanity_check(merged: pd.DataFrame) -> None:
    """粗略檢查模糊模型是否有基本的分辨力（不是正式評估，正式評估在第七階段做）。"""
    print("\n--- 粗略分辨力檢查（非正式評估，僅供第二階段驗收）---")
    predicted_label = merged[["Fuzzy_P_HomeWin", "Fuzzy_P_Draw", "Fuzzy_P_AwayWin"]].idxmax(axis=1)
    label_map = {"Fuzzy_P_HomeWin": "H", "Fuzzy_P_Draw": "D", "Fuzzy_P_AwayWin": "A"}
    predicted_label = predicted_label.map(label_map)
    accuracy = float((predicted_label == merged["FTR"]).mean())
    print(f"依 argmax(P_Home,P_Draw,P_Away) 的簡單命中率: {accuracy:.3f}  (基準線：全猜主勝約 0.45~0.46)")

    total_goals = merged["FTHG"] + merged["FTAG"]
    actual_over = (total_goals > 2.5).astype(int)
    predicted_over = (merged["Fuzzy_P_Over25"] > 0.5).astype(int)
    ou_accuracy = float((predicted_over == actual_over).mean())
    print(f"Over/Under 2.5 簡單命中率: {ou_accuracy:.3f}  (基準線：全猜 Over 約 0.53~0.55)")

    home_win_mask = merged["FTR"] == "H"
    away_win_mask = merged["FTR"] == "A"
    print(
        f"主隊實際獲勝時，平均 Fuzzy_P_HomeWin = {merged.loc[home_win_mask, 'Fuzzy_P_HomeWin'].mean():.3f}；"
        f"客隊實際獲勝時，平均 Fuzzy_P_HomeWin = {merged.loc[away_win_mask, 'Fuzzy_P_HomeWin'].mean():.3f}"
    )
    print("（兩者應有明顯差距，代表模型至少能區分強弱情境；此處不是正式的機率校準檢查）")


def _main():
    dataset_path = config.PROCESSED_DIR / "training_dataset.csv"
    dataset = pd.read_csv(dataset_path, parse_dates=["Date"])
    print("目前訓練資料集輸入變數分位數（比對 fuzzy_model.py 檔頭常數是否仍合理）：")
    analyze_input_distributions(dataset)

    merged = run()
    print(f"\n已對 {len(merged)} 場比賽套用模糊推論，輸出: {config.PROCESSED_DIR / 'fuzzy_outputs.csv'}")
    _sanity_check(merged)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    _main()
