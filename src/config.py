"""
專案共用設定：路徑、資料來源網址、賽季代碼換算、球隊名稱正規化。

賽季代碼規則：
- football-data.co.uk 使用 "起始年後兩碼+結束年後兩碼"，例如 2026/27 賽季 -> "2627"
- fixturedownload.com 使用賽季起始年，例如 2026/27 賽季 -> "epl-2026"

整個專案不得寫死單一賽季，一律透過 season_start_year 動態換算。
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

# ---------------------------------------------------------------------------
# 路徑設定
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

for _d in (RAW_DIR, PROCESSED_DIR, PREDICTIONS_DIR, MODELS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 資料來源
# ---------------------------------------------------------------------------

# 歷史 + 已賽比賽統計（比分、射門、射正、角球、犯規、卡牌、部分賽季含 xG）
FOOTBALL_DATA_BASE = "https://www.football-data.co.uk/mmz4281"
FOOTBALL_DATA_LEAGUE_CODE = "E0"  # English Premier League

# 完整賽季賽程（含未來場次，用於排程 / 休息天數 / 賽程密度計算）
FIXTURE_DOWNLOAD_BASE = "https://fixturedownload.com/download"
FIXTURE_LEAGUE_SLUG = "epl"

REQUEST_TIMEOUT_SECONDS = 30

# ---------------------------------------------------------------------------
# 賽季代碼換算
# ---------------------------------------------------------------------------

# 英超賽季通常在 7 月底 ~ 8 月初開踢，因此以 7 月作為新賽季的切分月份。
SEASON_START_MONTH = 7


def current_season_start_year(today: _dt.date | None = None) -> int:
    """回傳「目前賽季」的起始年份，例如 2026/27 賽季回傳 2026。"""
    today = today or _dt.date.today()
    if today.month >= SEASON_START_MONTH:
        return today.year
    return today.year - 1


def season_code_football_data(start_year: int) -> str:
    """2026 -> '2627'"""
    end_year = start_year + 1
    return f"{str(start_year)[-2:]}{str(end_year)[-2:]}"


def season_label(start_year: int) -> str:
    """2026 -> '2026-27'"""
    end_year = start_year + 1
    return f"{start_year}-{str(end_year)[-2:]}"


def fixture_slug(start_year: int) -> str:
    """2026 -> 'epl-2026'"""
    return f"{FIXTURE_LEAGUE_SLUG}-{start_year}"


def football_data_url(start_year: int) -> str:
    code = season_code_football_data(start_year)
    return f"{FOOTBALL_DATA_BASE}/{code}/{FOOTBALL_DATA_LEAGUE_CODE}.csv"


def fixture_download_url(start_year: int) -> str:
    slug = fixture_slug(start_year)
    return f"{FIXTURE_DOWNLOAD_BASE}/{slug}-GMTStandardTime.csv"


# ---------------------------------------------------------------------------
# 球隊名稱正規化
# ---------------------------------------------------------------------------
# 以 football-data.co.uk 的命名作為系統的「正規名稱 (canonical name)」，
# 因為主要的比賽統計（射門、射正、角球等）都來自這個來源。
# fixturedownload.com 少數球隊命名不同，需要對照表轉換。
# 若賽季更換升降級球隊、又出現新的命名差異，會由 normalize_team_name()
# 記錄未知名稱，而不是靜默錯配。

FIXTURE_SOURCE_TO_CANONICAL: dict[str, str] = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
}

# 已知歷史上其他命名差異（保留擴充空間，供未來賽季升降級新隊時比對）
EXTRA_ALIASES: dict[str, str] = {
    "Man Utd": "Man United",
    "Manchester United": "Man United",
    "Manchester Utd": "Man United",
    "Manchester City": "Man City",
    "Tottenham Hotspur": "Tottenham",
    "Spurs": "Tottenham",
    "Nottingham Forest": "Nott'm Forest",
    "Nott'm Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Newcastle United": "Newcastle",
    "West Ham United": "West Ham",
    "Brighton & Hove Albion": "Brighton",
    "Leeds United": "Leeds",
    "Ipswich Town": "Ipswich",
    "Luton Town": "Luton",
    "Sheffield United": "Sheffield United",
    "Sheffield Utd": "Sheffield United",
    "Leicester City": "Leicester",
    "Norwich City": "Norwich",
    "Cardiff City": "Cardiff",
    "Huddersfield Town": "Huddersfield",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "West Bromwich Albion": "West Brom",
    "AFC Bournemouth": "Bournemouth",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Sunderland AFC": "Sunderland",
}

_unknown_team_names: set[str] = set()


def normalize_team_name(name: str) -> str:
    """將任何來源的球隊名稱轉換為系統正規名稱。

    找不到對照時，回傳原始名稱並記錄下來（可用 get_unknown_team_names() 取得），
    絕不靜默竄改或猜測，避免製造假資料。
    """
    if name is None:
        return name
    name = name.strip()
    if name in FIXTURE_SOURCE_TO_CANONICAL:
        return FIXTURE_SOURCE_TO_CANONICAL[name]
    if name in EXTRA_ALIASES:
        return EXTRA_ALIASES[name]
    return name


def register_known_team_names(names: set[str]) -> None:
    """把某個資料來源目前實際出現過的名稱記錄起來，用於之後比對未知名稱。"""
    _unknown_team_names_baseline = names


def get_unknown_team_names() -> set[str]:
    return set(_unknown_team_names)


def flag_unknown_team_name(name: str) -> None:
    _unknown_team_names.add(name)
