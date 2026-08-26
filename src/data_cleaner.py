"""
資料清洗 + 驗證層。

輸入：data/raw/results_*.csv, data/raw/fixtures_*.csv（由 data_loader.py 產生）
輸出：
  - data/processed/matches_clean.csv   已賽比賽，清洗後的乾淨資料
  - data/processed/fixtures_full.csv   完整賽程日曆（已賽 + 未來場次），供排程/休息天數特徵使用
  - data/processed/data_quality_report.json  資料品質報告（缺值比例、重複筆數、異常筆數、
                                              各欄位可用性 -- 誠實列出「拿不到的資料」）

原則：只清洗/校驗/移除明確錯誤或重複的資料，絕不捏造缺失欄位的數值。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, data_loader

REQUIRED_CORE_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

# 目前系統會嘗試蒐集但可能因來源而異的欄位，用於品質報告誠實揭露涵蓋率
OPTIONAL_STAT_COLUMNS = {
    "HxG": "主隊 xG（賽後，僅部分近期賽季提供）",
    "AxG": "客隊 xG（賽後，僅部分近期賽季提供）",
    "HS": "主隊射門數",
    "AS": "客隊射門數",
    "HST": "主隊射正數",
    "AST": "客隊射正數",
    "HC": "主隊角球數",
    "AC": "客隊角球數",
    "HF": "主隊犯規數",
    "AF": "客隊犯規數",
    "HY": "主隊黃牌數",
    "AY": "客隊黃牌數",
    "HR": "主隊紅牌數",
    "AR": "客隊紅牌數",
    "Referee": "裁判",
}

# 使用者要求的欄位中，目前資料來源完全無法取得、且未偽造的項目
KNOWN_UNAVAILABLE_FIELDS = {
    "Possession": "控球率 -- football-data.co.uk 與 fixturedownload.com 均未提供，需付費資料源（如 Opta/StatsBomb）",
    "Injuries": "傷停資訊 -- 需付費或另行爬取官方英超網站傷停名單，目前未串接",
    "Lineups": "預計/實際先發陣容 -- 同上，未串接",
    "EuropeanFixtures": "歐戰賽程（影響疲勞） -- 目前只抓取英超賽程，未整合歐冠/歐霸賽程",
}


@dataclass
class QualityReport:
    seasons: list = field(default_factory=list)
    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    missing_core_dropped: int = 0
    ftr_mismatches_fixed: int = 0
    outlier_flags: list = field(default_factory=list)
    unknown_team_names: list = field(default_factory=list)
    stat_column_coverage: dict = field(default_factory=dict)
    unavailable_fields: dict = field(default_factory=dict)
    schedule_gaps: list = field(default_factory=list)

    def to_dict(self):
        return {
            "seasons": self.seasons,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "duplicates_removed": self.duplicates_removed,
            "missing_core_dropped": self.missing_core_dropped,
            "ftr_mismatches_fixed": self.ftr_mismatches_fixed,
            "outlier_flags_count": len(self.outlier_flags),
            "outlier_flags_sample": self.outlier_flags[:20],
            "unknown_team_names": self.unknown_team_names,
            "stat_column_coverage_pct": self.stat_column_coverage,
            "unavailable_fields": self.unavailable_fields,
            "schedule_gaps_count": len(self.schedule_gaps),
            "schedule_gaps_sample": self.schedule_gaps[:20],
        }


def _check_numeric_range(df: pd.DataFrame, col: str, lo: float, hi: float, report: QualityReport):
    if col not in df.columns:
        return
    bad = df[(df[col].notna()) & ((df[col] < lo) | (df[col] > hi))]
    for _, row in bad.iterrows():
        report.outlier_flags.append(
            {
                "date": str(row["Date"]),
                "match": f"{row['HomeTeam']} vs {row['AwayTeam']}",
                "column": col,
                "value": row[col],
                "expected_range": [lo, hi],
            }
        )


def clean_results(raw: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
    df = raw.copy()
    report.rows_in = len(df)

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    df["HomeTeam"] = df["HomeTeam"].apply(config.normalize_team_name)
    df["AwayTeam"] = df["AwayTeam"].apply(config.normalize_team_name)

    before = len(df)
    missing_core = df[REQUIRED_CORE_COLUMNS].isna().any(axis=1)
    report.missing_core_dropped = int(missing_core.sum())
    df = df[~missing_core].copy()

    before = len(df)
    df = df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="first")
    report.duplicates_removed = before - len(df)

    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    computed_ftr = np.where(df["FTHG"] > df["FTAG"], "H", np.where(df["FTHG"] < df["FTAG"], "A", "D"))
    mismatch = df["FTR"] != computed_ftr
    report.ftr_mismatches_fixed = int(mismatch.sum())
    df["FTR"] = computed_ftr

    _check_numeric_range(df, "FTHG", 0, 15, report)
    _check_numeric_range(df, "FTAG", 0, 15, report)
    _check_numeric_range(df, "HS", 0, 50, report)
    _check_numeric_range(df, "AS", 0, 50, report)
    _check_numeric_range(df, "HST", 0, 35, report)
    _check_numeric_range(df, "AST", 0, 35, report)
    _check_numeric_range(df, "HC", 0, 25, report)
    _check_numeric_range(df, "AC", 0, 25, report)
    _check_numeric_range(df, "HF", 0, 40, report)
    _check_numeric_range(df, "AF", 0, 40, report)
    _check_numeric_range(df, "HY", 0, 11, report)
    _check_numeric_range(df, "AY", 0, 11, report)
    _check_numeric_range(df, "HR", 0, 5, report)
    _check_numeric_range(df, "AR", 0, 5, report)
    if "HxG" in df.columns:
        _check_numeric_range(df, "HxG", 0, 9, report)
        _check_numeric_range(df, "AxG", 0, 9, report)

    df = df.sort_values("Date").reset_index(drop=True)
    df["MatchID"] = (
        df["Date"].dt.strftime("%Y%m%d") + "_" + df["HomeTeam"].str.replace(" ", "") + "_" + df["AwayTeam"].str.replace(" ", "")
    )

    core_cols = [
        "MatchID", "Season", "SeasonStartYear", "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    ]
    stat_cols = [c for c in OPTIONAL_STAT_COLUMNS if c in df.columns]
    keep_cols = [c for c in core_cols if c in df.columns] + stat_cols
    df = df[keep_cols]

    report.rows_out = len(df)
    for col, desc in OPTIONAL_STAT_COLUMNS.items():
        if col in df.columns:
            coverage = 100.0 * df[col].notna().mean()
        else:
            coverage = 0.0
        report.stat_column_coverage[col] = round(coverage, 1)

    report.unavailable_fields = KNOWN_UNAVAILABLE_FIELDS
    return df


def build_full_fixture_calendar(fixtures_raw: pd.DataFrame, matches_clean: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
    """合併完整賽程與已賽結果，標記 Played=True/False，供排程類特徵使用。"""
    fx = fixtures_raw.copy()
    fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True, errors="coerce")
    fx["HomeTeam"] = fx["Home Team"].apply(config.normalize_team_name)
    fx["AwayTeam"] = fx["Away Team"].apply(config.normalize_team_name)
    fx = fx.rename(columns={"Round Number": "RoundNumber", "Location": "Venue"})
    fx = fx[["Season", "SeasonStartYear", "RoundNumber", "Date", "HomeTeam", "AwayTeam", "Venue"]]

    played_keys = set(zip(matches_clean["Date"].dt.date, matches_clean["HomeTeam"], matches_clean["AwayTeam"]))
    fx["Played"] = fx.apply(lambda r: (r["Date"].date(), r["HomeTeam"], r["AwayTeam"]) in played_keys if pd.notna(r["Date"]) else False, axis=1)

    today = pd.Timestamp.now().normalize()
    gaps = fx[(fx["Played"] == False) & (fx["Date"] < today - pd.Timedelta(days=2))]  # noqa: E712
    for _, row in gaps.iterrows():
        report.schedule_gaps.append(
            {
                "date": str(row["Date"].date()),
                "match": f"{row['HomeTeam']} vs {row['AwayTeam']}",
                "note": "賽程顯示應已開踢但結果資料中找不到對應比賽（可能延賽或尚未同步）",
            }
        )

    fx = fx.sort_values(["Date", "RoundNumber"]).reset_index(drop=True)
    return fx


def run(seasons_back: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, QualityReport]:
    results_raw = data_loader.load_combined_results(seasons_back=seasons_back)
    fixtures_raw = data_loader.load_combined_fixtures(seasons_back=seasons_back)

    report = QualityReport()
    report.seasons = sorted(results_raw["Season"].unique().tolist())

    matches_clean = clean_results(results_raw, report)
    fixtures_full = build_full_fixture_calendar(fixtures_raw, matches_clean, report)

    unknown = config.get_unknown_team_names()
    report.unknown_team_names = sorted(unknown)

    matches_path = config.PROCESSED_DIR / "matches_clean.csv"
    fixtures_path = config.PROCESSED_DIR / "fixtures_full.csv"
    report_path = config.PROCESSED_DIR / "data_quality_report.json"

    matches_clean.to_csv(matches_path, index=False)
    fixtures_full.to_csv(fixtures_path, index=False)
    report_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    return matches_clean, fixtures_full, report


def _main():
    matches_clean, fixtures_full, report = run()
    print(f"清洗完成：{report.rows_in} 筆 -> {report.rows_out} 筆")
    print(f"  移除重複: {report.duplicates_removed}")
    print(f"  移除核心欄位缺漏: {report.missing_core_dropped}")
    print(f"  FTR 與比分不符已修正: {report.ftr_mismatches_fixed}")
    print(f"  數值異常標記: {len(report.outlier_flags)}")
    if report.unknown_team_names:
        print(f"  警告：發現未知球隊名稱（未在對照表中）: {report.unknown_team_names}")
    print("\n欄位涵蓋率 (%):")
    for col, pct in report.stat_column_coverage.items():
        print(f"  {col}: {pct}%")
    print("\n目前完全無法取得（未串接資料源）的欄位:")
    for field_name, desc in report.unavailable_fields.items():
        print(f"  - {field_name}: {desc}")
    if report.schedule_gaps:
        print(f"\n注意：{len(report.schedule_gaps)} 場賽程顯示應已開踢但缺少結果資料（可能延賽），詳見 data_quality_report.json")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    _main()
