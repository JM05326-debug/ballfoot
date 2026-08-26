"""
資料收集層：從外部資料來源下載英超比賽資料，並保存原始檔案（不覆蓋歷史賽季）。

資料來源（皆為公開、免費、可長期使用）：
1. football-data.co.uk
   - 提供每場「已賽」比賽的比分、射門、射正、角球、犯規、黃紅牌、裁判
   - 2025/26 賽季起額外提供 HxG / AxG（賽後 xG，非賽前）
   - 不提供未來賽程、控球率、傷停名單、先發陣容
2. fixturedownload.com
   - 提供「完整賽季賽程」(380 場，含未開踢場次)，含日期、輪次、主客場
   - 用於計算休息天數、賽程密度，以及取得尚未被 football-data.co.uk 收錄的未來賽事

若任何一個資料來源在下載當下失敗（HTTP 錯誤、逾時、格式不符預期），
一律拋出 DataSourceError，由呼叫端明確印出 "DATA SOURCE ERROR"，
絕不用假資料或上次快取結果偷偷頂替並假裝成功。

歷史（已結束）賽季下載成功後會快取在 data/raw/，不會每次執行都重新下載覆蓋；
只有「目前賽季」（因為比賽持續在踢）會每次強制重新下載更新。
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import datetime as _dt
from dataclasses import dataclass

import pandas as pd
import requests

from . import config


class DataSourceError(Exception):
    """資料來源下載失敗或格式不符預期時拋出。"""


@dataclass
class FetchResult:
    start_year: int
    path: "str"
    row_count: int
    source: str
    used_cache: bool


def _http_get_csv(url: str, source_name: str) -> pd.DataFrame:
    try:
        resp = requests.get(url, timeout=config.REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise DataSourceError(
            f"DATA SOURCE ERROR: 無法連線至 {source_name} ({url})：{exc}"
        ) from exc

    if resp.status_code != 200:
        raise DataSourceError(
            f"DATA SOURCE ERROR: {source_name} 回傳 HTTP {resp.status_code} ({url})"
        )

    text = resp.content.decode("utf-8-sig", errors="replace")
    if not text.strip():
        raise DataSourceError(f"DATA SOURCE ERROR: {source_name} 回傳空白內容 ({url})")

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:  # noqa: BLE001 - 任何解析失敗都視為資料來源錯誤
        raise DataSourceError(
            f"DATA SOURCE ERROR: 無法解析 {source_name} 回傳的 CSV ({url})：{exc}"
        ) from exc

    if df.empty or df.shape[1] < 3:
        raise DataSourceError(
            f"DATA SOURCE ERROR: {source_name} 回傳的資料格式異常，欄位數={df.shape[1]} ({url})"
        )

    return df


def _meta_path(raw_path):
    return raw_path.with_suffix(raw_path.suffix + ".meta.json")


def _write_meta(raw_path, row_count: int, source_url: str) -> None:
    meta = {
        "downloaded_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "row_count": row_count,
        "source_url": source_url,
    }
    _meta_path(raw_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def download_results_season(start_year: int, force: bool = False) -> FetchResult:
    """下載/更新單一賽季的已賽比賽統計資料。

    - 目前賽季：永遠重新下載（比賽持續進行中）
    - 已結束的歷史賽季：若本地已有快取檔，預設不重新下載（force=True 可強制）
    """
    raw_path = config.RAW_DIR / f"results_{start_year}.csv"
    is_current_season = start_year == config.current_season_start_year()

    if raw_path.exists() and not force and not is_current_season:
        cached = pd.read_csv(raw_path, parse_dates=["Date"], dayfirst=True)
        return FetchResult(start_year, str(raw_path), len(cached), "football-data.co.uk (cache)", True)

    url = config.football_data_url(start_year)
    df = _http_get_csv(url, "football-data.co.uk")

    required_cols = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    missing = required_cols - set(df.columns)
    if missing:
        raise DataSourceError(
            f"DATA SOURCE ERROR: football-data.co.uk 賽季 {start_year} 缺少必要欄位: {missing}"
        )

    df = df.copy()
    df = df.assign(
        HomeTeam=df["HomeTeam"].apply(config.normalize_team_name),
        AwayTeam=df["AwayTeam"].apply(config.normalize_team_name),
        SeasonStartYear=start_year,
        Season=config.season_label(start_year),
    ).copy()

    df.to_csv(raw_path, index=False)
    _write_meta(raw_path, len(df), url)
    return FetchResult(start_year, str(raw_path), len(df), "football-data.co.uk", False)


def download_fixtures_season(start_year: int) -> FetchResult:
    """下載/更新單一賽季完整賽程（含未來場次），永遠重新下載以取得最新賽果與日期異動。"""
    raw_path = config.RAW_DIR / f"fixtures_{start_year}.csv"
    url = config.fixture_download_url(start_year)
    df = _http_get_csv(url, "fixturedownload.com")

    required_cols = {"Round Number", "Date", "Home Team", "Away Team"}
    missing = required_cols - set(df.columns)
    if missing:
        raise DataSourceError(
            f"DATA SOURCE ERROR: fixturedownload.com 賽季 {start_year} 缺少必要欄位: {missing}"
        )

    df = df.assign(
        **{
            "Home Team": df["Home Team"].apply(config.normalize_team_name),
            "Away Team": df["Away Team"].apply(config.normalize_team_name),
        },
        SeasonStartYear=start_year,
        Season=config.season_label(start_year),
    ).copy()

    df.to_csv(raw_path, index=False)
    _write_meta(raw_path, len(df), url)
    return FetchResult(start_year, str(raw_path), len(df), "fixturedownload.com", False)


def update_all(seasons_back: int, force: bool = False) -> dict:
    """下載/更新最近 N 個賽季（含目前賽季）的比賽結果與完整賽程。

    seasons_back=9 代表包含目前賽季在內，共下載 9 個賽季的歷史資料。
    """
    current_start = config.current_season_start_year()
    start_years = [current_start - i for i in range(seasons_back)]
    start_years.sort()

    results_fetches: list[FetchResult] = []
    fixture_fetches: list[FetchResult] = []
    errors: list[str] = []

    for sy in start_years:
        try:
            results_fetches.append(download_results_season(sy, force=force))
        except DataSourceError as exc:
            print(str(exc), file=sys.stderr)
            errors.append(str(exc))

        try:
            fixture_fetches.append(download_fixtures_season(sy))
        except DataSourceError as exc:
            print(str(exc), file=sys.stderr)
            errors.append(str(exc))

    return {
        "seasons": start_years,
        "results_fetches": results_fetches,
        "fixture_fetches": fixture_fetches,
        "errors": errors,
    }


def load_combined_results(seasons_back: int | None = None) -> pd.DataFrame:
    """讀取本地已下載的歷史比賽結果並合併成單一 DataFrame（僅讀取本地檔案，不下載）。"""
    current_start = config.current_season_start_year()
    files = sorted(config.RAW_DIR.glob("results_*.csv"))
    if not files:
        raise DataSourceError(
            "DATA SOURCE ERROR: 找不到任何本地比賽結果資料，請先執行 update_data.py"
        )

    frames = []
    for f in files:
        start_year = int(f.stem.replace("results_", ""))
        if seasons_back is not None and start_year < current_start - seasons_back + 1:
            continue
        frames.append(pd.read_csv(f, parse_dates=["Date"], dayfirst=True))

    combined = pd.concat(frames, ignore_index=True)
    return combined


def load_combined_fixtures(seasons_back: int | None = None) -> pd.DataFrame:
    """讀取本地已下載的完整賽程並合併成單一 DataFrame（僅讀取本地檔案，不下載）。"""
    current_start = config.current_season_start_year()
    files = sorted(config.RAW_DIR.glob("fixtures_*.csv"))
    if not files:
        raise DataSourceError(
            "DATA SOURCE ERROR: 找不到任何本地賽程資料，請先執行 update_data.py"
        )

    frames = []
    for f in files:
        start_year = int(f.stem.replace("fixtures_", ""))
        if seasons_back is not None and start_year < current_start - seasons_back + 1:
            continue
        frames.append(pd.read_csv(f, parse_dates=["Date"], dayfirst=True))

    combined = pd.concat(frames, ignore_index=True)
    return combined


def _main():
    parser = argparse.ArgumentParser(description="下載/更新英超比賽資料")
    parser.add_argument("--seasons-back", type=int, default=9, help="包含目前賽季在內，下載幾個賽季")
    parser.add_argument("--force", action="store_true", help="強制重新下載，包括已結束的歷史賽季")
    args = parser.parse_args()

    summary = update_all(seasons_back=args.seasons_back, force=args.force)

    print(f"處理賽季: {[config.season_label(y) for y in summary['seasons']]}")
    for r in summary["results_fetches"]:
        tag = "(快取)" if r.used_cache else "(已下載)"
        print(f"  結果 {config.season_label(r.start_year)}: {r.row_count} 場 {tag}")
    for r in summary["fixture_fetches"]:
        print(f"  賽程 {config.season_label(r.start_year)}: {r.row_count} 場")

    if summary["errors"]:
        print(f"\n共 {len(summary['errors'])} 個 DATA SOURCE ERROR，請檢視上方訊息。", file=sys.stderr)
        sys.exit(1)

    print("\n資料下載/更新完成。")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    _main()
