#!/usr/bin/env python
"""
第一階段主要入口：一次執行完成「下載/更新資料 -> 清洗驗證 -> 特徵工程」。

用法：
    python update_data.py
    python update_data.py --seasons-back 5
    python update_data.py --force

不需要背景排程或常駐程式，每次手動執行即可得到最新可用的訓練資料集。
"""

from __future__ import annotations

import argparse
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src import config, data_cleaner, data_loader, feature_engineering


def main():
    parser = argparse.ArgumentParser(description="更新英超比賽資料並重建訓練資料集")
    parser.add_argument("--seasons-back", type=int, default=9, help="包含目前賽季在內，下載/使用幾個賽季的歷史資料")
    parser.add_argument("--force", action="store_true", help="強制重新下載，包括已經快取的歷史賽季")
    args = parser.parse_args()

    print("=" * 70)
    print("STEP 1/3: 下載 / 更新原始資料")
    print("=" * 70)
    summary = data_loader.update_all(seasons_back=args.seasons_back, force=args.force)
    for r in summary["results_fetches"]:
        tag = "(快取，未重新下載)" if r.used_cache else "(已下載/更新)"
        print(f"  結果資料 {config.season_label(r.start_year)}: {r.row_count} 場 {tag}")
    for r in summary["fixture_fetches"]:
        print(f"  賽程資料 {config.season_label(r.start_year)}: {r.row_count} 場")

    if summary["errors"]:
        print(f"\n共發生 {len(summary['errors'])} 個 DATA SOURCE ERROR：", file=sys.stderr)
        for e in summary["errors"]:
            print(f"  - {e}", file=sys.stderr)
        print("\n已中止後續清洗/特徵工程步驟，避免使用不完整或錯誤的原始資料建立資料集。", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 70)
    print("STEP 2/3: 資料清洗與驗證")
    print("=" * 70)
    matches_clean, fixtures_full, report = data_cleaner.run(seasons_back=args.seasons_back)
    print(f"  清洗完成：{report.rows_in} 筆 -> {report.rows_out} 筆")
    print(f"  移除重複: {report.duplicates_removed}, 移除核心欄位缺漏: {report.missing_core_dropped}")
    print(f"  數值異常標記: {len(report.outlier_flags)} 筆（詳見 data/processed/data_quality_report.json）")
    if report.unknown_team_names:
        print(f"  警告：發現未知球隊名稱: {report.unknown_team_names}", file=sys.stderr)
    if report.schedule_gaps:
        print(f"  注意：{len(report.schedule_gaps)} 場賽程缺少對應結果（可能延賽），詳見報告")

    print("\n  欄位涵蓋率:")
    for col, pct in report.stat_column_coverage.items():
        print(f"    {col}: {pct}%")
    print("\n  完全無法取得（未串接資料源，不會被造假）的欄位:")
    for field_name, desc in report.unavailable_fields.items():
        print(f"    - {field_name}: {desc}")

    print("\n" + "=" * 70)
    print("STEP 3/3: 特徵工程（Elo + 近期加權狀態 + 主客場拆分 + 賽程休息）")
    print("=" * 70)
    dataset = feature_engineering.run()
    print(f"  訓練資料集: {len(dataset)} 場比賽, {dataset.shape[1]} 個欄位")
    print(f"  輸出: {config.PROCESSED_DIR / 'training_dataset.csv'}")

    print("\n全部完成。可檢查 data/processed/ 目錄下的檔案。")


if __name__ == "__main__":
    main()
