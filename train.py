#!/usr/bin/env python
"""
第三 + 第四階段主要入口：
時間序切分 -> 訓練/比較所有模型 -> Ensemble 加權 -> 機率校準 -> Test 集最終誠實檢查
-> 存成有版本號的模型資料夾。

用法：
    python train.py

三層防漏設計（呼應規格「三、禁止資料洩漏」與「十一、模型校準」）：
  1. 所有 LogReg/RF/XGBoost/LightGBM/CatBoost/Poisson 只用 Train 訓練。
  2. Ensemble 權重、機率校準 (Platt Scaling) 只用 Validation 資料 fit。
  3. Test 賽季在整個流程中，直到最後一步「STEP 7」才第一次、也是唯一一次被用來
     計算最終指標——不會被拿來挑模型、調權重、或選校準方法，確保 Test 的表現
     是對「這整套流程能不能推廣到真的沒看過的資料」的誠實估計。

每次執行都會建立一個新的 models/model_vNNN/ 資料夾，不會覆蓋先前版本。
"""

from __future__ import annotations

import json
import sys
import datetime as _dt

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd

from src import calibration, config, ensemble, evaluation, ml_models, poisson_model, time_split

ENSEMBLE_MEMBER_CANDIDATES = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost", "Poisson", "Fuzzy"]


def next_model_version_dir() -> tuple[str, "object"]:
    existing = sorted(config.MODELS_DIR.glob("model_v*"))
    nums = []
    for p in existing:
        try:
            nums.append(int(p.name.replace("model_v", "")))
        except ValueError:
            continue
    next_num = (max(nums) + 1) if nums else 1
    version = f"v{next_num:03d}"
    out_dir = config.MODELS_DIR / f"model_{version}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return version, out_dir


def load_datasets():
    dataset_path = config.PROCESSED_DIR / "training_dataset.csv"
    fuzzy_path = config.PROCESSED_DIR / "fuzzy_outputs.csv"
    matches_path = config.PROCESSED_DIR / "matches_clean.csv"

    for p in (dataset_path, fuzzy_path, matches_path):
        if not p.exists():
            raise FileNotFoundError(f"找不到 {p}，請先執行 update_data.py 以及 python -m src.fuzzy_model")

    dataset = pd.read_csv(dataset_path, parse_dates=["Date"])
    fuzzy = pd.read_csv(fuzzy_path)
    matches_clean = pd.read_csv(matches_path, parse_dates=["Date"])

    dataset = dataset.merge(fuzzy, on="MatchID", how="left")
    return dataset, matches_clean


def collect_split_proba(
    split_df: pd.DataFrame,
    X_split: np.ndarray,
    fitted_models: dict,
    dc_model: poisson_model.DixonColesModel,
) -> dict[str, np.ndarray]:
    """對已經訓練好（只用 Train fit 過）的所有模型，套用到任意一個切分（Val 或 Test）上。

    Poisson 用凍結好的 dc_model 參數推論；Fuzzy 直接讀取 split_df 裡已經合併好的
    Fuzzy_P_* 欄位（模糊規則本來就是固定的，不需要、也不會用任何切分的資料重新配適）。
    """
    proba: dict[str, np.ndarray] = {}
    for name, model in fitted_models.items():
        proba[name] = ml_models.predict_proba_aligned(model, X_split)

    poisson_preds = poisson_model.predict_matches_df(dc_model, split_df[["MatchID", "HomeTeam", "AwayTeam"]])
    merged = split_df[["MatchID"]].merge(poisson_preds, on="MatchID", how="left")
    proba["Poisson"] = merged[["Poisson_P_HomeWin", "Poisson_P_Draw", "Poisson_P_AwayWin"]].to_numpy(dtype=float)

    proba["Fuzzy"] = split_df[["Fuzzy_P_HomeWin", "Fuzzy_P_Draw", "Fuzzy_P_AwayWin"]].to_numpy(dtype=float)
    return proba


def build_comparison_df(reports: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, r in reports.items():
        rows.append({
            "model": name,
            "accuracy": round(r["accuracy"], 4),
            "log_loss": round(r["log_loss"], 4),
            "brier_score": round(r["brier_score"], 4),
            "roc_auc_ovr_macro": round(r["roc_auc_ovr_macro"], 4) if r["roc_auc_ovr_macro"] is not None else None,
            "ece_home_win": round(r["ece_home_win"], 4),
            "acc_H": round(r["per_class_accuracy"]["H"], 3),
            "acc_D": round(r["per_class_accuracy"]["D"], 3),
            "acc_A": round(r["per_class_accuracy"]["A"], 3),
        })
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


def main():
    print("=" * 70)
    print("STEP 1/7: 載入資料 + 時間序切分")
    print("=" * 70)
    dataset, matches_clean = load_datasets()
    train_df, val_df, test_df, split_info = time_split.chronological_season_split(dataset)
    print(f"  Train: {split_info['train_seasons']} ({split_info['train_rows']} 場)")
    print(f"  Validation: {split_info['val_seasons']} ({split_info['val_rows']} 場)")
    print(f"  Test（保留到 STEP 7 才第一次使用）: {split_info['test_seasons']} ({split_info['test_rows']} 場)")
    print(f"  目前賽季（排除在外，保留給 predict.py）: {split_info['excluded_current_season']}")

    feature_cols = ml_models.get_feature_columns(dataset)
    all_nan_in_train = [c for c in feature_cols if train_df[c].notna().sum() == 0]
    if all_nan_in_train:
        print("  排除在 Train 期間完全沒有觀測值的欄位（目前資料涵蓋率為 0%，留著只會被當雜訊):")
        print(f"    {all_nan_in_train}")
        feature_cols = [c for c in feature_cols if c not in all_nan_in_train]
    print(f"  特徵欄位數: {len(feature_cols)}")

    X_train, y_train = ml_models.prepare_xy(train_df, feature_cols)
    X_val, y_val = ml_models.prepare_xy(val_df, feature_cols)
    X_test, y_test = ml_models.prepare_xy(test_df, feature_cols)

    print("\n" + "=" * 70)
    print("STEP 2/7: 訓練機器學習模型 + Poisson（只用 Train）")
    print("=" * 70)

    fitted_models: dict[str, object] = {}
    registry = ml_models.build_model_registry()
    for name in ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]:
        if name not in registry:
            print(f"  [跳過] {name}：套件未安裝於目前環境")
            continue
        model = registry[name]
        model.fit(X_train, y_train)
        fitted_models[name] = model
        print(f"  [完成] {name} 已在 Train 上訓練完成")

    train_season_years = [int(s.split("-")[0]) for s in split_info["train_seasons"]]
    train_matches = matches_clean[matches_clean["SeasonStartYear"].isin(train_season_years)]
    dc_model = poisson_model.fit_dixon_coles(train_matches)
    print("  [完成] Poisson(Dixon-Coles) 已在 Train 上估計完成")

    print("\n" + "=" * 70)
    print("STEP 3/7: 在 Validation 上評估每個模型 + 樸素基準線")
    print("=" * 70)
    val_proba_store = collect_split_proba(val_df, X_val, fitted_models, dc_model)

    train_freq = np.array([np.mean(y_train == i) for i in range(3)])
    val_proba_store["Baseline(TrainFreq)"] = np.tile(train_freq, (len(y_val), 1))

    val_reports = {name: evaluation.full_report(y_val, proba) for name, proba in val_proba_store.items()}
    for name, r in val_reports.items():
        print(f"  {name}: accuracy={r['accuracy']:.3f} log_loss={r['log_loss']:.3f} brier={r['brier_score']:.3f}")

    val_comparison_df = build_comparison_df(val_reports)
    print("\n" + val_comparison_df.to_string(index=False))

    baseline_ll = val_reports["Baseline(TrainFreq)"]["log_loss"]
    beats_baseline = val_comparison_df[(val_comparison_df["model"] != "Baseline(TrainFreq)") & (val_comparison_df["log_loss"] < baseline_ll)]
    print(f"\n  分析：{len(beats_baseline)}/{len(val_comparison_df)-1} 個模型的 Validation log loss 優於樸素基準線"
          f"（baseline log_loss={baseline_ll:.4f}）。")

    draw_accs = {name: r["per_class_accuracy"]["D"] for name, r in val_reports.items() if name != "Baseline(TrainFreq)"}
    near_zero_draw = [n for n, a in draw_accs.items() if a < 0.05]
    if near_zero_draw:
        print(f"  分析：{near_zero_draw} 在 Validation 上幾乎完全不會預測「和局」——這是足球預測常見的已知現象，"
              f"下面的 Ensemble 步驟會納入 Fuzzy（Draw accuracy 較高）來緩解，而不是硬調單一模型的參數。")

    print("\n" + "=" * 70)
    print("STEP 4/7: Ensemble（依 Validation log loss 分配權重，softmax 加權平均機率）")
    print("=" * 70)
    ensemble_members = [n for n in ENSEMBLE_MEMBER_CANDIDATES if n in val_proba_store]
    ensemble_weights = ensemble.compute_log_loss_weights(val_reports, ensemble_members)
    for name, w in sorted(ensemble_weights.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: weight={w:.3f}  (val log_loss={val_reports[name]['log_loss']:.4f})")

    ensemble_proba_val = ensemble.weighted_ensemble_proba(
        {name: val_proba_store[name] for name in ensemble_members}, ensemble_weights
    )
    ensemble_report_val = evaluation.full_report(y_val, ensemble_proba_val)
    val_reports["Ensemble"] = ensemble_report_val
    best_single_ll = val_comparison_df.iloc[0]["log_loss"]
    print(f"\n  Ensemble (Validation): accuracy={ensemble_report_val['accuracy']:.3f} "
          f"log_loss={ensemble_report_val['log_loss']:.3f} brier={ensemble_report_val['brier_score']:.3f} "
          f"acc_D={ensemble_report_val['per_class_accuracy']['D']:.3f}")
    if ensemble_report_val["log_loss"] < best_single_ll:
        print(f"  Ensemble log loss ({ensemble_report_val['log_loss']:.4f}) 優於最佳單一模型 ({best_single_ll:.4f}) —— 有加分。")
    else:
        print(f"  Ensemble log loss ({ensemble_report_val['log_loss']:.4f}) 沒有優於最佳單一模型 ({best_single_ll:.4f})，"
              f"但 Draw accuracy={ensemble_report_val['per_class_accuracy']['D']:.3f} 通常會比只用最佳單一模型高，"
              f"這是 Ensemble 追求穩定性/機率合理性、而非單純堆高 accuracy 的預期取捨。")

    print("\n" + "=" * 70)
    print("STEP 5/7: 機率校準（Platt Scaling，只用 Validation fit；Isotonic 僅供對照）")
    print("=" * 70)
    platt_calibrator = calibration.fit_calibrator(ensemble_proba_val, y_val, method=calibration.METHOD_PLATT)
    iso_calibrator = calibration.fit_calibrator(ensemble_proba_val, y_val, method=calibration.METHOD_ISOTONIC)

    calibrated_val_platt = calibration.apply_calibrator(platt_calibrator, ensemble_proba_val)
    calibrated_val_iso = calibration.apply_calibrator(iso_calibrator, ensemble_proba_val)
    report_platt_val = evaluation.full_report(y_val, calibrated_val_platt)
    report_iso_val = evaluation.full_report(y_val, calibrated_val_iso)

    print(f"  校準前 (Validation, 用來 fit 校準本身，數字會偏樂觀，只供參考): "
          f"ece_home_win={ensemble_report_val['ece_home_win']:.4f}")
    print(f"  Platt 校準後 (Validation)   : ece_home_win={report_platt_val['ece_home_win']:.4f} log_loss={report_platt_val['log_loss']:.4f}")
    print(f"  Isotonic 校準後 (Validation): ece_home_win={report_iso_val['ece_home_win']:.4f} log_loss={report_iso_val['log_loss']:.4f}")
    print("  （這裡的數字是用同一批 Validation 資料 fit 又拿來檢查，本來就會偏樂觀；"
          "真正有意義的校準效果要看 STEP 7 在 Test 上的結果）")
    print(f"  預設採用 Platt Scaling（原因：Validation 只有 {len(y_val)} 場比賽，"
          f"樣本量偏小，Isotonic 這種無母數方法在小樣本下容易過擬合雜訊——"
          f"這個選擇是看到 Test 結果之前就決定的，不是挑 Test 表現比較好的那個）。")

    print("\n" + "=" * 70)
    print("STEP 6/7: 用 Fuzzy 的 46 條規則邏輯，列出本次 Ensemble 權重的重點觀察")
    print("=" * 70)
    print(f"  Fuzzy 模型 Ensemble 權重 = {ensemble_weights.get('Fuzzy', 0):.3f}"
          f"（Validation log loss 較差，權重自然被壓低，但仍保留一定占比，"
          f"因為它是目前唯一對 Draw 有分辨力的模型，對 Ensemble 的 Draw 預測有幫助）")

    print("\n" + "=" * 70)
    print("STEP 7/7: Test 集最終誠實檢查（整個流程中唯一一次使用 Test）")
    print("=" * 70)
    test_proba_store = collect_split_proba(test_df, X_test, fitted_models, dc_model)
    test_reports_individual = {name: evaluation.full_report(y_test, proba) for name, proba in test_proba_store.items()}

    ensemble_proba_test = ensemble.weighted_ensemble_proba(
        {name: test_proba_store[name] for name in ensemble_members}, ensemble_weights
    )
    ensemble_report_test = evaluation.full_report(y_test, ensemble_proba_test)

    calibrated_test_platt = calibration.apply_calibrator(platt_calibrator, ensemble_proba_test)
    calibrated_report_test = evaluation.full_report(y_test, calibrated_test_platt)

    best_single_test_name = min(test_reports_individual, key=lambda n: test_reports_individual[n]["log_loss"])
    print(f"  最佳單一模型 (Test): {best_single_test_name} "
          f"log_loss={test_reports_individual[best_single_test_name]['log_loss']:.4f} "
          f"accuracy={test_reports_individual[best_single_test_name]['accuracy']:.3f}")
    print(f"  Ensemble 原始機率 (Test): log_loss={ensemble_report_test['log_loss']:.4f} "
          f"accuracy={ensemble_report_test['accuracy']:.3f} ece_home_win={ensemble_report_test['ece_home_win']:.4f}")
    print(f"  Ensemble + Platt 校準 (Test): log_loss={calibrated_report_test['log_loss']:.4f} "
          f"accuracy={calibrated_report_test['accuracy']:.3f} ece_home_win={calibrated_report_test['ece_home_win']:.4f}")

    if calibrated_report_test["ece_home_win"] < ensemble_report_test["ece_home_win"]:
        print("  => 校準在 Test 上確實降低了主勝機率的校準誤差 (ECE)，不是只在 Validation 上好看。")
    else:
        print("  => 校準在 Test 上沒有明顯改善 ECE，誠實記錄；可能是 Validation 樣本量太小、"
              "校準本身學到的是雜訊，之後累積更多賽季資料後應重新檢視。")

    print("\n" + "=" * 70)
    print("儲存有版本號的模型（不覆蓋舊版本）")
    print("=" * 70)
    version, out_dir = next_model_version_dir()

    for name, model in fitted_models.items():
        joblib.dump(model, out_dir / f"{name}.joblib")
    poisson_model.save_model(dc_model, out_dir / "poisson_dixon_coles.json")
    joblib.dump(platt_calibrator, out_dir / "calibrator_platt.joblib")
    joblib.dump(iso_calibrator, out_dir / "calibrator_isotonic.joblib")

    (out_dir / "feature_columns.json").write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "split_info.json").write_text(json.dumps(split_info, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "ensemble_weights.json").write_text(json.dumps(ensemble_weights, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics_validation.json").write_text(json.dumps(val_reports, ensure_ascii=False, indent=2), encoding="utf-8")

    test_summary = {
        "individual_models": test_reports_individual,
        "ensemble_raw": ensemble_report_test,
        "ensemble_calibrated_platt": calibrated_report_test,
    }
    (out_dir / "metrics_test_final_check.json").write_text(json.dumps(test_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    val_comparison_df.to_csv(out_dir / "comparison_table_validation.csv", index=False)

    model_card = f"""# Model {version}

訓練時間: {_dt.datetime.now().isoformat(timespec='seconds')}

## 資料切分
- Train: {split_info['train_seasons']} ({split_info['train_rows']} 場)
- Validation: {split_info['val_seasons']} ({split_info['val_rows']} 場) —— 用於模型比較、Ensemble 權重、校準 fit
- Test: {split_info['test_seasons']} ({split_info['test_rows']} 場) —— 只在最後一步用過一次，未參與任何調整

## Validation 表現比較（依 Log Loss 排序）

{val_comparison_df.to_markdown(index=False)}

## Ensemble 權重（依 Validation log loss 的 softmax）

{chr(10).join(f"- {name}: {w:.3f}" for name, w in sorted(ensemble_weights.items(), key=lambda kv: -kv[1]))}

## Test 集最終誠實檢查（整個流程唯一一次使用 Test）

- 最佳單一模型: {best_single_test_name}，log_loss={test_reports_individual[best_single_test_name]['log_loss']:.4f}
- Ensemble 原始: log_loss={ensemble_report_test['log_loss']:.4f}, accuracy={ensemble_report_test['accuracy']:.3f}, ece_home_win={ensemble_report_test['ece_home_win']:.4f}
- Ensemble + Platt 校準: log_loss={calibrated_report_test['log_loss']:.4f}, accuracy={calibrated_report_test['accuracy']:.3f}, ece_home_win={calibrated_report_test['ece_home_win']:.4f}

## 說明
- Poisson 模型只用 Train 賽季的比賽結果估計球隊攻防參數，套用到 Validation/Test 時參數是凍結的。
- Fuzzy 模型的隸屬函數/規則庫在特徵工程階段就已經固定，沒有用任何 Train/Validation/Test 資料重新配適。
- Ensemble 權重與校準器都只用 Validation fit，Test 只用來做最後一次誠實檢查，沒有拿 Test 表現去挑模型/調權重/選校準方法。
"""
    (out_dir / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")

    print(f"\n  已儲存於: {out_dir}")
    print(f"  版本: {version}")
    print("\n全部完成。")


if __name__ == "__main__":
    main()
