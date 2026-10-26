# Model v012

訓練時間: 2026-10-26T13:45:51

## 資料切分
- Train: ['2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24'] (2280 場)
- Validation: ['2024-25'] (380 場) —— 用於模型比較、Ensemble 權重、校準 fit
- Test: ['2025-26'] (380 場) —— 只在最後一步用過一次，未參與任何調整

## Validation 表現比較（依 Log Loss 排序）

| model               |   accuracy |   log_loss |   brier_score |   roc_auc_ovr_macro |   ece_home_win |   acc_H |   acc_D |   acc_A |
|:--------------------|-----------:|-----------:|--------------:|--------------------:|---------------:|--------:|--------:|--------:|
| RandomForest        |     0.5263 |     1.0078 |        0.602  |              0.6453 |         0.0643 |   0.819 |   0     |   0.553 |
| CatBoost            |     0.5316 |     1.0122 |        0.6059 |              0.6503 |         0.0745 |   0.826 |   0     |   0.561 |
| Poisson             |     0.4974 |     1.0144 |        0.6078 |              0.6431 |         0.0532 |   0.8   |   0     |   0.492 |
| LogisticRegression  |     0.4947 |     1.0278 |        0.6174 |              0.6171 |         0.0634 |   0.768 |   0.011 |   0.515 |
| Fuzzy               |     0.4447 |     1.054  |        0.6316 |              0.6184 |         0.0599 |   0.497 |   0.366 |   0.439 |
| Baseline(TrainFreq) |     0.4079 |     1.0809 |        0.6554 |              0.5    |         0.039  |   1     |   0     |   0     |
| XGBoost             |     0.5079 |     1.0899 |        0.6459 |              0.6361 |         0.1366 |   0.819 |   0.075 |   0.447 |
| LightGBM            |     0.4789 |     1.1803 |        0.6785 |              0.6231 |         0.1495 |   0.781 |   0.108 |   0.386 |

## Ensemble 權重（依 Validation log loss 的 softmax）

- RandomForest: 0.150
- CatBoost: 0.149
- Poisson: 0.149
- LogisticRegression: 0.147
- Fuzzy: 0.143
- XGBoost: 0.138
- LightGBM: 0.126

## Test 集最終誠實檢查（整個流程唯一一次使用 Test）

- 最佳單一模型: Poisson，log_loss=1.0388
- Ensemble 原始: log_loss=1.0501, accuracy=0.474, ece_home_win=0.0604
- Ensemble + Platt 校準: log_loss=1.0450, accuracy=0.463, ece_home_win=0.0308

## 說明
- Poisson 模型只用 Train 賽季的比賽結果估計球隊攻防參數，套用到 Validation/Test 時參數是凍結的。
- Fuzzy 模型的隸屬函數/規則庫在特徵工程階段就已經固定，沒有用任何 Train/Validation/Test 資料重新配適。
- Ensemble 權重與校準器都只用 Validation fit，Test 只用來做最後一次誠實檢查，沒有拿 Test 表現去挑模型/調權重/選校準方法。
