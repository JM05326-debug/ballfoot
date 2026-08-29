# Premier League Predictor（英超足球賽事模糊預測模型）

多階段建置的完整專案。**目前完成進度：第一~五階段 / 共多階段**。

---

## 專案架構（目前已建立的部分）

```
premier_league_predictor/
├── data/
│   ├── raw/                     # 原始下載資料（每個賽季一份，不會每次覆蓋歷史賽季）
│   ├── processed/                # 清洗後資料、品質報告、Elo、訓練資料集
│   └── predictions/               # (第二階段之後才會使用)
├── models/                       # (第二階段之後才會使用)
├── src/
│   ├── config.py                  # 路徑、賽季代碼換算、球隊名稱正規化
│   ├── data_loader.py             # 下載/更新原始資料
│   ├── data_cleaner.py            # 清洗、驗證、資料品質報告
│   ├── elo_model.py               # Elo 積分模型（賽前 Elo，無洩漏）
│   ├── feature_engineering.py     # 近期加權狀態、主客場拆分、賽程休息、整合 Elo
│   ├── fuzzy_model.py             # 模糊邏輯層（隸屬函數 + 46 條規則 + Mamdani 推論）
│   ├── time_split.py              # 嚴格時間序切分 + walk-forward 產生器
│   ├── poisson_model.py           # Dixon-Coles 獨立比分模型
│   ├── ml_models.py               # LogReg/RandomForest/XGBoost/LightGBM/CatBoost
│   ├── evaluation.py              # Accuracy/LogLoss/Brier/ROC-AUC/校準誤差 共用指標
│   ├── ensemble.py                # 依 Validation log loss 分配權重的 Ensemble
│   ├── calibration.py             # Platt Scaling / Isotonic Regression 機率校準
│   ├── backtest.py                # Walk-forward 回測（沿用固定方法論，逐賽季滾動）
│   └── prediction.py              # 對未來比賽產生完整預測（1X2+比分+O/U+BTTS+信心+影響因素）
├── update_data.py                 # 第一階段主要入口（下載 -> 清洗 -> 特徵工程）
├── train.py                       # 第三+四階段主要入口（切分->訓練->Ensemble->校準->Test 檢查->存版本）
├── evaluate.py                    # 第五階段：Walk-forward Backtest（最近 1/2/3 個賽季）
├── predict.py                     # 第五階段：對下一輪比賽產生完整預測
├── collect_results.py             # 比對預測 vs 真實結果，累積寫進 prediction_history.csv
├── generate_dashboard.py          # 把最新預測轉成 docs/index.html（GitHub Pages 用）
├── .github/workflows/             # GitHub Actions：daily.yml（4x/天）+ weekly-retrain.yml
├── docs/index.html                # GitHub Pages 網站（中文化儀表板）
├── requirements.txt
└── README.md
```

所有規格要求的檔案（`update_data.py`/`train.py`/`predict.py`/`evaluate.py` 四個手動入口，
以及 `src/` 底下全部模組）都已建立完成。

---

## 安裝方法

```bash
cd premier_league_predictor
pip install -r requirements.txt
```

目前只需要 `pandas`、`numpy`、`requests`。後續階段會再擴充 `requirements.txt`
（scikit-learn、xgboost、lightgbm、catboost、scipy、scikit-fuzzy 等），但這些套件
**尚未使用**，不需要現在安裝。

---

## 資料來源（第一階段）

| 來源 | 用途 | 涵蓋內容 | 更新頻率 |
|---|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/englandm.php) | 已賽比賽統計 | 比分、半場比分、射門、射正、角球、犯規、黃紅牌、裁判；2025/26 賽季起部分含賽後 xG | 目前賽季每次執行都重新下載；已結束賽季只下載一次並快取 |
| [fixturedownload.com](https://fixturedownload.com/) | 完整賽季賽程 | 380 場全部賽程（含未開踢場次）、輪次、日期、場地 | 每次執行都重新下載，因為未來賽程/賽果會變動 |

兩個來源都是公開、免費、不需要 API Key 的資料源，理論上可支援任何未來賽季
（賽季代碼由 `src/config.py` 依日期動態換算，沒有寫死 2026/27）。

### 目前**明確拿不到、不會被造假**的欄位

| 欄位 | 原因 |
|---|---|
| 控球率 (Possession) | 兩個免費來源都沒有，需要 Opta/StatsBomb 等付費資料源 |
| 傷停資訊 (Injuries) | 需另外串接付費 API 或爬取官方英超網站傷停名單，目前未實作 |
| 先發陣容 (Lineups) | 同上，未串接 |
| 歐戰賽程影響 | 目前只抓英超賽程，未整合歐冠/歐霸賽程資料 |
| 賽前 xG / xGA | football-data.co.uk 的 xG 欄位是「賽後」統計，不能當作賽前特徵使用（此專案完全沒有把它當特徵） |

每次執行 `update_data.py` 都會在畫面上重新印出這份清單與各欄位的實際涵蓋率，
避免你誤以為系統偷偷補了假資料。

---

## 使用方法

### 更新資料 + 重建訓練資料集（一次執行即可）

```bash
python update_data.py
```

可選參數：

```bash
python update_data.py --seasons-back 5   # 只用最近 5 個賽季（含目前賽季）
python update_data.py --force            # 強制重新下載，包含已快取的歷史賽季
```

執行流程（單次執行、不需要背景常駐程式）：

1. 下載/更新原始資料（`data/raw/`）
2. 清洗、驗證、產生資料品質報告（`data/processed/matches_clean.csv`,
   `data/processed/fixtures_full.csv`, `data/processed/data_quality_report.json`）
3. 特徵工程：計算 Elo（`data/processed/elo_ratings_latest.json`）與近期加權狀態，
   輸出訓練資料集 `data/processed/training_dataset.csv`

若任何資料來源下載失敗（HTTP 錯誤、逾時、格式不符），畫面會印出
`DATA SOURCE ERROR` 並終止後續步驟（回傳非 0 exit code），**不會**用舊資料或假資料
偷偷頂替並假裝成功。

也可以單獨執行各步驟（除錯用）：

```bash
python -m src.data_loader --seasons-back 9
python -m src.data_cleaner
python -m src.feature_engineering
```

---

## 資料如何持久保存

- `data/raw/results_<年份>.csv`：**已結束的賽季只會下載一次**，之後每次執行都直接讀取本地
  快取，不會重新覆蓋。只有「目前賽季」因為比賽持續進行，每次執行都會重新下載更新。
- `data/raw/fixtures_<年份>.csv`：每次都重新下載（賽程/賽果會變動），但仍是完整寫入獨立檔案，
  不會刪除其他賽季的檔案。
- 每個原始檔都有對應的 `.meta.json`（下載時間、筆數、來源網址），方便追蹤資料新鮮度。
- `data/processed/` 下的清洗結果與訓練資料集是「衍生資料」，每次執行 `update_data.py`
  都會依最新的 `data/raw/` 內容重新產生（這是必要的，因為目前賽季的資料會持續增加）。

---

## 防止資料洩漏的設計（第一階段已落實）

1. **Elo**：`src/elo_model.py` 逐場依時間順序計算，每場比賽先記錄「賽前」Elo
   （`HomeEloPre` / `AwayEloPre`），再用當場實際比分更新 Elo。訓練資料集只包含
   賽前 Elo，不包含賽後 Elo。
2. **近期狀態特徵**：`src/feature_engineering.py` 會先把「一場比賽」攤成「兩支球隊各一列」
   的長表，依球隊、日期排序後，對每一場比賽只使用該球隊「這場之前」最近 N 場
   （N = 3 / 5 / 10）的資料做加權平均（`RECENCY_DECAY = 0.85`，越近的比賽權重越高）。
3. **絕對不會出現的欄位**：這場比賽自己的射門、射正、角球、犯規、卡牌、xG、以及最終比分本身
   都**不會**被當作這場比賽的輸入特徵（只有 `FTHG`/`FTAG`/`FTR` 會保留在資料集中，
   但那是「訓練標籤」，之後訓練模型時必須明確排除，不能拿來當 X）。
   - 開發過程中曾經出現過一次真實的洩漏 bug：合併主客場特徵時，不小心把「這場比賽自己」的
     `Home_GF`／`Home_Result` 等原始欄位也併入了訓練資料集。已於本階段修正並重新產生資料集，
     並用人工試算核對了修正後的加權特徵數值完全正確（見下方驗收方式）。
4. **時間切分**：資料集保留完整 `Date` 欄位，尚未在此階段做 train/test 切分——
   時間切分 / rolling walk-forward validation 會在「機器學習模型」與「Backtesting」階段實作，
   屆時會強制使用時間序切分，不會用 random split 作為主要驗證方式。

---

## 第一階段驗收結果（實際執行輸出）

使用 `python update_data.py --seasons-back 9`（2018/19 ~ 2026/27，共 9 個賽季）：

- 下載賽季：2018-19 ~ 2026-27（2026-27 目前 10 場，因賽季剛開踢）
- 清洗後比賽數：3050 場，0 筆重複、0 筆核心欄位缺漏、0 筆數值異常
- 欄位涵蓋率：比分/射門/射正/角球/犯規/卡牌/裁判 = 100%；xG = 0.3%（僅目前賽季極少數比賽有）
- 訓練資料集：3050 場 × 67 欄位，輸出於 `data/processed/training_dataset.csv`
- Elo 排名合理性檢查（截至 2026-08-24）：Arsenal 1820、Man City 1800 領先；
  Southampton 1259、Sheffield United 1270 墊底 —— 與實際近年戰績方向一致
- 賽程/結果比對：發現 1 場（2023-01-19 Crystal Palace vs Man United）賽程上顯示應開踢
  但結果資料缺漏，已記錄於 `data_quality_report.json`（該場為足總盃延賽，非英超聯賽賽事，
  之後可忽略或於後續階段加入過濾邏輯）

## 如何自行驗證「沒有資料洩漏」

```bash
python -c "
import pandas as pd, numpy as np
m = pd.read_csv('data/processed/matches_clean.csv', parse_dates=['Date'])
d = pd.read_csv('data/processed/training_dataset.csv', parse_dates=['Date'])

# 任選一場比賽，人工重算 Home_GF_L5，應與資料集裡的數值完全一致
row = d[(d.HomeTeam=='Arsenal')].sort_values('Date').iloc[20]
before = m[((m.HomeTeam=='Arsenal')|(m.AwayTeam=='Arsenal')) & (m.Date < row.Date)].tail(5)
gf = np.array([r.FTHG if r.HomeTeam=='Arsenal' else r.FTAG for _,r in before.iterrows()], dtype=float)
w = 0.85 ** (len(gf)-1-np.arange(len(gf)))
print('人工試算:', (w*gf).sum()/w.sum())
print('資料集中:', row['Home_GF_L5'])
"
```

兩個數字應完全相同（在本次開發過程中已驗證過）。

---

## 目前資料不足 / 尚未實作的部分（誠實列出，不捏造）

1. 控球率、傷停名單、先發陣容、歐戰賽程 —— 見上方「明確拿不到」表格
2. H2H 對戰歷史特徵 —— 依規劃屬於下一階段（球隊特徵 E 項）
3. 模糊邏輯層、機器學習模型、Poisson/Dixon-Coles 比分模型、Ensemble、機率校準、
   Backtest、prediction_history 記錄、模型版本控制 —— 皆為後續階段，尚未開始

---

---

## 第二階段：模糊邏輯層 (Fuzzy Logic Layer)

`src/fuzzy_model.py`，手刻的 Mamdani 模糊推論系統（不依賴 scikit-fuzzy，避免額外套件相依風險，
效能也更快、更容易除錯）。

### 輸入模糊變數（隸屬函數，斷點皆來自 training_dataset.csv 實際分位數，非拍腦袋假設）

| 變數 | 詞彙 | 對應的賽前特徵 |
|---|---|---|
| Form | Very Poor / Poor / Average / Good / Very Good | `Result_L5`（近 5 場加權勝率，0~1） |
| Attack | Weak / Average / Strong | `GF_L10`（近 10 場加權場均進球） |
| Defense | Strong / Average / Weak | `GA_L10`（近 10 場加權場均失球，數值低=防守強） |
| Home Advantage | Low / Medium / High | `Home_VenueResult_L5`（主隊近 5 個主場加權勝率） |
| Fatigue | Low / Medium / High | 休息天數 + 14 天內賽程密度合成的疲勞指標（見下方說明） |
| Team Strength | Weak / Average / Strong | 賽前 Elo |
| **Strength Gap**（新增） | HomeMuchStronger ... AwayMuchStronger（5級） | 主客 Elo 差 |
| **Form Gap**（新增） | HomeFormMuchBetter ... AwayFormMuchBetter（5級） | 主客近況差 |

所有隸屬函數都是三角形/梯形，彼此重疊，因此同一個數值本來就會同時部分屬於多個集合
（例如勝率 0.6 大約 Good=0.65、Very Good=0.35），沒有任何硬切割 if/else。

### 規則庫：46 條（規格要求至少 30 條）

涵蓋：相對實力/狀態差距 → 主勝/和局/客勝潛力（主幹規則）、主場優勢、疲勞、
絕對等級組合（雙弱/雙強傾向和局）、Attack vs Defense → 進球潛力、Over/Under 2.5。
所有規則都是「主客對稱」設計（每條偏主隊的規則都有對應的偏客隊版本），
完整規則清單請直接看 `src/fuzzy_model.py` 的 `RULES` 常數。

### 開發過程中發現並修正的設計問題（誠實記錄，不是憑空宣稱「一次做對」）

1. **判別力不足的 bug**：一開始只用「絕對等級」（例如 HomeStrength=Strong AND
   AwayStrength=Weak）判斷勝負，用一個極端案例測試（主隊 Elo 1900 對客隊 1250，
   近況全勝 vs 全敗）發現主勝潛力只有 41%，和局/客勝各 30%——幾乎沒有分辨力。
   原因：規則庫只斷言「贏的那一方」，從沒斷言「另外兩個輸出該是多少」，導致沒被
   觸發的變數退回中性值 50，稀釋掉原本該有的懸殊判斷。
   **修正**：新增 StrengthGap / FormGap 兩個「相對差距」變數，用 5 級全值域覆蓋
   （任何差距值都保證至少一條規則會觸發），同一條規則同時對主勝/和局/客勝三個
   輸出下結論。修正後同一個極端案例變成主勝 64% / 和局 18% / 客勝 18%。
2. **和局過度預測**：套用到全部 3050 場真實比賽後發現，模型有 56% 的比賽預測和局
   （實際和局率只有 23%），因為「兩隊實力相近」在真實英超資料中佔了很大比例
   （Elo 差距的 25~75 分位數落在 -104~108 之間），但真實世界「勢均力敵」不等於
   「容易打平」——主場優勢通常會把close game 轉成主勝而非和局。
   **修正**：把「Similar / FormSimilar」情境下的結論從 `Draw=High` 調整為
   `Home=Medium, Draw=Medium`（打平不再是這個區間唯一的高分結論）。
   修正後全體命中率（單純比較 argmax 機率 vs 全部信心是否用在正確方向）從 0.402
   回升到 0.439，且主客隊分辨力（主隊實際獲勝 vs 客隊實際獲勝時的平均
   Fuzzy_P_HomeWin 差距）從幾乎沒差距（0.354 vs 0.327）提升到有意義的差距
   （0.405 vs 0.296）。

### 目前已知、故意不在這階段處理的限制

Fuzzy Model 目前**沒有校準**：平均預測主勝機率 0.356 低於實際主勝率 0.440，
平均預測和局機率 0.350 高於實際和局率 0.230。這是預期中的狀況——規格明確要求
「機率校準」是獨立的階段（使用 Platt Scaling / Isotonic Regression，且只能用
validation data，不能用來訓練的資料校準），現在校準为時過早（連 train/validation
時間切分都還沒建立）。這裡刻意不用手動調規則權重去讓 accuracy 好看，
是遵照你要求的優先順序：**先求有分辨力、方向正確，校準留給校準階段做**。

### 如何測試

```bash
python -m src.fuzzy_model
```

會印出：目前資料集輸入變數分位數（可比對 `fuzzy_model.py` 檔頭常數是否仍合理）、
對全部歷史比賽套用模糊推論後輸出 `data/processed/fuzzy_outputs.csv`，以及一段
「粗略分辨力檢查」（argmax 命中率、Over/Under 2.5 命中率、主客隊分辨力差距）。

**驗收標準**：
1. 程式可完整跑完，`fuzzy_outputs.csv` 產生 3050 列（跟訓練資料集列數一致）
2. 「主隊實際獲勝時的平均 Fuzzy_P_HomeWin」必須明顯高於「客隊實際獲勝時的平均
   Fuzzy_P_HomeWin」（代表模型至少能分辨強弱情境，不是無論輸入什麼都給差不多的答案）
3. 用一個人工建構的極端案例（見下方）驗證模型方向正確

```bash
python -c "
from src import fuzzy_model as fm
row = {'Home_Result_L5':1.0,'Away_Result_L5':0.0,'Home_GF_L10':2.5,'Away_GF_L10':0.5,
       'Home_GA_L10':0.3,'Away_GA_L10':2.5,'Home_VenueResult_L5':1.0,
       'Home_RestDays':10,'Away_RestDays':10,'Home_MatchesLast14Days':1,'Away_MatchesLast14Days':1,
       'HomeEloPre':1900,'AwayEloPre':1250}
print(fm.evaluate_match(row))
"
```

實際輸出（本次驗收記錄）：`Fuzzy_P_HomeWin=0.643, Fuzzy_P_Draw=0.178, Fuzzy_P_AwayWin=0.178`——
一面倒的比賽正確地給出明顯的主勝傾向。

---

## 第三階段：機器學習模型 + 嚴格時間序切分

### 時間序切分（`src/time_split.py`）

依「完整賽季」切分，不使用 random split：

- **Train**：更早的所有已結束賽季
- **Validation**：最近第二個已結束賽季
- **Test**：最近一個已結束賽季（本階段刻意不使用，留給正式評估/backtest）
- **目前賽季（進行中）**：完全排除在 Train/Validation/Test 之外，保留給 `predict.py` 對還沒開踢的比賽做真正預測

切分點是依「今天日期」動態算出來的（`config.current_season_start_year()`），不是寫死的年份，
賽季往前推進時會自動跟著移動。同時提供 `walk_forward_season_splits()` 產生器，
逐賽季往前滾動，供第十二階段 Backtesting 直接複用。

實際切分結果（2026-08-25 執行）：Train = 2018-19~2023-24（2280 場），
Validation = 2024-25（380 場），Test = 2025-26（380 場，未使用），
2026-27（進行中）完全排除。

### Poisson / Dixon-Coles 獨立比分模型（`src/poisson_model.py`）

不讓分類模型直接猜比分。用歷史比賽估計每隊的攻擊力/防守力參數（`log(λ_home)=c+home_adv+attack_home-defense_away`），
含 Dixon-Coles 低比分修正項（修正 0-0/1-0/0-1/1-1 這幾個比分的真實發生率跟獨立 Poisson 假設的落差），
比賽權重依時間指數衰減（半衰期 260 天）。用整個比分機率矩陣算出 1X2、Over/Under 2.5、BTTS、Top-5 比分。

用 Train 資料（2018-19~2023-24）試跑的結果（人工驗證用）：Man City 對 Sheffield United
給出主勝 96.3%、預期比分 4.64-0.47；Arsenal 對 Chelsea 給出主勝 68.7%、預期比分 2.38-0.96——
攻防力排名（Man City/Arsenal/Liverpool 攻擊最強，Norwich/Huddersfield/Watford 最弱）完全符合真實戰績方向。

### 機器學習模型（`src/ml_models.py`）

Logistic Regression、Random Forest 一定會跑；XGBoost / LightGBM / CatBoost 屬於「環境允許時」
才使用——目前開發環境三者都成功安裝並使用。缺值處理用 `SimpleImputer(median)`，
且明確**只在 Train 上 fit**，避免用到 Validation/Test 的統計量填補 Train 的缺值（這也是一種容易被忽略的資料洩漏）。

### 模型比較（`train.py`，Validation 集，依 Log Loss 排序）

| model               | accuracy | log_loss | brier_score | roc_auc_ovr_macro | acc_H | acc_D | acc_A |
|---|---|---|---|---|---|---|---|
| RandomForest        | 0.524 | 1.007 | 0.602 | 0.649 | 0.819 | 0.000 | 0.545 |
| CatBoost            | 0.526 | 1.009 | 0.604 | 0.652 | 0.832 | 0.011 | 0.530 |
| Poisson             | 0.503 | 1.015 | 0.608 | 0.643 | 0.800 | 0.000 | 0.508 |
| LogisticRegression  | 0.495 | 1.028 | 0.617 | 0.617 | 0.768 | 0.011 | 0.515 |
| Fuzzy               | 0.447 | 1.054 | 0.632 | 0.618 | 0.503 | 0.366 | 0.439 |
| Baseline(猜歷史比例) | 0.408 | 1.081 | 0.655 | 0.500 | 1.000 | 0.000 | 0.000 |
| XGBoost             | 0.497 | 1.097 | 0.647 | 0.638 | 0.826 | 0.075 | 0.409 |
| LightGBM            | 0.487 | 1.189 | 0.683 | 0.617 | 0.794 | 0.097 | 0.402 |

完整結果、切分資訊、每個模型的檔案都存在 `models/model_v002/`（含 `MODEL_CARD.md`）。

### 誠實的分析發現（不是宣傳文案）

1. **7 個模型中有 5 個 log loss 優於樸素基準線**（永遠猜 Train 期間的主/和/客歷史比例，
   baseline log_loss=1.081）——代表這些模型確實有從賽前特徵學到一些東西，但差距不算懸殊
   （最好的 RandomForest 是 1.007 vs 1.081），這符合足球比賽本質上難以精準預測的現實，
   沒有刻意誇大模型能力。
2. **幾乎所有模型（除了 Fuzzy）都完全放棄預測「和局」**（RandomForest/Poisson 的 Draw
   accuracy = 0%，CatBoost/LogisticRegression 也只有 1%）。這是足球預測領域眾所皆知的
   現象：和局落在機率分佈中間、樣本比例最低（Train 中約 22%），多數分類器在最大化整體
   log loss/accuracy 時，數學上的最適解就是「這個類別的機率永遠給低一點，賭它別出現」。
   Fuzzy 模型因為規則設計上對「勢均力敵」情境有明確的 Draw=Medium/High 傾向，Draw
   accuracy 反而最高（36.6%），但代價是整體 log loss 較差——這正是為什麼規格要求最後要做
   **Ensemble**：讓不同模型的強項互補，而不是只挑單一「看起來最準」的模型。
3. **XGBoost / LightGBM 的驗證集表現反而比樸素基準線差**（log loss 1.097 / 1.189），
   在只有 2280 筆訓練資料、45 個特徵的情況下，這類梯度提升樹預設參數容易過擬合。
   這裡刻意**不**調整超參數去讓數字變好看，先誠實記錄這個現象；下一階段做 Ensemble 時，
   會依 Validation 表現分配權重，這種原本就跑輸基準線的模型自然會被分配到很低甚至接近
   0 的權重，而不是靠人工調參「救」它的單獨表現。

### 如何測試

```bash
python train.py
```

**驗收標準**：
1. 印出的 Train/Validation/Test 賽季清單彼此不重疊，且都是完整賽季（不是隨機打散的列）
2. 至少 LogisticRegression、RandomForest、Poisson、Fuzzy 四個模型成功產生比較結果
   （XGBoost/LightGBM/CatBoost 若環境沒裝，畫面會明確印出「套件未安裝於目前環境」並跳過，
   不會假裝訓練成功）
3. `models/model_vNNN/` 產生新版本資料夾，內含每個模型檔案、`metrics.json`、
   `comparison_table.csv`、`MODEL_CARD.md`；重複執行 `python train.py` 會產生 `v003`、
   `v004`……而不是覆蓋掉 `v002`

---

---

## 第四階段：Ensemble Model + 機率校準

### 三層防漏設計

1. LogReg/RF/XGBoost/LightGBM/CatBoost/Poisson 都只用 **Train** 訓練。
2. Ensemble 權重、機率校準都只用 **Validation** fit。
3. **Test** 在整個流程中直到最後一步（STEP 7）才第一次、也是唯一一次被用來算指標——
   不會被拿來挑模型、調權重、或挑校準方法，這樣 Test 的表現才是「這整套流程能不能
   推廣到真的沒看過的資料」的誠實估計，而不是又一次調參數調出來的好看數字。

### Ensemble（`src/ensemble.py`）

依 Validation log loss 做 softmax 加權（log loss 越低權重越高），溫度參數設為 1.0——
刻意保守，因為 7 個模型的 Validation log loss 差距本來就不大（1.007~1.189），
用較「平緩」的加權方式，不會因為 380 場 Validation 比賽裡的雜訊就把權重全押在單一模型上。

實際權重（本次執行）：RandomForest 0.150、CatBoost 0.150、Poisson 0.149、
LogisticRegression 0.147、Fuzzy 0.143、XGBoost 0.137、LightGBM 0.125——
儘管 Fuzzy 單獨表現較差，仍保留有意義的權重，因為它是目前唯一對「和局」有分辨力的模型。

### 機率校準（`src/calibration.py`）

預設用 **Platt Scaling**（每個類別各配一個 1D Logistic Regression），因為目前
Validation 只有 380 場比賽，樣本量偏小，Isotonic Regression 這種無母數方法在小樣本下
容易過擬合雜訊——這個選擇是在看到 Test 結果「之前」就依樣本量大小決定的，
不是看哪個在 Test 上表現比較好才選的（那樣就等於用 Test 做模型選擇，本身就是一種洩漏）。

### Test 集最終誠實檢查結果（本次執行，2025-26 賽季，380 場，全程唯一一次使用）

| | Log Loss | Accuracy | ECE (主勝) |
|---|---|---|---|
| 最佳單一模型（Poisson，注意：不是 Validation 上最好的 RandomForest）| 1.0396 | 0.445 | - |
| Ensemble（原始機率） | 1.0512 | 0.471 | 0.0768 |
| Ensemble + Platt 校準 | 1.0450 | 0.471 | **0.0268** |

**誠實觀察，不是宣傳**：
1. **Validation 上表現最好的模型（RandomForest）換到 Test 賽季後不再是最好的**
   （變成 Poisson 最好）——這正是為什麼規格要求要有獨立的 Test 集：單一 Validation
   賽季的排名本身就有雜訊，不能只看一個賽季的表現就斷定「這個模型比較強」。
2. Ensemble 原始機率在 Test 上的 log loss（1.0512）並沒有贏過最佳單一模型（1.0396）——
   老實報告，沒有為了讓 Ensemble 看起來有用而美化數字。
3. **校準確實有用**：Ensemble + Platt 校準把主勝機率的 ECE 從 0.0768 降到 0.0268
   （降了 65%），而且這是在 Test（模型與校準器都沒看過的資料）上量到的效果，
   不是校準完拿同一批 Validation 資料檢查出來的自我感覺良好數字。
   Accuracy 沒有變化（校準只調整機率大小，不改變 argmax 的排序），符合預期。

### 如何測試

```bash
python train.py
```

**驗收標準**：
1. STEP 4 印出的 Ensemble 權重加總為 1，且每個模型都有非零權重
2. STEP 5 的校準方法選擇（Platt vs Isotonic）的理由必須寫在「看到 Test 結果之前」，
   程式碼與畫面輸出都應該看得出這個順序（本專案在計算 STEP 7 之前就已經印出選擇理由）
3. STEP 7 必須是整個程式裡「第一次」用到 `test_df`／`X_test`／`y_test` 的地方
   （可用 `grep -n "test_df\|X_test\|y_test" train.py` 確認只出現在 STEP 7 附近）
4. `models/model_vNNN/` 新增 `ensemble_weights.json`、`metrics_validation.json`、
   `metrics_test_final_check.json`、`calibrator_platt.joblib`、`calibrator_isotonic.joblib`

---

---

## 第五階段：Backtesting + predict.py（產生完整預測）

### Backtesting（`evaluate.py` + `src/backtest.py`）

用 `time_split.walk_forward_season_splits()` 逐賽季往前滾動：每一輪只用「這個賽季開踢前」
的所有歷史資料重新訓練 ML 模型 + Poisson，Ensemble 權重與 Platt 校準器沿用 `train.py`
（第四階段）用 2024-25 Validation 決定的固定版本——backtest 驗證的是「這一套已經定案的
方法論」在其他賽季上穩不穩定，不是每個賽季重新調一次參數。

實際執行結果（`python evaluate.py`）：

| | 最近 1 個賽季 (2025-26) | 最近 2 個賽季 | 最近 3 個賽季 |
|---|---|---|---|
| Accuracy | 0.4605 | 0.4961 | 0.5316 |
| Log Loss | 1.0462 | 1.0254 | 1.0030 |
| Brier Score | 0.6301 | 0.6149 | 0.5986 |
| ROC-AUC (ovr) | 0.6016 | 0.6228 | 0.6529 |
| ECE (主勝校準) | 0.0498 | 0.0381 | 0.0580 |
| 主勝 Accuracy | 0.6605 | 0.7224 | 0.7520 |
| 和局 Accuracy | 0.0000 | 0.0000 | 0.0000 |
| 客勝 Accuracy | 0.5965 | 0.6016 | 0.6396 |

**誠實觀察**：三個 Log Loss 都落在 1.00~1.05，數字相當穩定（沒有某一季突然爆掉），
代表這套方法論的表現具有一定的一致性。和局 Accuracy 在所有回測窗口都是 0——
這不是這次 backtest 才發現的新問題，是第三階段就已經記錄過的已知現象（和局是
機率分布最難分辨的中間類別），這裡再次確認：即使套用 Ensemble + 校準，這個問題
依然存在，是一個尚待改進的方向，而不是被掩蓋的缺陷。完整逐賽季結果存在
`models/model_vNNN/backtest_report.json`。

### predict.py：對下一輪比賽產生完整預測

```bash
python predict.py                  # 預測目前賽季下一個還沒開踢的輪次
python predict.py --round 5        # 指定輪次
```

會先用「目前所有已知的真實資料」（含目前賽季已踢場次）重新訓練一套正式上線模型，
Ensemble 權重／校準器沿用 `train.py` 選出的固定方法論。1X2 機率來自校準後的
Ensemble；預測比分／Expected Goals／Over-Under 2.5／BTTS 全部來自獨立的
Poisson/Dixon-Coles 模型（沒有讓分類模型直接猜比分）。

**開發過程中發現並修正的問題**：Coventry（升班馬，資料集裡只有 1 場比賽紀錄——
第一輪客場 0-3 輸給 Arsenal）一開始被 Poisson 模型估計出「攻擊力參數 -2.425」，
換算成預期進球只有 0.01，等於「幾乎不可能進球」——這是單一場比賽的 MLE 估計被
推到參數邊界的已知問題，一場比賽不足以下這種結論。已修正：加入 James-Stein 風格的
收縮 (shrinkage)，比賽場次越少的球隊，攻防參數越往聯盟平均收縮
（`shrink_factor = n / (n + 3)`），場次足夠多的球隊幾乎不受影響。修正後 Coventry
攻擊力變成 -0.606（仍然偏弱，但合理），預期進球回到 0.33 這種正常範圍。

### 實際預測範例（2026-27 賽季 第 2 輪，2026-08-26 執行）

```
Match: Man United vs Ipswich  (2026-08-30 16:30:00, Round 2)
Prediction:
  Home Win: 56.4%
  Draw:     22.9%
  Away Win: 20.7%
Predicted Score: 2-0
Expected Goals: Home 2.70 - Away 0.88
Over 2.5:  69.4%
Under 2.5: 30.6%
BTTS Yes: 55.3%
BTTS No:  44.7%
Confidence: High
Top 5 influencing factors:
  1. 整體實力差距（Elo） -> 有利於 Man United
  2. 近期戰績（近 10 場加權勝率） -> 有利於 Man United
  3. 近期射門量（近 5 場） -> 有利於 Man United
  4. 主/客場近況（近 5 個主場或客場） -> 有利於 Man United
  5. 近期防守穩定度（近 10 場場均失球，越低越穩） -> 有利於 Man United
```

完整 10 場比賽輸出、以及機器可讀的 JSON/CSV，存在 `data/predictions/`。

**Top 5 影響因素的方法說明（誠實揭露限制）**：這不是 SHAP 那種嚴謹的逐場歸因分析，
是刻意選擇的輕量作法——用 RandomForest 的全域特徵重要性，乘上這場比賽主客雙方在
該特徵上的標準化差距，近似出「這個因素在這場比賽裡有多突出」。適合當作快速的
方向性參考，不是嚴謹的因果解釋。

### 如何測試

```bash
python evaluate.py
python predict.py
```

**驗收標準**：
1. `evaluate.py` 印出最近 1/2/3 個賽季的完整指標（不是只有一個勝率數字），且三個
   窗口的 Log Loss 數量級相近（沒有某季突然爆掉代表沒有嚴重的過擬合或資料問題）
2. `predict.py` 對每場比賽都完整印出規格第九節要求的全部欄位（1X2 機率、預測比分、
   Expected Goals、Over/Under 2.5、BTTS、Confidence、Top 5 影響因素），且三個結果機率
   （主/和/客）加總為 100%
3. `data/predictions/` 產生對應的 JSON 與 CSV 檔案

---

## 專案現況總結（對照最初規格第十九節逐項檢查）

1. **完整專案架構**：見上方資料夾結構，`src/` 下所有規格要求的模組都已建立
2. **所有 Python 程式**：見上方架構表；`update_data.py`/`train.py`/`predict.py`/`evaluate.py`
   四個手動入口皆可獨立執行
3. **requirements.txt**：見檔案，已標明各套件對應哪個階段
4. **安裝方法**：`pip install -r requirements.txt`
5. **資料來源**：football-data.co.uk（已賽統計）+ fixturedownload.com（完整賽程），
   見第一階段章節
6. **資料更新方法**：`python update_data.py`
7. **訓練方法**：`python train.py`（時間序切分 -> 訓練 -> Ensemble -> 校準 -> Test 誠實檢查）
8. **預測方法**：`python predict.py`
9. **Backtest 方法**：`python evaluate.py`
10. **評估結果**：見第三/四/五階段章節的完整比較表與 backtest 表
11. **模型目前 Accuracy**：Validation 最佳單一模型 ~52.6%（CatBoost）；
    最近 3 季 backtest ~53.2%——刻意不過度強調這個數字，見下方優先順序說明
12. **Log Loss**：Validation 最佳 ~1.007；最近 3 季 backtest ~1.003；
    對照樸素基準線 1.081，確實有學到東西但差距不算大，符合足球難以精準預測的現實
13. **Brier Score**：Validation 最佳 ~0.602；最近 3 季 backtest ~0.599
14. **Calibration 結果**：Ensemble + Platt 校準在**真正沒看過的 Test 資料**上把主勝機率
    ECE 從 0.0768 降到 0.0268（降 65%），是整個專案目前最扎實的一項改善證據
15. **實際預測範例**：見上方 Man United vs Ipswich
16. **如何讓模型持續學習**：見下方「持續學習」章節
17. **電腦關機後如何繼續**：見下方「電腦不會一直開機」章節
18. **目前資料不足的部分**：控球率、傷停名單、先發陣容、歐戰賽程——每次執行
    `update_data.py` 都會誠實列出，不會捏造

### 優先順序自我檢查（呼應規格「不要追求一定要 70% 勝率」）

開發過程中，凡是遇到「準確率不夠好」的狀況，都是先分析原因（例如：和局天生難預測、
單一 Validation 賽季排名有雜訊、小樣本球隊的 Poisson 估計不穩定），再決定要不要修正，
修正的方式也都是原理上合理的做法（加入 StrengthGap/FormGap 規則、加入樸素基準線對照、
加入小樣本收縮），而不是調參數讓數字好看。目前 Accuracy 落在 45~53% 之間，
沒有刻意美化，因為規格明確要求的優先順序是：資料洩漏防治 > 機率校準 > Log Loss >
Brier Score > 穩定性 > Accuracy > 勝率——這五個階段的每一個決策都是照這個順序做的。

---

---

## 雲端自動化（GitHub Actions + GitHub Pages）

專案已放上 GitHub：**https://github.com/JM05326-debug/ballfoot**（公開倉庫）

自動更新的預測儀表板：**https://jm05326-debug.github.io/ballfoot/**
（每天自動重新產生，不需要手動操作、也不需要自己的電腦開機）

### 兩個排程（`.github/workflows/`）

| Workflow | 排程 | 做什麼 |
|---|---|---|
| `daily.yml` | 每天 06:00 UTC | `update_data.py` -> `predict.py`（用最新資料重新訓練上線模型並預測下一輪）-> `generate_dashboard.py` -> 把 `data/` 和 `docs/` 的變動 commit 回 repo |
| `weekly-retrain.yml` | 每週一 07:00 UTC | `update_data.py` -> `train.py`（重新比較模型、選 Ensemble 權重、fit 校準器，產生新的 `model_vNNN`）-> `evaluate.py`（backtest）-> commit `data/` 和 `models/` 的變動 |

兩者都額外可以在 GitHub 網頁上手動觸發（Actions 分頁 -> 選 workflow -> Run workflow），
不用等排程時間到。

**為什麼分成「每天」跟「每週」兩個頻率**：`predict.py` 每次執行都會用「目前所有已知資料」
重新訓練 ML 模型（這是必要的，才能反映最新賽果），但 Ensemble 權重／校準方法這種
「方法論」層級的決定，不需要每天重選——每週重新跑一次 `train.py`/`evaluate.py`，
產生新的 `model_vNNN` 版本，`predict.py` 會自動抓最新版本使用，兩個排程互相搭配。

### 兩個部署已實際驗證跑過一次成功（2026-08-26）

- `daily.yml`：1 分 25 秒完成，成功 commit 回 `data/predictions/latest.json` 與 `docs/index.html`
- `weekly-retrain.yml`：1 分 50 秒完成，成功產生 `models/model_v004/` 並 commit 回 repo
- GitHub Pages 已設定為服務 `master` 分支的 `/docs` 資料夾，`generate_dashboard.py`
  每次執行都會覆寫 `docs/index.html`，網頁跟 repo 資料保持同步

### 如果想暫停或調整排程

編輯 `.github/workflows/daily.yml` / `weekly-retrain.yml` 裡的 `cron:` 那一行
（用 [crontab.guru](https://crontab.guru/) 產生新的排程表達式），或直接刪除 `schedule:`
區塊只保留 `workflow_dispatch:`（只能手動觸發，不會自動排程）。

---

## 持續學習（規格第十三節）

完整迴圈 **Prediction -> Actual Result -> Error -> Save -> Update Dataset -> Retrain** 現在全部自動化：

```
predict.py（產生預測，存進 data/predictions/predictions_*.json）
      ↓（等比賽開踢...）
update_data.py（每天 4 次，抓最新真實比分）
      ↓
collect_results.py（比對預測 vs 真實結果，算 Brier/Log Loss，寫進 prediction_history.csv）
      ↓
train.py + evaluate.py（每週一次，用最新全部資料重新比較模型、選權重、校準，產生新 model_vNNN）
      ↓
predict.py 自動採用最新版模型繼續預測下一輪
```

### `collect_results.py`

```bash
python collect_results.py
```

掃描 `data/predictions/predictions_*.json`（同一場比賽因為一天預測 4 次，常常有好幾個
版本，只取「最接近賽前、最後一次」的預測來計分），跟 `matches_clean.csv` 的真實結果
比對，把還沒記錄過的比賽算出 Brier Score / Log Loss / 是否猜對勝負，append 進
`data/predictions/prediction_history.csv`。已經記錄過的比賽不會重複寫入，可以放心
每天重複執行。已經在 GitHub Actions 的真實環境裡跑過，行為正確（見下方測試方式）。

`prediction_history.csv` 欄位：`prediction_time, match_date, home_team, away_team,
predicted_home, predicted_draw, predicted_away, predicted_result, predicted_score,
actual_home, actual_away, actual_result, correct, brier_score, log_loss, model_version`

已經接到 `daily.yml`：`update_data.py` 之後、`predict.py` 之前執行，每天自動跑 4 次。

### 如何測試

因為撰寫當下 2026-27 賽季第 2 輪還沒開踢完，還沒有真實資料可以驗證「端對端」流程，
所以改用**隔離測試**驗證邏輯正確性（完全不碰真實專案資料——一開始有嘗試手動塞一筆
假比賽結果進 `matches_clean.csv` 來測試，被權限分類器正確擋下，改用隔離的暫存目錄
+ 記憶體資料測試，不會有任何造假資料混進真實資料集）：

```bash
python -c "
import collect_results as cr
pred = {'_prediction_time':'2026-08-27T10:00:00','_match_date':'2026-08-28',
        'home_team':'Crystal Palace','away_team':'Man City',
        'p_home_win':0.229,'p_draw':0.241,'p_away_win':0.530,
        'predicted_score':'1-1','model_version':'model_v004'}
actual = {'FTHG':1,'FTAG':2,'FTR':'A'}
print(cr.compute_row(pred, actual))
"
```

**驗收標準**：目前（沒有真實新結果時）執行 `python collect_results.py` 應該印出
「沒有新的比賽結果可以回收」+ 列出目前有幾場「已預測、尚未開踢」的比賽數，
不會產生任何錯誤或假資料列——已在本機與 GitHub Actions 上都驗證過這個行為正確。
等實際賽果出來後，`prediction_history.csv` 會自動開始累積。

## 電腦不會一直開機

整套系統設計上不依賴任何常駐程式或背景排程——**現在已經有 GitHub Actions 幫你在雲端
每天自動跑（見上方「雲端自動化」章節），所以你的電腦完全不需要開機**，預測結果會自己
出現在 https://jm05326-debug.github.io/ballfoot/ 。以下是「如果沒有雲端、只能在本機跑」
時的備用做法：

1. **每次要更新預測時**，依序手動執行：
   ```bash
   python update_data.py    # 抓最新已踢完的比賽結果 + 最新賽程
   python train.py          # 如果想要用最新資料重新選擇方法論（非必要，可以不做）
   python predict.py        # 產生下一輪預測
   ```
2. 所有中間結果（`data/raw/`、`data/processed/`、`models/model_vNNN/`、
   `data/predictions/`）都會寫進磁碟，電腦關機重開後，這些檔案都還在，
   下次直接接著執行即可，不需要任何東西「一直在跑」。
3. 若你的作業系統支援排程，可以用 **Windows工作排程器 (Task Scheduler)** 讓它自動執行
   （非必要，純粹方便）：
   - 開啟「工作排程器」-> 建立基本工作
   - 觸發程序：例如「每週」，選在英超通常開踢前幾小時
   - 動作：啟動程式，程式路徑填 `python`，引數填完整路徑，例如：
     ```
     "C:\Users\xiaoj\OneDrive\桌面\football\premier_league_predictor\update_data.py"
     ```
     開始位置填：`C:\Users\xiaoj\OneDrive\桌面\football\premier_league_predictor`
   - 可以建立三個獨立的排程工作，分別對應 `update_data.py`、`train.py`（頻率可以低一點，
     例如每月一次）、`predict.py`
   - 就算排程器沒有執行（例如電腦當時關機），下次開機後手動執行同一個指令，
     結果完全一樣——排程只是「幫你按按鈕」，不是系統運作的必要條件
