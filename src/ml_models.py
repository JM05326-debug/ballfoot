"""
機器學習模型層：Logistic Regression / Random Forest / XGBoost / LightGBM / CatBoost。

所有模型都只認得 feature_engineering.py 產生的 Home_*/Away_*/EloDiffPre 等賽前特徵，
不會使用 MatchID/Season/Date/HomeTeam/AwayTeam/Referee 這些非數值或有洩漏疑慮的欄位，
更不會使用 FTHG/FTAG/FTR（那是要預測的標籤）。

缺值處理：一律用 SimpleImputer(strategy="median")，而且明確只在 Train 資料上
fit，Validation/Test 只做 transform——避免用到「未來」的統計量去填補「過去」的缺值，
這也是資料洩漏的一種常見疏忽。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import evaluation

NON_FEATURE_COLUMNS = {
    "MatchID", "Season", "SeasonStartYear", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "Referee",
}


def get_feature_columns(dataset: pd.DataFrame) -> list[str]:
    return [c for c in dataset.columns if c not in NON_FEATURE_COLUMNS and not c.startswith("Fuzzy_") and not c.startswith("Poisson_")]


def prepare_xy(dataset: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = dataset[feature_cols].to_numpy(dtype=float)
    y = evaluation.labels_to_index(dataset["FTR"].to_numpy())
    return X, y


def _try_import_xgboost():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError:
        return None


def _try_import_lightgbm():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier
    except ImportError:
        return None


def _try_import_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except ImportError:
        return None


def build_model_registry(random_state: int = 42) -> dict[str, Pipeline]:
    """回傳 {模型名稱: sklearn Pipeline}。缺少的套件（xgboost/lightgbm/catboost）會被跳過，
    並在 train.py 執行時明確印出訊息，而不是靜默假裝有跑。"""
    registry: dict[str, Pipeline] = {}

    registry["LogisticRegression"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, C=1.0)),
    ])

    registry["RandomForest"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=5,
            random_state=random_state, n_jobs=-1,
        )),
    ])

    XGBClassifier = _try_import_xgboost()
    if XGBClassifier is not None:
        registry["XGBoost"] = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                random_state=random_state, n_jobs=-1,
            )),
        ])

    LGBMClassifier = _try_import_lightgbm()
    if LGBMClassifier is not None:
        registry["LightGBM"] = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", LGBMClassifier(
                n_estimators=300, max_depth=-1, learning_rate=0.05,
                num_leaves=15, objective="multiclass", num_class=3,
                random_state=random_state, verbosity=-1,
            )),
        ])

    CatBoostClassifier = _try_import_catboost()
    if CatBoostClassifier is not None:
        registry["CatBoost"] = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", CatBoostClassifier(
                iterations=300, depth=4, learning_rate=0.05,
                loss_function="MultiClass", random_seed=random_state, verbose=False,
            )),
        ])

    return registry


def predict_proba_aligned(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """回傳機率預測，欄位順序固定為 [H, D, A]（見 evaluation.CLASS_LABELS）。

    sklearn 的 classes_ 是依訓練資料出現過的類別排序，若訓練集剛好缺某個類別
    （理論上不太可能，因為英超每季都有主/和/客結果，但仍在此明確防呆），
    需要把欄位對齊回 [H, D, A] 固定順序，避免欄位錯位造成的隱性 bug。
    """
    proba = model.predict_proba(X)
    full_proba = np.zeros((proba.shape[0], 3))
    classes = model.named_steps["clf"].classes_ if hasattr(model, "named_steps") else model.classes_
    for i, cls in enumerate(classes):
        full_proba[:, int(cls)] = proba[:, i]
    return full_proba


def train_and_predict(
    model: Pipeline,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,
) -> np.ndarray:
    """訓練後回傳驗證集上的機率預測，欄位順序固定為 [H, D, A]。"""
    model.fit(X_train, y_train)
    return predict_proba_aligned(model, X_val)
