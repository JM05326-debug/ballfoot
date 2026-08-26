"""
機率校準 (Probability Calibration)。

規格明確要求：即使 Accuracy 很高也不代表模型好；必須檢查「預測主勝 70% 的比賽，
長期實際主勝率是否接近 70%」。校準一律用 Platt Scaling 或 Isotonic Regression，
而且**只能用 Validation 資料 fit**，絕對不能碰 Train（模型本身已經在學 Train 了，
用同一批資料校準會低估真實誤差）或 Test（Test 要留著做最後一次、唯一一次的
「真的沒看過」總檢查，見 train.py 最後一步）。

方法選擇：Platt Scaling（每個類別各配一個 1D Logistic Regression，屬於 parametric，
只有 2 個參數）vs Isotonic Regression（non-parametric，彈性更高但需要較多樣本才穩定）。
本專案目前 Validation 只有 380 場比賽，屬於偏小的樣本量，因此預設用 Platt Scaling；
Isotonic 一併提供、一併報告數字，但只作為診斷參考，不當作預設方法
——這個選擇是在「看到 Test 表現之前」就依樣本量大小決定的，不是拿 Test 表現去挑校準方法。

多類別處理：三個類別（主/和/客）分別各自做 one-vs-rest 校準，校準後再重新正規化，
確保三個機率加總仍是 1。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

METHOD_PLATT = "platt"
METHOD_ISOTONIC = "isotonic"


@dataclass
class ProbabilityCalibrator:
    method: str
    calibrators: dict = field(default_factory=dict)  # class_idx -> fitted estimator


def fit_calibrator(proba: np.ndarray, y_true_idx: np.ndarray, method: str = METHOD_PLATT) -> ProbabilityCalibrator:
    n_classes = proba.shape[1]
    calibrators = {}
    for k in range(n_classes):
        y_bin = (y_true_idx == k).astype(int)
        p_k = proba[:, k]
        if method == METHOD_PLATT:
            est = LogisticRegression()
            est.fit(p_k.reshape(-1, 1), y_bin)
        elif method == METHOD_ISOTONIC:
            est = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            est.fit(p_k, y_bin)
        else:
            raise ValueError(f"未知校準方法: {method}")
        calibrators[k] = est
    return ProbabilityCalibrator(method=method, calibrators=calibrators)


def apply_calibrator(calibrator: ProbabilityCalibrator, proba: np.ndarray) -> np.ndarray:
    out = np.zeros_like(proba)
    for k, est in calibrator.calibrators.items():
        p_k = proba[:, k]
        if calibrator.method == METHOD_PLATT:
            out[:, k] = est.predict_proba(p_k.reshape(-1, 1))[:, 1]
        else:
            out[:, k] = est.predict(p_k)
    out = np.clip(out, 1e-6, None)
    out = out / out.sum(axis=1, keepdims=True)
    return out
