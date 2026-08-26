"""
共用評估指標。

規格明確要求優先順序：資料洩漏防治 > 機率校準 > Log Loss > Brier Score > 穩定性 >
Accuracy > 勝率。本模組提供的函式刻意把「機率品質」類指標（log loss、brier、
校準誤差）放在跟 accuracy 同等重要的位置，而不是只回傳一個 accuracy 數字。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

CLASS_LABELS = ["H", "D", "A"]
CLASS_INDEX = {label: i for i, label in enumerate(CLASS_LABELS)}


def labels_to_index(labels: pd.Series | np.ndarray) -> np.ndarray:
    return np.array([CLASS_INDEX[label] for label in labels])


def accuracy(y_true_idx: np.ndarray, proba: np.ndarray) -> float:
    y_pred_idx = np.argmax(proba, axis=1)
    return float(np.mean(y_pred_idx == y_true_idx))


def per_class_accuracy(y_true_idx: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    y_pred_idx = np.argmax(proba, axis=1)
    out = {}
    for label, idx in CLASS_INDEX.items():
        mask = y_true_idx == idx
        if mask.sum() == 0:
            out[label] = float("nan")
        else:
            out[label] = float(np.mean(y_pred_idx[mask] == idx))
    return out


def multiclass_log_loss(y_true_idx: np.ndarray, proba: np.ndarray) -> float:
    proba = np.clip(proba, 1e-15, 1 - 1e-15)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return float(log_loss(y_true_idx, proba, labels=[0, 1, 2]))


def brier_score_multiclass(y_true_idx: np.ndarray, proba: np.ndarray) -> float:
    """多類別 Brier Score：對每個類別做 one-hot，取 (p-y)^2 的平均。"""
    n_classes = proba.shape[1]
    onehot = np.eye(n_classes)[y_true_idx]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def roc_auc_ovr_macro(y_true_idx: np.ndarray, proba: np.ndarray) -> float | None:
    n_classes = proba.shape[1]
    onehot = np.eye(n_classes)[y_true_idx]
    try:
        return float(roc_auc_score(onehot, proba, average="macro", multi_class="ovr"))
    except ValueError:
        # 若驗證集裡某個類別樣本數為 0（例如驗證賽季剛好沒有和局），ROC-AUC 無法定義
        return None


def expected_calibration_error(y_true_binary: np.ndarray, proba_binary: np.ndarray, n_bins: int = 10) -> float:
    """單一類別（例如「主勝」）的 Expected Calibration Error。

    只是「量測」目前的校準誤差，不會對機率做任何修正——修正屬於第十一階段
    （Platt Scaling / Isotonic Regression，且只能用 validation data fit）的工作。
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(proba_binary, bins[1:-1])
    ece = 0.0
    n = len(proba_binary)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        avg_pred = proba_binary[mask].mean()
        avg_actual = y_true_binary[mask].mean()
        ece += (mask.sum() / n) * abs(avg_pred - avg_actual)
    return float(ece)


def calibration_bins(y_true_binary: np.ndarray, proba_binary: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """回傳每個機率區間的 (預測平均機率, 實際發生比例, 樣本數)，供畫校準曲線用。"""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(proba_binary, bins[1:-1])
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
                "predicted_mean": float(proba_binary[mask].mean()),
                "actual_mean": float(y_true_binary[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def full_report(y_true_idx: np.ndarray, proba: np.ndarray) -> dict:
    """組合出一個模型在某個資料切分上的完整評估報告（stage 7 比較表要用的所有指標）。"""
    home_binary = (y_true_idx == CLASS_INDEX["H"]).astype(int)
    return {
        "accuracy": accuracy(y_true_idx, proba),
        "per_class_accuracy": per_class_accuracy(y_true_idx, proba),
        "log_loss": multiclass_log_loss(y_true_idx, proba),
        "brier_score": brier_score_multiclass(y_true_idx, proba),
        "roc_auc_ovr_macro": roc_auc_ovr_macro(y_true_idx, proba),
        "ece_home_win": expected_calibration_error(home_binary, proba[:, CLASS_INDEX["H"]]),
        "n_samples": int(len(y_true_idx)),
    }
