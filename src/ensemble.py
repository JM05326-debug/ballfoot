"""
Ensemble：依 Validation 表現決定各模型權重，不假設哪個模型最好。

方法：對每個候選模型的 Validation log loss 取負號後做 softmax（log loss 越低 -> 權重越高），
temperature 控制「表現差距要放大多少」。目前 7 個候選模型的 Validation log loss 落在
1.01~1.19 之間，差距本來就不算懸殊（都只比樸素基準線好一點點——見 train.py 的分析），
用 temperature=1.0 的 softmax 不會因為 380 場 Validation 比賽裡的雜訊，就把權重
過度集中在單一模型上，這是刻意的保守設計。

Ensemble 的機率 = 每個模型機率的加權平均（加權前後都會重新正規化，確保三個類別機率和為 1）。
"""

from __future__ import annotations

import numpy as np

DEFAULT_TEMPERATURE = 1.0


def compute_log_loss_weights(
    reports: dict[str, dict],
    model_names: list[str],
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict[str, float]:
    losses = np.array([reports[name]["log_loss"] for name in model_names], dtype=float)
    scaled = -losses / temperature
    scaled = scaled - scaled.max()  # 數值穩定，避免 exp 溢位
    w = np.exp(scaled)
    w = w / w.sum()
    return {name: float(weight) for name, weight in zip(model_names, w)}


def weighted_ensemble_proba(proba_dict: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    names = list(proba_dict.keys())
    stacked = np.stack([proba_dict[name] * weights[name] for name in names], axis=0)
    combined = stacked.sum(axis=0)
    combined = np.clip(combined, 1e-9, None)
    combined = combined / combined.sum(axis=1, keepdims=True)
    return combined
