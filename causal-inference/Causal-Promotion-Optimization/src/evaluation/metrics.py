from __future__ import annotations

import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def wape(y_true, y_pred):
    y_true = np.asarray(y_true)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - np.asarray(y_pred))) / denom)


def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def regression_metrics(y_true, y_pred):
    return {
        'mae': mae(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'wape': wape(y_true, y_pred),
        'smape': smape(y_true, y_pred),
    }
