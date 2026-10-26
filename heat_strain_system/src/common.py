from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, recall_score, roc_auc_score,
                              accuracy_score, precision_score)

LABELS = ["Low", "Moderate", "High"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}


class Imputer:
    """Median imputation fit on a training fold only (no leakage across
    subject-independent / condition-independent splits)."""
    def __init__(self, cols):
        self.cols = cols
        self.medians = {}

    def fit(self, df):
        for c in self.cols:
            self.medians[c] = df[c].median()
            if pd.isna(self.medians[c]):
                self.medians[c] = 0.0
        return self

    def transform(self, df):
        df = df.copy()
        for c in self.cols:
            df[c] = df[c].fillna(self.medians[c])
        return df


def class_weights_from(y, n_classes=3):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return w


def compute_metrics(y_true, y_pred, y_proba, n_classes=3) -> dict:
    out = {}
    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["macro_f1"] = f1_score(y_true, y_pred, average="macro", labels=list(range(n_classes)), zero_division=0)
    out["macro_precision"] = precision_score(y_true, y_pred, average="macro", labels=list(range(n_classes)), zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0)
    for i, lab in enumerate(LABELS):
        out[f"recall_{lab}"] = recalls[i] if i < len(recalls) else np.nan
    try:
        if len(np.unique(y_true)) > 1 and y_proba.shape[1] == n_classes:
            out["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr", labels=list(range(n_classes)))
        else:
            out["roc_auc_ovr"] = np.nan
    except Exception:
        out["roc_auc_ovr"] = np.nan
    out["n"] = len(y_true)
    return out


def expected_calibration_error(y_true, y_proba, n_bins=10) -> float:
    conf = y_proba.max(axis=1)
    pred = y_proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc_bin = correct[m].mean()
        conf_bin = conf[m].mean()
        ece += (m.sum() / len(conf)) * abs(acc_bin - conf_bin)
    return float(ece)


def reliability_curve(y_true, y_proba, n_bins=10):
    conf = y_proba.max(axis=1)
    pred = y_proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        centers.append((lo + hi) / 2)
        counts.append(int(m.sum()))
        accs.append(correct[m].mean() if m.sum() else np.nan)
        confs.append(conf[m].mean() if m.sum() else np.nan)
    return np.array(centers), np.array(accs), np.array(confs), np.array(counts)


def fit_temperature(logits: np.ndarray, y_true: np.ndarray, n_classes=3) -> float:
    y1h = np.eye(n_classes)[y_true]

    def nll(T):
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.mean(np.sum(y1h * np.log(p), axis=1))

    Ts = np.linspace(0.2, 5.0, 97)
    losses = [nll(T) for T in Ts]
    return float(Ts[int(np.argmin(losses))])


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    return p


def proba_to_logits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p)
