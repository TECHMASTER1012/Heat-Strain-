from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def fit_predict_logreg(Xtr, ytr, Xte, n_classes=3, seed=0):
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(scaler.transform(Xtr), ytr)
    proba = clf.predict_proba(scaler.transform(Xte))
    # ensure all n_classes columns present even if a class was absent in train
    full = np.zeros((len(Xte), n_classes))
    full[:, clf.classes_] = proba
    return full


def fit_predict_rf(Xtr, ytr, Xte, n_classes=3, seed=0):
    clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                  class_weight="balanced", random_state=seed, n_jobs=-1)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    full = np.zeros((len(Xte), n_classes))
    full[:, clf.classes_] = proba
    return full


def fit_predict_xgb(Xtr, ytr, Xte, n_classes=3, seed=0):
    counts = np.bincount(ytr, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    sample_weight = w[ytr]
    clf = xgb.XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softprob",
        num_class=n_classes, eval_metric="mlogloss", random_state=seed,
        n_jobs=1, verbosity=0,
    )
    clf.fit(Xtr, ytr, sample_weight=sample_weight)
    proba = clf.predict_proba(Xte)
    return proba
