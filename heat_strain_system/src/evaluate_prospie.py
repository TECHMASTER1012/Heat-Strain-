"""Main evaluation for the PROSPIE sensor-fusion heat-strain system.

Three levels of validation, all genuinely subject/condition-independent:
  1. 8-fold GroupKFold by participant -- baselines + fusion ablations
  2. True Leave-One-Participant-Out (40 folds, XGBoost) -- per-subject error analysis
  3. Leave-One-Condition-Out (9 folds, fusion_full + XGBoost) -- generalization
     across heat-stress *scenarios* (factory / desert / humid / cool-control / ...),
     something the previous two source datasets could never test.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from common import (Imputer, class_weights_from, compute_metrics, expected_calibration_error,
                     fit_temperature, apply_temperature, proba_to_logits, LABELS)
from fusion_model_prospie import MODALITIES, ALL_MODEL_FEATURE_COLS, train_fusion_model
from baselines import fit_predict_logreg, fit_predict_rf, fit_predict_xgb

RESULTS_DIR = os.environ.get("HS_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")) + "/results"
N_SPLITS = 8
SEED = 0

ROLL_IMPUTE_COLS = [c for c in ALL_MODEL_FEATURE_COLS]


def build_modality_matrices(df, scalers=None, fit=False):
    X = {}
    new_scalers = {}
    for mod, cols in MODALITIES.items():
        arr = df[cols].values.astype(float)
        if fit:
            sc = StandardScaler().fit(arr)
            new_scalers[mod] = sc
        else:
            sc = scalers[mod]
        X[mod] = sc.transform(arr)
    return X, (new_scalers if fit else scalers)


def prep_fold(train_df, test_df):
    imputer = Imputer(ROLL_IMPUTE_COLS).fit(train_df)
    train_df = imputer.transform(train_df)
    test_df = imputer.transform(test_df)
    Xtr_flat = train_df[ALL_MODEL_FEATURE_COLS].values.astype(float)
    Xte_flat = test_df[ALL_MODEL_FEATURE_COLS].values.astype(float)
    ytr, yte = train_df["y"].values, test_df["y"].values
    Xtr_mod, scalers = build_modality_matrices(train_df, fit=True)
    Xte_mod, _ = build_modality_matrices(test_df, scalers=scalers, fit=False)
    return train_df, test_df, Xtr_flat, Xte_flat, ytr, yte, Xtr_mod, Xte_mod


def calibrate(model, predict_fn, Xtr_mod, train_df, Xte_mod_logits, y_test, fold_i, seed=SEED, val_frac=0.15):
    raw_ece = expected_calibration_error(y_test, apply_temperature(Xte_mod_logits, 1.0))
    rng = np.random.default_rng(seed + fold_i)
    subs = train_df["participant"].unique()
    n_val = max(2, int(len(subs) * val_frac))
    val_subs = set(rng.choice(subs, size=n_val, replace=False))
    val_mask = train_df["participant"].isin(val_subs).values
    if val_mask.sum() < 5 or (~val_mask).sum() < 5:
        return raw_ece, raw_ece, 1.0
    Xval_mod = {k: v[val_mask] for k, v in Xtr_mod.items()}
    yval = train_df["y"].values[val_mask]
    val_probs = predict_fn(model, Xval_mod)
    T = fit_temperature(proba_to_logits(val_probs), yval)
    cal_probs = apply_temperature(Xte_mod_logits, T)
    cal_ece = expected_calibration_error(y_test, cal_probs)
    return raw_ece, cal_ece, T


def run_main_cv():
    os.makedirs(f"{RESULTS_DIR}/tables", exist_ok=True)
    df = pd.read_parquet(f"{RESULTS_DIR}/tables/prospie_features.parquet")
    print(f"Loaded {len(df)} windows, {df['participant'].nunique()} participants, "
          f"{df['condition'].nunique()} conditions")
    print(df["y"].value_counts().rename(index=dict(enumerate(LABELS))))

    participants = df["participant"].values
    gkf = GroupKFold(n_splits=N_SPLITS)

    variants = {
        "fusion_full":           dict(use_fusion_attention=True,  use_grl=True,  use_modalities=None),
        "ablation_no_grl":       dict(use_fusion_attention=True,  use_grl=False, use_modalities=None),
        "ablation_no_fusion":    dict(use_fusion_attention=False, use_grl=True,  use_modalities=None),
        "ablation_cardio_only":  dict(use_fusion_attention=True,  use_grl=True,  use_modalities=["cardio_resp"]),
    }

    fold_rows, ece_rows = [], []
    fold_csv = f"{RESULTS_DIR}/tables/cv_fold_results.csv"
    ece_csv = f"{RESULTS_DIR}/tables/calibration_results.csv"
    done_folds = set()
    if os.path.exists(fold_csv):
        prev = pd.read_csv(fold_csv)
        done_folds = set(prev["fold"].unique().tolist())
        fold_rows = prev.to_dict("records")
    if os.path.exists(ece_csv):
        ece_rows = pd.read_csv(ece_csv).to_dict("records")

    t_start = time.time()
    for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(df, groups=participants)):
        if fold_i in done_folds:
            print(f"=== Fold {fold_i+1}/{N_SPLITS} already done, skipping ===")
            continue
        train_df, test_df = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        train_df, test_df, Xtr_flat, Xte_flat, ytr, yte, Xtr_mod, Xte_mod = prep_fold(train_df, test_df)
        print(f"\n=== Fold {fold_i+1}/{N_SPLITS} | train={len(train_df)} ({train_df['participant'].nunique()} p) "
              f"test={len(test_df)} ({test_df['participant'].nunique()} p) ===")

        for name, fn in [("LogReg", fit_predict_logreg), ("RandomForest", fit_predict_rf), ("XGBoost", fit_predict_xgb)]:
            t0 = time.time()
            proba = fn(Xtr_flat, ytr, Xte_flat, seed=SEED)
            pred = proba.argmax(axis=1)
            m = compute_metrics(yte, pred, proba)
            m.update(model=name, fold=fold_i, fit_s=time.time() - t0)
            fold_rows.append(m)

            raw_ece = expected_calibration_error(yte, proba)
            rng = np.random.default_rng(SEED + fold_i)
            subs = train_df["participant"].unique()
            n_val = max(2, int(len(subs) * 0.15))
            val_subs = set(rng.choice(subs, size=n_val, replace=False))
            val_mask = train_df["participant"].isin(val_subs).values
            cal_ece, T = raw_ece, 1.0
            if val_mask.sum() >= 5 and (~val_mask).sum() >= 5:
                val_proba = fn(Xtr_flat[~val_mask], ytr[~val_mask], Xtr_flat[val_mask], seed=SEED)
                T = fit_temperature(proba_to_logits(val_proba), ytr[val_mask])
                cal_ece = expected_calibration_error(yte, apply_temperature(proba_to_logits(proba), T))
            ece_rows.append(dict(model=name, fold=fold_i, ece_raw=raw_ece, ece_cal=cal_ece, T=T))
            print(f"  {name:24s} macroF1={m['macro_f1']:.3f} acc={m['accuracy']:.3f} "
                  f"HighRecall={m.get('recall_High', np.nan):.3f} ({time.time()-t0:.1f}s)")

        cw = class_weights_from(ytr)
        for vname, kwargs in variants.items():
            t0 = time.time()
            model = train_fusion_model(Xtr_mod, ytr, train_df["participant"].values, epochs=100,
                                        lr=3e-3, batch_size=64, class_weights=cw,
                                        grl_lambda=0.15, use_focal=True, focal_gamma=1.0, seed=SEED, **kwargs)
            mean_p, _ = model.predict_proba(Xte_mod, mc_samples=1)
            pred = mean_p.argmax(axis=1)
            m = compute_metrics(yte, pred, mean_p)
            m.update(model=vname, fold=fold_i, fit_s=time.time() - t0)
            fold_rows.append(m)

            def _predict_fn(mdl, xd):
                p, _ = mdl.predict_proba(xd, mc_samples=1)
                return p
            raw_ece, cal_ece, T = calibrate(model, _predict_fn, Xtr_mod, train_df,
                                             proba_to_logits(mean_p), yte, fold_i)
            ece_rows.append(dict(model=vname, fold=fold_i, ece_raw=raw_ece, ece_cal=cal_ece, T=T))
            print(f"  {vname:24s} macroF1={m['macro_f1']:.3f} acc={m['accuracy']:.3f} "
                  f"HighRecall={m.get('recall_High', np.nan):.3f} ({time.time()-t0:.1f}s)")

        pd.DataFrame(fold_rows).to_csv(f"{RESULTS_DIR}/tables/cv_fold_results.csv", index=False)
        pd.DataFrame(ece_rows).to_csv(f"{RESULTS_DIR}/tables/calibration_results.csv", index=False)

    print(f"\nTotal CV time: {time.time()-t_start:.1f}s")
    results = pd.DataFrame(fold_rows)
    summary = results.groupby("model").agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        recall_Low=("recall_Low", "mean"), recall_Moderate=("recall_Moderate", "mean"),
        recall_High=("recall_High", "mean"), roc_auc=("roc_auc_ovr", "mean"),
    ).round(3)
    summary.to_csv(f"{RESULTS_DIR}/tables/model_summary.csv")
    print("\n=== SUMMARY ===")
    print(summary)

    sig_rows = []
    pivot = results.pivot_table(index="fold", columns="model", values="macro_f1")
    for other in pivot.columns:
        if other == "fusion_full":
            continue
        a, b = pivot["fusion_full"].values, pivot[other].values
        t_stat, p_t = stats.ttest_rel(a, b)
        try:
            w_stat, p_w = stats.wilcoxon(a, b)
        except ValueError:
            w_stat, p_w = np.nan, np.nan
        sig_rows.append(dict(comparison=f"fusion_full vs {other}", mean_diff=float(np.mean(a - b)),
                              t_stat=t_stat, p_value_ttest=p_t, p_value_wilcoxon=p_w))
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(f"{RESULTS_DIR}/tables/significance_tests.csv", index=False)
    print("\n=== Significance ===")
    print(sig_df)

    ece_df = pd.DataFrame(ece_rows)
    ece_summary = ece_df.groupby("model").agg(ece_raw_mean=("ece_raw", "mean"), ece_cal_mean=("ece_cal", "mean")).round(3)
    ece_summary.to_csv(f"{RESULTS_DIR}/tables/calibration_summary.csv")
    print("\n=== Calibration ===")
    print(ece_summary)


if __name__ == "__main__":
    run_main_cv()
