"""Leave-One-Condition-Out (9 heat-stress scenarios) generalization test,
and true Leave-One-Participant-Out (40 folds) error analysis -- both novel
validations the earlier proxy-label system could never run because it had
no real distinct "conditions"."""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, os.path.dirname(__file__))
from common import Imputer, class_weights_from, compute_metrics, LABELS
from evaluate_prospie import build_modality_matrices, ROLL_IMPUTE_COLS, RESULTS_DIR
from fusion_model_prospie import ALL_MODEL_FEATURE_COLS, train_fusion_model
from baselines import fit_predict_xgb


def loco_condition_generalization():
    df = pd.read_parquet(f"{RESULTS_DIR}/tables/prospie_features.parquet")
    logo = LeaveOneGroupOut()
    rows = []
    for tr_idx, te_idx in logo.split(df, groups=df["condition"].values):
        train_df, test_df = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        cond = test_df["condition"].iloc[0]
        scenario = test_df["scenario"].iloc[0]
        imputer = Imputer(ROLL_IMPUTE_COLS).fit(train_df)
        train_df_i, test_df_i = imputer.transform(train_df), imputer.transform(test_df)
        Xtr_flat = train_df_i[ALL_MODEL_FEATURE_COLS].values.astype(float)
        Xte_flat = test_df_i[ALL_MODEL_FEATURE_COLS].values.astype(float)
        ytr, yte = train_df_i["y"].values, test_df_i["y"].values

        t0 = time.time()
        proba_xgb = fit_predict_xgb(Xtr_flat, ytr, Xte_flat, seed=0)
        pred_xgb = proba_xgb.argmax(axis=1)
        m = compute_metrics(yte, pred_xgb, proba_xgb)
        m.update(model="XGBoost", condition=cond, scenario=scenario, n_train_windows=len(train_df),
                  n_test_windows=len(test_df))
        rows.append(m)

        Xtr_mod, scalers = build_modality_matrices(train_df_i, fit=True)
        Xte_mod, _ = build_modality_matrices(test_df_i, scalers=scalers, fit=False)
        cw = class_weights_from(ytr)
        model = train_fusion_model(Xtr_mod, ytr, train_df_i["participant"].values, epochs=100, lr=3e-3,
                                    batch_size=64, class_weights=cw, grl_lambda=0.15, use_focal=True,
                                    focal_gamma=1.0, seed=0, verbose=False)
        mean_p, _ = model.predict_proba(Xte_mod, mc_samples=1)
        pred_fu = mean_p.argmax(axis=1)
        m2 = compute_metrics(yte, pred_fu, mean_p)
        m2.update(model="fusion_full", condition=cond, scenario=scenario, n_train_windows=len(train_df),
                   n_test_windows=len(test_df))
        rows.append(m2)
        print(f"cond {cond:2d} ({scenario[:40]:40s}) XGB macroF1={m['macro_f1']:.3f} acc={m['accuracy']:.3f} | "
              f"Fusion macroF1={m2['macro_f1']:.3f} acc={m2['accuracy']:.3f}  ({time.time()-t0:.1f}s)")

    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS_DIR}/tables/loco_condition_results.csv", index=False)
    print("\n=== LOCO summary ===")
    print(out.groupby("model")[["accuracy", "macro_f1"]].agg(["mean", "std"]))
    return out


def lopo_error_analysis():
    df = pd.read_parquet(f"{RESULTS_DIR}/tables/prospie_features.parquet")
    logo = LeaveOneGroupOut()
    rows = []
    t0 = time.time()
    for tr_idx, te_idx in logo.split(df, groups=df["participant"].values):
        train_df, test_df = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        p = test_df["participant"].iloc[0]
        imputer = Imputer(ROLL_IMPUTE_COLS).fit(train_df)
        train_df_i, test_df_i = imputer.transform(train_df), imputer.transform(test_df)
        Xtr = train_df_i[ALL_MODEL_FEATURE_COLS].values.astype(float)
        Xte = test_df_i[ALL_MODEL_FEATURE_COLS].values.astype(float)
        ytr, yte = train_df_i["y"].values, test_df_i["y"].values
        proba = fit_predict_xgb(Xtr, ytr, Xte, seed=0)
        pred = proba.argmax(axis=1)
        m = compute_metrics(yte, pred, proba)
        m["participant"] = p
        m["n_windows"] = len(test_df)
        m["pct_high_windows"] = float((test_df_i["y"] == 2).mean())
        m["age"] = test_df_i["age"].iloc[0]
        m["acclimation"] = test_df_i["acclimation"].iloc[0]
        m["vo2peak"] = test_df_i["vo2peak"].iloc[0]
        rows.append(m)
    print(f"LOPO ({len(rows)} participants) done in {time.time()-t0:.1f}s")
    out = pd.DataFrame(rows).sort_values("accuracy")
    out.to_csv(f"{RESULTS_DIR}/tables/lopo_per_participant.csv", index=False)
    print(out[["participant", "n_windows", "accuracy", "macro_f1", "pct_high_windows", "vo2peak"]].to_string(index=False))
    print(f"\nMean={out['accuracy'].mean():.3f} Std={out['accuracy'].std():.3f} "
          f"Min={out['accuracy'].min():.3f} Max={out['accuracy'].max():.3f}")
    corr = out[["n_windows", "pct_high_windows", "age", "vo2peak", "accuracy"]].corr()["accuracy"].drop("accuracy")
    print("\nCorrelates of per-participant accuracy:")
    print(corr)
    corr.to_csv(f"{RESULTS_DIR}/tables/lopo_variance_correlates.csv")
    return out


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("loco", "both"):
        loco_condition_generalization()
    if which in ("lopo", "both"):
        lopo_error_analysis()
