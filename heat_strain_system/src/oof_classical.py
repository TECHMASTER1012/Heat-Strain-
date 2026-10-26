"""Classical-model out-of-fold analysis: multi-seed OOF probabilities (LogReg/RF/XGB),
single-modality and leave-one-modality-out XGBoost, and a label-permutation sanity check."""
import os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(__file__))
from common import compute_metrics
from evaluate_prospie import prep_fold, RESULTS_DIR, N_SPLITS
from fusion_model_prospie import MODALITIES, ALL_MODEL_FEATURE_COLS
from baselines import fit_predict_logreg, fit_predict_rf, fit_predict_xgb

OUT = f"{RESULTS_DIR}/extra"
df = pd.read_parquet(f"{RESULTS_DIR}/tables/prospie_features.parquet")
gkf = list(GroupKFold(n_splits=N_SPLITS).split(df, groups=df["participant"].values))
N = len(df)

def run_cv(fn, cols, seed, y_override=None):
    proba = np.zeros((N, 3)); ms = []
    for tr, te in gkf:
        tdf, sdf, _, _, ytr, yte, _, _ = prep_fold(df.iloc[tr].copy(), df.iloc[te].copy())
        if y_override is not None:
            ytr, yte = y_override[tr], y_override[te]
        Xtr = tdf[cols].values.astype(float); Xte = sdf[cols].values.astype(float)
        p = fn(Xtr, ytr, Xte, seed=seed)
        proba[te] = p
        ms.append(compute_metrics(yte, p.argmax(1), p))
    return proba, pd.DataFrame(ms)

if __name__ == "__main__":
    y = df["y"].values
    rows = []
    for seed in (0, 1, 2):
        for name, fn in [("LogReg", fit_predict_logreg), ("RandomForest", fit_predict_rf), ("XGBoost", fit_predict_xgb)]:
            p, m = run_cv(fn, ALL_MODEL_FEATURE_COLS, seed)
            np.savez(f"{OUT}/clf_{name}_s{seed}.npz", proba=p, y=y)
            rows.append(dict(model=name, seed=seed, macro_f1=m.macro_f1.mean(), acc=m.accuracy.mean()))
            print(name, seed, round(m.macro_f1.mean(), 4), round(m.accuracy.mean(), 4), flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/classical_seed_summary.csv", index=False)

    # modality studies with XGBoost (seed 0)
    mrows = []
    for mod, cols in MODALITIES.items():
        _, m = run_cv(fit_predict_xgb, cols, 0)
        mrows.append(dict(setting=f"only {mod}", n_features=len(cols), macro_f1=m.macro_f1.mean(), macro_f1_sd=m.macro_f1.std(),
                          acc=m.accuracy.mean(), recall_high=m.recall_High.mean()))
        rest = [c for k, v in MODALITIES.items() if k != mod for c in v]
        _, m = run_cv(fit_predict_xgb, rest, 0)
        mrows.append(dict(setting=f"all except {mod}", n_features=len(set(rest)), macro_f1=m.macro_f1.mean(), macro_f1_sd=m.macro_f1.std(),
                          acc=m.accuracy.mean(), recall_high=m.recall_High.mean()))
        print(mrows[-2], mrows[-1], flush=True)
    pd.DataFrame(mrows).to_csv(f"{OUT}/xgb_modality_studies.csv", index=False)

    # label-permutation sanity check
    prow = []
    rng = np.random.default_rng(123)
    for r in range(5):
        yp = rng.permutation(y)
        _, m = run_cv(fit_predict_xgb, ALL_MODEL_FEATURE_COLS, 0, y_override=yp)
        prow.append(dict(rep=r, macro_f1=m.macro_f1.mean(), acc=m.accuracy.mean()))
        print("perm", prow[-1], flush=True)
    pd.DataFrame(prow).to_csv(f"{OUT}/permutation_test.csv", index=False)
