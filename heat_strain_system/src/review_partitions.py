"""Repeat participant-grouped 8-fold CV over several random partitions of the 40 participants (learners with fixed seed)."""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_prospie import prep_fold, RESULTS_DIR
from baselines import fit_predict_logreg, fit_predict_rf, fit_predict_xgb
df = pd.read_parquet(f"{RESULTS_DIR}/tables/prospie_features.parquet").reset_index(drop=True)
part = df["participant"].values; y = df["y"].values; ups = np.unique(part); rows = []
for rep in range(5):
    rng = np.random.default_rng(100 + rep); perm = rng.permutation(ups); grp = {u: i % 8 for i, u in enumerate(perm)}
    g = np.array([grp[u] for u in part])
    for k in range(8):
        tr, te = np.where(g != k)[0], np.where(g == k)[0]
        trd, ted, Xtr, Xte, ytr, yte, _, _ = prep_fold(df.iloc[tr].copy(), df.iloc[te].copy())
        for nm, fn in [("LogReg", fit_predict_logreg), ("RandomForest", fit_predict_rf), ("XGBoost", fit_predict_xgb)]:
            p = fn(Xtr, ytr, Xte, seed=0).argmax(1)
            rows.append(dict(rep=rep, fold=k, model=nm, macro_f1=f1_score(yte, p, average="macro", labels=[0, 1, 2], zero_division=0)))
    print("rep", rep, flush=True)
    pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/extra/partitions_by_fold.csv", index=False)
r = pd.DataFrame(rows); per = r.groupby(["model", "rep"]).macro_f1.mean().unstack(0)
summ = per.agg(["mean", "std", "min", "max"]).T; summ.round(3).to_csv(f"{RESULTS_DIR}/extra/partitions_summary.csv"); print(summ.round(3))
print("XGB highest in", int((per.XGBoost > per[["LogReg", "RandomForest"]].max(axis=1)).sum()), "of 5 partitions")
