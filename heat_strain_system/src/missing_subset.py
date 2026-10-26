import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupKFold
from common import compute_metrics
from evaluate_prospie import prep_fold, RESULTS_DIR, N_SPLITS
from fusion_model_prospie import MODALITIES, ALL_MODEL_FEATURE_COLS
from baselines import fit_predict_xgb
X = f"{RESULTS_DIR}/extra"
df = pd.read_parquet(f"{RESULTS_DIR}/tables/prospie_features.parquet")
miss = df[["hr", "breathing_rate", "skin_chest", "skin_insulated", "ta_chamber", "cond_wbgt", "vo2peak", "bodyfat_pct"]].isna().mean() * 100
print("% windows missing:\n", miss.round(2))
tr = df.groupby(["participant", "condition"]).agg(n=("y", "size"), hr_miss=("hr", lambda s: s.isna().mean()), br_miss=("breathing_rate", lambda s: s.isna().mean()))
print("trials:", len(tr), "| trials with any HR missing:", int((tr.hr_miss > 0).sum()), "| any BR missing:", int((tr.br_miss > 0).sum()),
      "| BR fully missing:", int((tr.br_miss == 1).sum()), "| HR fully missing:", int((tr.hr_miss == 1).sum()))
all_miss = df[ALL_MODEL_FEATURE_COLS].isna().mean() * 100
print("top missing feature cols:\n", all_miss.sort_values(ascending=False).head(8).round(2))
miss.to_frame("pct_windows_missing").to_csv(f"{X}/missingness_selected.csv")
comp = df[df.hr.notna() & df.breathing_rate.notna()].reset_index(drop=True)
print("complete HR+BR windows:", len(comp), "of", len(df), "| participants:", comp.participant.nunique())
gk = list(GroupKFold(n_splits=N_SPLITS).split(comp, groups=comp.participant.values))
out = []
for name, cols in [("all features", ALL_MODEL_FEATURE_COLS), ("cardio-resp only", MODALITIES["cardio_resp"])]:
    ms = []
    for a, b in gk:
        tdf, sdf, _, _, ytr, yte, _, _ = prep_fold(comp.iloc[a].copy(), comp.iloc[b].copy())
        p = fit_predict_xgb(tdf[cols].values.astype(float), ytr, sdf[cols].values.astype(float), seed=0)
        ms.append(compute_metrics(yte, p.argmax(1), p))
    m = pd.DataFrame(ms)
    out.append(dict(subset="complete HR+BR windows", features=name, windows=len(comp), macro_f1=m.macro_f1.mean(), sd=m.macro_f1.std(), acc=m.accuracy.mean()))
    print(out[-1], flush=True)
pd.DataFrame(out).to_csv(f"{X}/complete_hrbr_subset.csv", index=False)
