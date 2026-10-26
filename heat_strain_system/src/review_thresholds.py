"""Label-threshold sensitivity: retrain XGBoost on labels built with different core-temperature cut points."""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_prospie import prep_fold, RESULTS_DIR
from load_prospie import load_prospie
from baselines import fit_predict_xgb
OUT = f"{RESULTS_DIR}/extra"
df = pd.read_parquet(f"{RESULTS_DIR}/tables/prospie_features.parquet").reset_index(drop=True)
raw = load_prospie()
# numeric Tre at t+15 for each window, using the same nearest-row rule as the feature builder
nxt = {}
for (p, c), tr in raw.groupby(["participant", "condition"]):
    tr = tr.sort_values("t_min").reset_index(drop=True); times = tr["t_min"].values
    for t0 in times:
        j = int(np.argmin(np.abs(times - (t0 + 15))))
        if abs(times[j] - (t0 + 15)) <= 3: nxt[(p, c, t0)] = tr["core_temp"].iloc[j]
df["core_next"] = [nxt.get((p, c, t), np.nan) for p, c, t in zip(df.participant, df.condition, df.t_min)]
base = np.where(df.core_next < 38.0, 0, np.where(df.core_next < 38.5, 1, 2))
print("label reproduction: agree with stored y on", round(float((base == df.y.values).mean()), 4), "of windows; NaN:", int(df.core_next.isna().sum()))
part = df["participant"].values; folds = list(GroupKFold(8).split(df, groups=part))
prepped = [prep_fold(df.iloc[tr].copy(), df.iloc[te].copy()) for tr, te in folds]
from common import compute_metrics
def uniform_f1(p): return float(np.mean([2 * pk * (1 / 3) / (pk + 1 / 3) for pk in p]))
rows = []
for name, (a, b) in {"38.0 / 38.5 (paper)": (38.0, 38.5), "38.0 / 39.0": (38.0, 39.0), "37.5 / 38.0": (37.5, 38.0), "37.5 / 38.5": (37.5, 38.5), "38.5 / 39.0": (38.5, 39.0)}.items():
    yy = np.where(df.core_next < a, 0, np.where(df.core_next < b, 1, 2)); share = np.bincount(yy, minlength=3) / len(yy)
    f1s, accs, rhs, aucs = [], [], [], []; maj_f1 = []
    for (tr, te), pp in zip(folds, prepped):
        trd, ted, Xtr, Xte, _, _, _, _ = pp
        pr = fit_predict_xgb(Xtr, yy[tr], Xte, seed=0)
        m = compute_metrics(yy[te], pr.argmax(1), pr); f1s.append(m["macro_f1"]); accs.append(m["accuracy"]); rhs.append(m["recall_High"]); aucs.append(m["roc_auc_ovr"])
        mj = np.bincount(yy[tr], minlength=3).argmax(); maj_f1.append(f1_score(yy[te], np.full(len(te), mj), average="macro", labels=[0, 1, 2], zero_division=0))
    rows.append(dict(cutpoints=name, share_low=share[0], share_mod=share[1], share_high=share[2], xgb_macro_f1=np.mean(f1s), xgb_sd=np.std(f1s, ddof=1),
                     xgb_acc=np.mean(accs), recall_high=np.mean(rhs), auroc_ovr=np.mean(aucs), majority_macro_f1=np.mean(maj_f1), uniform_macro_f1=uniform_f1(share)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).round(3).to_csv(f"{OUT}/threshold_sensitivity.csv", index=False)
