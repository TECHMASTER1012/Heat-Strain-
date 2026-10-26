"""PHS (ISO 7933 via pythermalcomfort) and aPSI baselines on the same 8 participant-grouped folds.
PHS is a NON-oracle forecast from scenario conditions and person data; aPSI is an ORACLE (uses measured Tre, Tsk, HR)."""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, recall_score, cohen_kappa_score
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_prospie import RESULTS_DIR
from pythermalcomfort.models import phs
OUT = f"{RESULTS_DIR}/extra"
df = pd.read_parquet(f"{RESULTS_DIR}/tables/prospie_features.parquet").reset_index(drop=True)
y = df["y"].values; part = df["participant"].values
SKIN11 = ["skin_abdomen", "skin_calf", "skin_chest", "skin_foot", "skin_hand", "skin_head", "skin_lowerarm", "skin_lowerback", "skin_thigh", "skin_upperarm", "skin_upperback"]

# ---- nominal scenario conditions from the dataset's Conditions sheet ----
NOM = {1: (40, 35), 2: (33, 85), 3: (40, 35), 6: (25, 50), 7: (40, 20), 8: (40, 20), 9: (30, 35), 10: (40, 20), 11: (40, 35)}
# clothing is not given as insulation values; assumed (documented in the paper): permeable ensemble 0.9 clo, im 0.38; impermeable 1.0 clo, im 0.10
CLO = {1: (0.9, 0.38), 2: (1.0, 0.10)}
# ---- metabolic rate per trial from treadmill settings (Givoni-Goldman form used in ISO 8996 walking), average over the trial ----
def bsa(m, h): return 0.007184 * m ** 0.425 * h ** 0.725
rows = []
for (p, c), g in df.groupby(["participant", "condition"]):
    m = g["bodymass_kg"].iloc[0]; h = g["height_cm"].iloc[0]; A = bsa(m, h)
    v = g["treadmill_speed"].fillna(0).values / 3.6; gr = g["treadmill_grade"].fillna(0).values
    M = np.where(v > 0, 1.5 * m + m * (1.5 * v ** 2 + 0.35 * v * gr), 1.2 * 58.15 * A)   # W; rest = 1.2 met
    Wext = np.where(v > 0, m * 9.81 * v * gr / 100.0, 0.0)
    met = M.mean() / A / 58.15; wme = Wext.mean() / A / 58.15
    spd = v[v > 0].mean() if (v > 0).any() else 0.0
    cl = CLO[int(g["clothing"].iloc[0])]
    rows.append(dict(participant=p, condition=c, met=met, wme=wme, walk=spd, weight=m, height=h / 100, wind=float(g["wind_chamber"].mean()),
                     tdb=NOM[c][0], rh=NOM[c][1], clo=cl[0], im=cl[1]))
tr_par = pd.DataFrame(rows)
df = df.merge(tr_par, on=["participant", "condition"], how="left")
# PHS predicted Tre at t+15 (duration = elapsed time + 15), vectorised over windows sharing the same duration
df["dur"] = (df["t_min"] + 15).round().astype(int)
from joblib import Parallel, delayed
def one(r):
    o = phs(tdb=r.tdb, tr=r.tdb, v=max(r.wind, 0.1), rh=r.rh, met=r.met, clo=r.clo, posture="standing", wme=r.wme, i_mst=r.im,
            weight=r.weight, height=r.height, walk_sp=r.walk, acclimatized=0, duration=int(r.dur), model="7933-2004", limit_inputs=False, round_output=False)
    return float(np.ravel(o.t_re)[0])
df["phs_tre"] = Parallel(n_jobs=4, batch_size=64)(delayed(one)(r) for r in df.itertuples())
print("PHS predicted Tre range", df.phs_tre.min().round(2), df.phs_tre.max().round(2))
# diagnostic: PHS vs measured Tre at t+15
core_next = df["label_next"].map({"Low": 37.5, "Moderate": 38.25, "High": 38.9})  # class midpoints only for class-level bias
print("PHS predicted class share:", np.bincount(np.where(df.phs_tre < 38.0, 0, np.where(df.phs_tre < 38.5, 1, 2)), minlength=3) / len(df))
# ---- aPSI (oracle) ----
def apsi(g):
    g = g.sort_values("t_min"); tc0 = g["core_temp_now"].iloc[0]
    hr0 = g["hr"].dropna().iloc[:5].median() if g["hr"].notna().any() else np.nan
    tsk = g[SKIN11].mean(axis=1); crit = 39.5 + (g["core_temp_now"] - tsk - 4) / 4
    thermal = 5 * (g["core_temp_now"] - tc0) / (crit - tc0)
    card = 5 * (g["hr"] - hr0) / (180 - hr0) if not np.isnan(hr0) else 0 * thermal
    return (thermal + card.fillna(0)).rename("apsi")
df["apsi"] = df.groupby(["participant", "condition"], group_keys=False).apply(apsi)
def fit_thr(score, yy):
    qs = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 49))); best, bt = -1, (qs[0], qs[-1])
    for i, a in enumerate(qs):
        for b in qs[i + 1:]:
            p = np.where(score < a, 0, np.where(score < b, 1, 2)); f = f1_score(yy, p, average="macro", labels=[0, 1, 2], zero_division=0)
            if f > best: best, bt = f, (a, b)
    return bt
res = []; pooled = {}
def rec(name, k, te, pred):
    pooled.setdefault(name, np.full(len(df), -1))[te] = pred
    res.append(dict(model=name, fold=k, macro_f1=f1_score(y[te], pred, average="macro", labels=[0, 1, 2], zero_division=0),
                    accuracy=accuracy_score(y[te], pred), recall_High=recall_score(y[te], pred, labels=[2], average=None, zero_division=0)[0]))
for k, (tr, te) in enumerate(GroupKFold(8).split(df, groups=part)):
    d = df.iloc[te]
    rec("PHS, ISO thresholds 38.0/38.5 (no fitting)", k, te, np.where(d.phs_tre < 38.0, 0, np.where(d.phs_tre < 38.5, 1, 2)))
    a, b = fit_thr(df.phs_tre.values[tr], y[tr]); rec("PHS, thresholds fitted on training folds", k, te, np.where(d.phs_tre < a, 0, np.where(d.phs_tre < b, 1, 2)))
    a, b = fit_thr(df.apsi.values[tr], y[tr]); rec("aPSI tiers (oracle), thresholds fitted", k, te, np.where(d.apsi < a, 0, np.where(d.apsi < b, 1, 2)))
r = pd.DataFrame(res); s = r.groupby("model").agg(macro_f1=("macro_f1", "mean"), sd=("macro_f1", "std"), accuracy=("accuracy", "mean"), recall_High=("recall_High", "mean"))
s["qwk_pooled"] = [cohen_kappa_score(y, pooled[m], weights="quadratic") for m in s.index]
s.round(3).to_csv(f"{OUT}/index_baselines.csv"); print(s.round(3))
# PHS predicted-Tre bias per true class (diagnostic)
print(df.groupby("label_next").phs_tre.agg(["mean", "std"]).round(2))
df[["participant", "condition", "t_min", "phs_tre", "apsi", "met", "wme"]].to_csv(f"{OUT}/index_values.csv", index=False)
