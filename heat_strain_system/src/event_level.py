"""Event-level view of the forecasts: for each held-out trial, when does the first High forecast
(or the first run of three consecutive High forecasts) appear relative to the moment measured Tre first reaches 38.5 C?"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from evaluate_prospie import RESULTS_DIR
from sklearn.model_selection import GroupKFold
from common import compute_metrics
X = f"{RESULTS_DIR}/extra"
df = pd.read_parquet(f"{RESULTS_DIR}/tables/prospie_features.parquet"); N = len(df)
def pooled_fusion(seed=0):
    P = np.zeros((N, 3))
    for f in range(8):
        z = np.load(f"{X}/fusion_s{seed}_fold{f}.npz"); P[z["idx"]] = z["proba"]
    return P
out = []
for nm, P in [("XGBoost", np.load(f"{X}/clf_XGBoost_s0.npz")["proba"]), ("Fusion (full)", pooled_fusion(0))]:
    d = df[["participant", "condition", "t_min", "core_temp_now"]].copy(); d["pred"] = P.argmax(1)
    rows = []
    for (p, c), g in d.groupby(["participant", "condition"]):
        g = g.sort_values("t_min"); cross = g[g.core_temp_now >= 38.5]
        hp = (g.pred == 2).values; t = g.t_min.values
        pers = np.convolve(hp.astype(int), np.ones(3, int), "valid") >= 3
        rows.append(dict(reach=len(cross) > 0, t_cross=cross.t_min.iloc[0] if len(cross) else np.nan,
                         t_first=t[np.argmax(hp)] if hp.any() else np.nan, t_pers=t[np.argmax(pers)] if pers.any() else np.nan))
    r = pd.DataFrame(rows); pos, neg = r[r.reach], r[~r.reach]
    lead, leadp = pos.t_cross - pos.t_first, pos.t_cross - pos.t_pers
    out.append(dict(model=nm, trials=len(r), trials_reaching_385=len(pos), alerted=int(pos.t_first.notna().sum()),
                    alert_before_crossing=int((lead > 0).sum()), median_lead_min=lead.median(), lead_q25=lead.quantile(.25), lead_q75=lead.quantile(.75),
                    persistent_before_crossing=int((leadp > 0).sum()), median_lead_persistent=leadp.median(),
                    trials_never_reaching=len(neg), false_alert_trials=int(neg.t_first.notna().sum()), false_alert_trials_persistent=int(neg.t_pers.notna().sum())))
pd.DataFrame(out).to_csv(f"{X}/event_level.csv", index=False); print(pd.DataFrame(out).T)
# fold-level sd for the XGBoost seed-0 reference row
gk = list(GroupKFold(n_splits=8).split(df, groups=df["participant"].values))
P = np.load(f"{X}/clf_XGBoost_s0.npz")["proba"]; y = df.y.values
f1 = [compute_metrics(y[te], P[te].argmax(1), P[te])["macro_f1"] for _, te in gk]
print("XGB seed0 fold macro-F1 mean/sd", np.mean(f1).round(3), np.std(f1, ddof=1).round(3))
