"""Event-level view with an explicit lead definition. A forecast issued at minute t refers to the state at t+15, so a
High forecast issued 0 < lead <= 15 min before the first 38.5 C reading is 'on time'; lead > 15 is an 'early' alert."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_prospie import RESULTS_DIR
X = f"{RESULTS_DIR}/extra"
df = pd.read_parquet(f"{RESULTS_DIR}/tables/prospie_features.parquet"); N = len(df)
Pf = np.zeros((N, 3))
for f in range(8):
    z = np.load(f"{X}/fusion_s0_fold{f}.npz"); Pf[z["idx"]] = z["proba"]
out = []
for nm, P in [("XGBoost", np.load(f"{X}/clf_XGBoost_s0.npz")["proba"]), ("Fusion (full)", Pf)]:
    d = df[["participant", "condition", "t_min", "core_temp_now", "y"]].copy(); d["pred"] = P.argmax(1)
    rows = []
    for (p, c), g in d.groupby(["participant", "condition"]):
        g = g.sort_values("t_min"); cross = g[g.core_temp_now >= 38.5]; t = g.t_min.values
        hp = (g.pred == 2).values; pers = np.convolve(hp.astype(int), np.ones(3, int), "valid") >= 3
        rows.append(dict(reach=len(cross) > 0, t_cross=cross.t_min.iloc[0] if len(cross) else np.nan,
                         t_first=t[np.argmax(hp)] if hp.any() else np.nan, t_pers=t[np.argmax(pers)] if pers.any() else np.nan))
    r = pd.DataFrame(rows); pos, neg = r[r.reach], r[~r.reach]
    lead = (pos.t_cross - pos.t_first); lp = (pos.t_cross - pos.t_pers)
    ontime = lead[(lead > 0) & (lead <= 15)]
    nonhigh = d[d.y != 2]; fa_per_hour = float((nonhigh.pred == 2).sum() / (len(d) / 60.0))
    out.append(dict(model=nm, reaching=len(pos), no_alert_before_crossing=int(((lead <= 0) | lead.isna()).sum()),
                    ontime=int(len(ontime)), early_gt15=int((lead > 15).sum()),
                    ontime_median=ontime.median(), ontime_q25=ontime.quantile(.25), ontime_q75=ontime.quantile(.75),
                    persistent_ontime=int(((lp > 0) & (lp <= 15)).sum()), persistent_early=int((lp > 15).sum()),
                    never_reaching=len(neg), never_with_any_high=int(neg.t_first.notna().sum()), never_with_persistent=int(neg.t_pers.notna().sum()),
                    false_high_forecasts_per_hour=fa_per_hour))
o = pd.DataFrame(out); o.round(3).to_csv(f"{X}/event_level_v2.csv", index=False); print(o.round(3).T)
