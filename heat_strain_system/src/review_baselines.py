"""Reviewer-requested baselines, tests and sensor-set runs (same 8 grouped folds as the main CV)."""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, recall_score, cohen_kappa_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_prospie import prep_fold, RESULTS_DIR, ALL_MODEL_FEATURE_COLS
from baselines import fit_predict_xgb
OUT = f"{RESULTS_DIR}/extra"
df = pd.read_parquet(f"{RESULTS_DIR}/tables/prospie_features.parquet").reset_index(drop=True)
y = df["y"].values; part = df["participant"].values
lab2i = {"Low": 0, "Moderate": 1, "High": 2}
df["y_now"] = df["label_now"].map(lab2i)

# ---- PSI (oracle: needs measured Tre) ----
def psi_col(g):
    g = g.sort_values("t_min")
    c0 = g["core_temp_now"].iloc[0]
    h = g["hr"]; h0 = h.dropna().iloc[0] if h.notna().any() else np.nan
    th = 5 * (g["core_temp_now"] - c0) / (39.5 - c0)
    ca = 5 * (h - h0) / (180 - h0) if not np.isnan(h0) else 0 * th
    return (th + ca.fillna(0)).rename("psi")
df["psi"] = df.groupby(["participant", "condition"], group_keys=False).apply(psi_col)

def fit_thr(score, yy):
    qs = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 49)))
    best, bt = -1, (qs[0], qs[-1])
    for i, a in enumerate(qs):
        for b in qs[i + 1:]:
            p = np.where(score < a, 0, np.where(score < b, 1, 2))
            f = f1_score(yy, p, average="macro", labels=[0, 1, 2], zero_division=0)
            if f > best: best, bt = f, (a, b)
    return bt

gkf = GroupKFold(n_splits=8)
splits = list(gkf.split(df, groups=part))
preds = {}  # name -> pooled predicted class
rows = []
def record(name, fold, te, pred):
    preds.setdefault(name, np.full(len(df), -1))[te] = pred
    yt = y[te]
    rows.append(dict(model=name, fold=fold,
        macro_f1=f1_score(yt, pred, average="macro", labels=[0, 1, 2], zero_division=0),
        accuracy=accuracy_score(yt, pred),
        recall_High=recall_score(yt, pred, labels=[2], average=None, zero_division=0)[0]))

for k, (tr, te) in enumerate(splits):
    trd, ted = df.iloc[tr], df.iloc[te]
    # majority baselines
    maj = np.bincount(y[tr], minlength=3).argmax()
    record("Majority (pooled)", k, te, np.full(len(te), maj))
    sc_maj = trd.groupby("condition")["y"].agg(lambda s: np.bincount(s, minlength=3).argmax())
    record("Majority per scenario", k, te, ted["condition"].map(sc_maj).fillna(maj).astype(int).values)
    pp = np.zeros(len(te), int)
    for p_ in np.unique(part[te]):
        m = part[te] == p_; pp[m] = np.bincount(y[te][m], minlength=3).argmax()
    record("Majority per participant (oracle)", k, te, pp)
    # time-only models
    for name, cols in [("Time-only (XGBoost)", ["t_min"]), ("Time + scenario (XGBoost)", ["t_min", "condition"]),
                       ("WBGT + time (XGBoost)", ["cond_wbgt", "t_min"])]:
        pr = fit_predict_xgb(trd[cols].values.astype(float), y[tr], ted[cols].values.astype(float), seed=0)
        record(name, k, te, pr.argmax(1))
    # rule-based
    a, b = fit_thr(trd["cond_wbgt"].values, y[tr])
    record("WBGT threshold rule", k, te, np.where(ted["cond_wbgt"] < a, 0, np.where(ted["cond_wbgt"] < b, 1, 2)))
    a, b = fit_thr(trd["psi"].values, y[tr])
    record("PSI tiers (oracle, uses Tre)", k, te, np.where(ted["psi"] < a, 0, np.where(ted["psi"] < b, 1, 2)))
    record("Tre persistence (oracle)", k, te, ted["y_now"].values)
    # extra learners and sensor sets
    trd2, ted2, Xtr, Xte, ytr, yte, _, _ = prep_fold(trd.copy(), ted.copy())
    sc = StandardScaler().fit(Xtr)
    mlp = MLPClassifier((64, 32), alpha=1e-3, max_iter=200, random_state=0).fit(sc.transform(Xtr), ytr)
    record("MLP (early fusion)", k, te, mlp.predict(sc.transform(Xte)))
    hgb = HistGradientBoostingClassifier(class_weight="balanced", random_state=0).fit(Xtr, ytr)
    record("HistGradientBoosting", k, te, hgb.predict(Xte))
    idx = {c: i for i, c in enumerate(ALL_MODEL_FEATURE_COLS)}
    def cols_for(bases):
        out = []
        for b_ in bases:
            out += [c for c in ALL_MODEL_FEATURE_COLS if c == b_ or c in (b_ + "_roll_mean", b_ + "_roll_slope")]
        return [idx[c] for c in out]
    sets = {"Chest patch (HR, BR, chest skin, time)": ["hr", "breathing_rate", "skin_chest", "t_min"],
            "Patch + 2 skin sites (thigh, upper arm)": ["hr", "breathing_rate", "skin_chest", "skin_thigh", "skin_upperarm", "t_min"],
            "Wrist HR + ambient (air T, RH, time)": ["hr", "ta_chamber", "rh_chamber", "t_min"]}
    for name, bases in sets.items():
        ci = cols_for(bases)
        pr = fit_predict_xgb(Xtr[:, ci], ytr, Xte[:, ci], seed=0)
        record(name + " [XGBoost]", k, te, pr.argmax(1))
    print("fold", k, "done", flush=True)

res = pd.DataFrame(rows)
summ = res.groupby("model").agg(macro_f1=("macro_f1", "mean"), sd=("macro_f1", "std"), accuracy=("accuracy", "mean"),
                                recall_High=("recall_High", "mean"))
summ["qwk_pooled"] = [cohen_kappa_score(y, preds[m], weights="quadratic") for m in summ.index]
# reference learned models from existing out-of-fold files (seed 0)
for nm, f in [("XGBoost (full)", "clf_XGBoost_s0.npz"), ("Random Forest (full)", "clf_RandomForest_s0.npz"),
              ("Logistic Regression (full)", "clf_LogReg_s0.npz")]:
    z = np.load(f"{OUT}/{f}"); pr = z["proba"].argmax(1)
    fm = [f1_score(y[te], pr[te], average="macro", labels=[0, 1, 2]) for _, te in splits]
    summ.loc[nm] = [np.mean(fm), np.std(fm, ddof=1), np.mean([accuracy_score(y[te], pr[te]) for _, te in splits]),
                    np.mean([recall_score(y[te], pr[te], labels=[2], average=None)[0] for _, te in splits]),
                    cohen_kappa_score(y, pr, weights="quadratic")]
summ.round(3).sort_values("macro_f1").to_csv(f"{OUT}/baselines_summary.csv")
res.to_csv(f"{OUT}/baselines_by_fold.csv", index=False)
print(summ.round(3).sort_values("macro_f1"))

# per-participant majority vs LOPO XGBoost (pooled majority was 0.489)
lopo = pd.read_csv(f"{RESULTS_DIR}/tables/lopo_per_participant.csv")
pm = df.groupby("participant")["y"].agg(lambda s: np.bincount(s, minlength=3).max() / len(s))
lopo_cols = lopo.columns.tolist(); print(lopo_cols)
pcol = [c for c in lopo_cols if "particip" in c.lower()][0]
lopo["own_majority"] = lopo[pcol].map(pm)
lopo[[pcol, "accuracy", "own_majority"]].to_csv(f"{OUT}/lopo_vs_own_majority.csv", index=False)
print("LOPO XGB beats own-majority for", int((lopo.accuracy > lopo.own_majority).sum()), "of", len(lopo), "participants; mean own-majority", round(lopo.own_majority.mean(), 3))

# ---- Holm tests on the original (Table IV) fold results ----
cv = pd.read_csv(f"{RESULTS_DIR}/tables/cv_fold_results.csv")
pv = cv.pivot_table(index="fold", columns="model", values="macro_f1")
rows2 = []
for other in ["RandomForest", "LogReg", "fusion_full"]:
    a, b = pv["XGBoost"].values, pv[other].values
    rows2.append(dict(comparison=f"XGBoost vs {other}", mean_diff=np.mean(a - b), p_t=stats.ttest_rel(a, b).pvalue,
                      p_wilcoxon=stats.wilcoxon(a, b).pvalue))
t = pd.DataFrame(rows2)
for col in ["p_t", "p_wilcoxon"]:
    order = np.argsort(t[col].values); m = len(t); adj = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * t[col].values[i]); adj[i] = min(1, run)
    t[col + "_holm"] = adj
t.round(4).to_csv(f"{OUT}/holm_xgboost_vs_others.csv", index=False); print(t.round(4))

# ---- participant bootstrap on pooled OOF predictions (seed 0) ----
P = {"XGBoost": np.load(f"{OUT}/clf_XGBoost_s0.npz")["proba"].argmax(1),
     "Random Forest": np.load(f"{OUT}/clf_RandomForest_s0.npz")["proba"].argmax(1),
     "Logistic Regression": np.load(f"{OUT}/clf_LogReg_s0.npz")["proba"].argmax(1)}
fp = np.full((len(df), 3), np.nan)
for k in range(8):
    z = np.load(f"{OUT}/fusion_s0_fold{k}.npz"); fp[z["idx"]] = z["proba"]
P["Fusion (full)"] = fp.argmax(1)
ups = np.unique(part)
def conf(pr, m):
    return np.bincount(y[m] * 3 + pr[m], minlength=9).reshape(3, 3)
C = {n: np.stack([conf(p, part == u) for u in ups]) for n, p in P.items()}
def mf1(c):
    tp = np.diag(c); pr_ = c.sum(0); re_ = c.sum(1)
    f = np.where(pr_ + re_ > 0, 2 * tp / np.maximum(pr_ + re_, 1), 0); return f.mean()
rng = np.random.default_rng(0); B = 2000
boot = {n: [] for n in P}
for _ in range(B):
    s = rng.integers(0, len(ups), len(ups))
    for n in P: boot[n].append(mf1(C[n][s].sum(0)))
out = []
for n in P:
    v = np.array(boot[n]); out.append(dict(item=n, point=mf1(C[n].sum(0)), lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5)))
for n in ["Random Forest", "Logistic Regression", "Fusion (full)"]:
    d = np.array(boot["XGBoost"]) - np.array(boot[n])
    out.append(dict(item=f"XGBoost minus {n}", point=mf1(C["XGBoost"].sum(0)) - mf1(C[n].sum(0)), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
pd.DataFrame(out).round(3).to_csv(f"{OUT}/bootstrap_ci.csv", index=False); print(pd.DataFrame(out).round(3))
