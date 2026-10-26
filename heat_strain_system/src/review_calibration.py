"""Corrected calibration protocol: temperature / vector scaling fitted on held-out VALIDATION PARTICIPANTS
that the model did not train on, using the same output (single pass) that is scored. Also MC-dropout uncertainty."""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import class_weights_from, expected_calibration_error, fit_temperature, apply_temperature, proba_to_logits
from evaluate_prospie import prep_fold, RESULTS_DIR, SEED
from fusion_model_prospie import train_fusion_model
from baselines import fit_predict_xgb
OUT = f"{RESULTS_DIR}/extra"
df = pd.read_parquet(f"{RESULTS_DIR}/tables/prospie_features.parquet").reset_index(drop=True)
part = df["participant"].values

def nll(p, y): return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))))
def brier(p, y): return float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1)))
def ece_high(p, y, nb=10):
    q = p[:, 2]; t = (y == 2).astype(float); e = 0; bins = np.linspace(0, 1, nb + 1)
    for i in range(nb):
        m = (q >= bins[i]) & (q < bins[i + 1] if i < nb - 1 else q <= 1)
        if m.sum(): e += m.mean() * abs(q[m].mean() - t[m].mean())
    return float(e)
def fit_vector(lg, y):
    def f(w):
        a, b = w[:3], w[3:]; z = lg * a + b; z -= z.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        return -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))) + 1e-3 * np.sum((w[:3] - 1) ** 2)
    r = minimize(f, np.r_[np.ones(3), np.zeros(3)], method="L-BFGS-B"); return r.x
def apply_vector(lg, w):
    z = lg * w[:3] + w[3:]; z -= z.max(1, keepdims=True); return np.exp(z) / np.exp(z).sum(1, keepdims=True)

N = len(df); store = {m: {k: np.zeros((N, 3)) for k in ["raw", "temp", "vec"]} for m in ["XGBoost", "Fusion"]}
mc_std = np.zeros(N); mc_prob = np.zeros((N, 3)); yv = df["y"].values; Ts = []
for k, (tr, te) in enumerate(GroupKFold(8).split(df, groups=part)):
    trd, ted, Xtr, Xte, ytr, yte, Xtr_m, Xte_m = prep_fold(df.iloc[tr].copy(), df.iloc[te].copy())
    rng = np.random.default_rng(SEED + k); subs = trd["participant"].unique()
    vs = set(rng.choice(subs, size=max(2, int(len(subs) * 0.15)), replace=False))
    vm = trd["participant"].isin(vs).values
    # XGBoost: fit on non-val, calibrate on val, then predict test with the same model
    pv = fit_predict_xgb(Xtr[~vm], ytr[~vm], np.vstack([Xtr[vm], Xte]), seed=SEED)
    pval, ptest = pv[:vm.sum()], pv[vm.sum():]
    T = fit_temperature(proba_to_logits(pval), ytr[vm]); w = fit_vector(proba_to_logits(pval), ytr[vm])
    store["XGBoost"]["raw"][te] = ptest
    store["XGBoost"]["temp"][te] = apply_temperature(proba_to_logits(ptest), T)
    store["XGBoost"]["vec"][te] = apply_vector(proba_to_logits(ptest), w)
    # Fusion: train on non-val participants only
    cw = class_weights_from(ytr[~vm])
    sub = lambda X, m: {a: v[m] for a, v in X.items()}
    model = train_fusion_model(sub(Xtr_m, ~vm), ytr[~vm], trd["participant"].values[~vm], epochs=100, lr=3e-3, batch_size=64,
                               class_weights=cw, grl_lambda=0.15, use_focal=True, focal_gamma=1.0, seed=SEED)
    pval, _ = model.predict_proba(sub(Xtr_m, vm), mc_samples=1); ptest, _ = model.predict_proba(Xte_m, mc_samples=1)
    T = fit_temperature(proba_to_logits(pval), ytr[vm]); w = fit_vector(proba_to_logits(pval), ytr[vm]); Ts.append(T)
    store["Fusion"]["raw"][te] = ptest
    store["Fusion"]["temp"][te] = apply_temperature(proba_to_logits(ptest), T)
    store["Fusion"]["vec"][te] = apply_vector(proba_to_logits(ptest), w)
    mp, ms = model.predict_proba(Xte_m, mc_samples=20); mc_prob[te] = mp; mc_std[te] = ms.max(1)
    print(f"fold {k} T_fusion={T:.2f} val_n={vm.sum()}", flush=True)

rows = []
for m, d in store.items():
    for kind, p in d.items():
        rows.append(dict(model=m, calibration={"raw": "none", "temp": "temperature", "vec": "vector"}[kind],
                         ece=expected_calibration_error(yv, p), ece_high=ece_high(p, yv), nll=nll(p, yv), brier=brier(p, yv),
                         acc=float((p.argmax(1) == yv).mean())))
pd.DataFrame(rows).round(4).to_csv(f"{OUT}/calibration_fixed.csv", index=False); print(pd.DataFrame(rows).round(4))
np.savez(f"{OUT}/calibration_fixed_probs.npz", y=yv, **{f"{m}_{k}": v for m, d in store.items() for k, v in d.items()})
# uncertainty vs error (MC dropout, 20 passes) on test windows
ent = -np.sum(mc_prob * np.log(np.clip(mc_prob, 1e-9, 1)), axis=1); err = (mc_prob.argmax(1) != yv)
ur = []
for cov in [1.0, 0.8, 0.6, 0.4]:
    keep = ent <= np.quantile(ent, cov); ur.append(dict(coverage=cov, accuracy=float(1 - err[keep].mean()), n=int(keep.sum())))
pd.DataFrame(ur).round(4).to_csv(f"{OUT}/mc_dropout_abstention.csv", index=False); print(pd.DataFrame(ur).round(4))
print("AUROC of entropy for error:", end=" ")
from sklearn.metrics import roc_auc_score; print(round(roc_auc_score(err, ent), 3))
