"""Builds every extra table/figure from REAL out-of-fold predictions (no synthetic values)."""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    accuracy_score, f1_score)
sys.path.insert(0, os.path.dirname(__file__))
from common import expected_calibration_error, reliability_curve

R = os.environ.get("HS_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")) + "/results"; X = f"{R}/extra"; FIG = f"{X}/figs"; os.makedirs(FIG, exist_ok=True)
df = pd.read_parquet(f"{R}/tables/prospie_features.parquet"); N = len(df); y = df["y"].values
LAB = ["Low", "Moderate", "High"]; COL = ["#2e9e5b", "#e8a33d", "#c0392b"]
MODS = ["cardio_resp", "skin_thermal", "environment", "individual"]
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.8,
                     "axes.spines.top": False, "axes.spines.right": False})
SEEDS = [s for s in (0, 1, 2) if len(glob.glob(f"{X}/fusion_s{s}_fold*.npz")) == 8]
print("seeds available:", SEEDS)

def load_fusion(seed):
    P = np.zeros((N, 3)); W = np.zeros((N, 4)); hist = []; restarts = 0
    for f in range(8):
        z = np.load(f"{X}/fusion_s{seed}_fold{f}.npz")
        P[z["idx"]] = z["proba"]; W[z["idx"]] = z["w"]; hist.append(z["hist"]); restarts += int(z["restarts"])
    return P, W, np.stack(hist), restarts

def load_clf(name, seed):
    return np.load(f"{X}/clf_{name}_s{seed}.npz")["proba"]

def metrics(p):
    pred = p.argmax(1)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    out = dict(accuracy=accuracy_score(y, pred), balanced_acc=balanced_accuracy_score(y, pred), macro_f1=f1.mean(),
               kappa=cohen_kappa_score(y, pred), mcc=matthews_corrcoef(y, pred),
               auroc_macro=roc_auc_score(y, p, multi_class="ovr", average="macro"),
               auprc_high=average_precision_score((y == 2).astype(int), p[:, 2]),
               brier=float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1))), ece=expected_calibration_error(y, p))
    for i, l in enumerate(LAB):
        out[f"prec_{l}"] = pr[i]; out[f"rec_{l}"] = rc[i]; out[f"f1_{l}"] = f1[i]
    return out

models = {}
F = {s: load_fusion(s) for s in SEEDS}
models["Fusion (full)"] = {s: F[s][0] for s in SEEDS}
for nm, key in [("XGBoost", "XGBoost"), ("Random Forest", "RandomForest"), ("Logistic Regression", "LogReg")]:
    models[nm] = {s: load_clf(key, s) for s in SEEDS}

# ---------- pooled OOF metrics: mean +- sd over seeds ----------
rows = []
for nm, d in models.items():
    ms = pd.DataFrame([metrics(d[s]) for s in SEEDS])
    r = {"model": nm}
    for c in ms.columns: r[c] = f"{ms[c].mean():.3f} +- {ms[c].std(ddof=1):.3f}" if len(SEEDS) > 1 else f"{ms[c].mean():.3f}"
    rows.append(r)
    ms.assign(model=nm, seed=SEEDS).to_csv(f"{X}/metrics_{nm.replace(' ','_')}_by_seed.csv", index=False)
pd.DataFrame(rows).to_csv(f"{X}/extended_metrics_table.csv", index=False)
print(pd.DataFrame(rows).T.to_string())

# ---------- confusion matrices (seed 0 pooled OOF) ----------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
for ax, nm in zip(axs, ["XGBoost", "Fusion (full)"]):
    cm = confusion_matrix(y, models[nm][0].argmax(1), labels=[0, 1, 2])
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i,j]*100:.1f}%\n({cm[i,j]:,})", ha="center", va="center", fontsize=7.5,
                    color="white" if cmn[i, j] > 0.55 else "black")
    ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(LAB); ax.set_yticklabels(LAB)
    ax.set_xlabel("Predicted class (t + 15 min)"); ax.set_ylabel("True class")
    ax.set_title(f"{nm}: row-normalised confusion matrix"); ax.spines[:].set_visible(False)
    pd.DataFrame(cm, index=LAB, columns=LAB).to_csv(f"{X}/confusion_{nm.split()[0]}.csv")
plt.tight_layout(); plt.savefig(f"{FIG}/confusion.png", dpi=240); plt.close()

# ---------- attention (fusion) weights ----------
W = F[0][1]; cond = df["condition"].values; conds = sorted(np.unique(cond))
wmean = W.mean(0); print("mean attention weights", dict(zip(MODS, wmean.round(3))))
wm_by_seed = np.array([F[s][1].mean(0) for s in SEEDS])
pd.DataFrame(wm_by_seed, columns=MODS, index=[f"seed{s}" for s in SEEDS]).to_csv(f"{X}/attention_mean_by_seed.csv")
Hc = np.array([[W[cond == c, k].mean() for c in conds] for k in range(4)])
Hy = np.array([[W[y == c, k].mean() for c in range(3)] for k in range(4)])
pd.DataFrame(Hc, index=MODS, columns=[f"C{c}" for c in conds]).to_csv(f"{X}/attention_by_condition.csv")
pd.DataFrame(Hy, index=MODS, columns=LAB).to_csv(f"{X}/attention_by_class.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.5, 1]})
im = axs[0].imshow(Hc, cmap="viridis", aspect="auto"); axs[0].set_yticks(range(4))
axs[0].set_yticklabels(["Cardio-resp.", "Skin thermal", "Environment", "Individual"])
axs[0].set_xticks(range(len(conds))); axs[0].set_xticklabels([f"C{c}" for c in conds])
for i in range(4):
    for j in range(len(conds)):
        axs[0].text(j, i, f"{Hc[i,j]:.2f}", ha="center", va="center", color="white" if Hc[i, j] < Hc.max() * 0.75 else "black", fontsize=6.5)
axs[0].set_title("Mean attention weight by held-out scenario"); axs[0].spines[:].set_visible(False)
plt.colorbar(im, ax=axs[0], fraction=0.03, pad=0.02)
xx = np.arange(3); wd = 0.2
for k, nmod in enumerate(["Cardio-resp.", "Skin thermal", "Environment", "Individual"]):
    axs[1].bar(xx + (k - 1.5) * wd, Hy[k], wd, label=nmod)
axs[1].set_xticks(xx); axs[1].set_xticklabels(LAB); axs[1].set_xlabel("True risk class at t + 15 min")
axs[1].set_ylabel("Mean attention weight"); axs[1].axhline(0.25, color="gray", ls=":", lw=0.8)
axs[1].set_title("Attention weight by risk class"); axs[1].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28))
plt.tight_layout(); plt.savefig(f"{FIG}/attention.png", dpi=240); plt.close()

# ---------- forecast trajectories ----------
pf, px = models["Fusion (full)"][0], models["XGBoost"][0]
d = df[["participant", "condition", "t_min", "core_temp_now", "y"]].copy()
d["pf"] = pf.argmax(1); d["px"] = px.argmax(1)
d["pHigh"] = pf[:, 2]; d["pXHigh"] = px[:, 2]
trials = []
for (p, c), g in d.groupby(["participant", "condition"]):
    if g["y"].nunique() == 3:
        trials.append((p, c, (g["pf"] == g["y"]).mean(), len(g)))
tr = pd.DataFrame(trials, columns=["participant", "condition", "acc", "n"]).sort_values("acc").reset_index(drop=True)
print("trials with all 3 classes:", len(tr)); tr.to_csv(f"{X}/trajectory_candidate_trials.csv", index=False)
qs = [0.25, 0.5, 0.75]
pick = [tr.iloc[int(round(q * (len(tr) - 1)))] for q in qs]
fig = plt.figure(figsize=(7.2, 6.2)); gs = fig.add_gridspec(3, 1, hspace=0.55)
cmap = ListedColormap(COL)
for k, row in enumerate(pick):
    g = d[(d.participant == row.participant) & (d.condition == row.condition)].sort_values("t_min")
    sub = gs[k].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=0.12)
    a1 = fig.add_subplot(sub[0]); a2 = fig.add_subplot(sub[1], sharex=a1)
    a1.axhspan(g.core_temp_now.min() - 0.1, 38.0, color=COL[0], alpha=0.10); a1.axhspan(38.0, 38.5, color=COL[1], alpha=0.12)
    a1.axhspan(38.5, max(g.core_temp_now.max() + 0.1, 38.6), color=COL[2], alpha=0.10)
    a1.plot(g.t_min + 15, g.core_temp_now.shift(-0) if False else g.core_temp_now.values * 0 + np.nan, alpha=0)
    a1.plot(g.t_min, g.core_temp_now, color="black", lw=1.4, label="Measured $T_{re}$ (not a model input)")
    a1.axhline(38.0, color="gray", ls="--", lw=0.7); a1.axhline(38.5, color="gray", ls="--", lw=0.7)
    a1.set_ylabel("$T_{re}$ (°C)"); a1.tick_params(labelbottom=False)
    a1.set_title(f"Held-out trial: participant {int(row.participant)}, scenario C{int(row.condition)} "
                 f"(fusion accuracy {row.acc:.2f}, {int(row.n)} windows)", loc="left")
    pa = a1.twinx(); pa.plot(g.t_min, g.pHigh, color="#0fa3b1", lw=1.0, label="Fusion P(High at t+15)")
    pa.plot(g.t_min, g.pXHigh, color="#ff6b35", lw=1.0, ls="-", alpha=0.8, label="XGBoost P(High at t+15)")
    pa.set_ylim(0, 1); pa.set_ylabel("P(High)"); pa.spines["right"].set_visible(True)
    if k == 0:
        h1, l1 = a1.get_legend_handles_labels(); h2, l2 = pa.get_legend_handles_labels()
        a1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", ncol=1, fontsize=6.2)
    strip = np.vstack([g.y.values, g.px.values, g.pf.values])
    a2.imshow(strip, aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest",
              extent=[g.t_min.min(), g.t_min.max(), 3, 0])
    a2.set_yticks([0.5, 1.5, 2.5]); a2.set_yticklabels(["True", "XGBoost", "Fusion"], fontsize=6.5)
    a2.set_xlabel("Time at which the forecast is issued (min); each cell is the class 15 min later"); a2.spines[:].set_visible(False)
import matplotlib.patches as mp
fig.legend(handles=[mp.Patch(color=c, label=l) for c, l in zip(COL, LAB)], loc="lower center", ncol=3, frameon=False)
plt.savefig(f"{FIG}/trajectories.png", dpi=240, bbox_inches="tight"); plt.close()

# ---------- training curves ----------
H = F[0][2]  # (folds, epochs, 3): epoch, loss, train acc
ep = H[0, :, 0]; fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.0))
for a, j, lab in [(axs[0], 1, "Training loss"), (axs[1], 2, "Training accuracy")]:
    m, s = H[:, :, j].mean(0), H[:, :, j].std(0)
    a.plot(ep, m, color="#2e6da4"); a.fill_between(ep, m - s, m + s, color="#2e6da4", alpha=0.2)
    a.set_xlabel("Epoch"); a.set_title(lab, fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIG}/training_curves.png", dpi=240); plt.close()
print("loss first/last:", H[:, 0, 1].mean().round(3), H[:, -1, 1].mean().round(3), "train acc last:", H[:, -1, 2].mean().round(3))

# ---------- ROC / PR (High) / reliability ----------
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
for nm, pc, ls in [("XGBoost", px, "-"), ("Fusion (full)", pf, "--")]:
    fpr, tpr, _ = roc_curve(y == 2, pc[:, 2]); axs[0].plot(fpr, tpr, ls, label=f"{nm} (AUC {roc_auc_score(y==2, pc[:,2]):.2f})")
    pr_, rc_, _ = precision_recall_curve(y == 2, pc[:, 2]); axs[1].plot(rc_, pr_, ls, label=f"{nm} (AP {average_precision_score(y==2, pc[:,2]):.2f})")
    cx, ac, cf, cnt = reliability_curve(y, pc); m = ~np.isnan(ac); axs[2].plot(cf[m], ac[m], "o" + ls, ms=3, label=nm)
axs[0].plot([0, 1], [0, 1], ":", color="gray"); axs[0].set_xlabel("False-positive rate"); axs[0].set_ylabel("True-positive rate"); axs[0].set_title("ROC, High class")
axs[1].axhline((y == 2).mean(), ls=":", color="gray", label="prevalence"); axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision"); axs[1].set_title("Precision-recall, High class")
axs[2].plot([0, 1], [0, 1], ":", color="gray"); axs[2].set_xlabel("Mean confidence"); axs[2].set_ylabel("Accuracy"); axs[2].set_title("Reliability (pooled OOF, raw)")
for a in axs: a.legend(frameon=False)
plt.tight_layout(); plt.savefig(f"{FIG}/roc_pr_reliability.png", dpi=240); plt.close()

# ---------- real LOPO / LOCO / correlation figures ----------
lo = pd.read_csv(f"{R}/tables/lopo_per_participant.csv").sort_values("accuracy").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(3.4, 4.4))
ax.barh(np.arange(len(lo)), lo.accuracy, color=["#e8a33d" if a < lo.accuracy.mean() else "#2e9e5b" for a in lo.accuracy], height=0.8)
ax.set_yticks(np.arange(len(lo))); ax.set_yticklabels(lo.participant.astype(int), fontsize=5.5)
ax.axvline(lo.accuracy.mean(), color="black", ls="--", lw=0.9, label=f"mean {lo.accuracy.mean():.3f}"); ax.axvline(1/3, color="gray", ls=":", lw=0.9, label="uniform chance")
ax.set_xlabel("LOPO accuracy (XGBoost)"); ax.set_ylabel("Participant ID"); ax.legend(frameon=False, loc="lower right"); plt.tight_layout()
plt.savefig(f"{FIG}/lopo_real.png", dpi=240); plt.close()
r = np.corrcoef(lo.pct_high_windows, lo.accuracy)[0, 1]
fig, ax = plt.subplots(figsize=(3.4, 2.5)); ax.scatter(lo.pct_high_windows, lo.accuracy, s=14, color="#5b8bb0", edgecolor="white", lw=0.4)
ax.set_xlabel("Fraction of participant's windows labelled High"); ax.set_ylabel("LOPO accuracy"); ax.set_title(f"Per-participant accuracy vs. High prevalence (r = {r:.2f})", fontsize=7.5)
plt.tight_layout(); plt.savefig(f"{FIG}/scatter_real.png", dpi=240); plt.close()
lc = pd.read_csv(f"{R}/tables/loco_condition_results.csv")
pv = lc.pivot_table(index="condition", columns="model", values="accuracy"); xx = np.arange(len(pv)); w = 0.38
fig, ax = plt.subplots(figsize=(3.5, 2.5)); ax.bar(xx - w/2, pv["XGBoost"], w, label="XGBoost", color="#8c8c8c"); ax.bar(xx + w/2, pv["fusion_full"], w, label="Fusion (full)", color="#2e6da4")
ax.axhline(1/3, color="red", ls="--", lw=0.8, label="uniform chance"); ax.set_xticks(xx); ax.set_xticklabels([f"C{c}" for c in pv.index])
ax.set_ylabel("Accuracy (held-out scenario)"); ax.set_title("Leave-one-condition-out (real per-scenario values)", fontsize=7.5); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig(f"{FIG}/loco_real.png", dpi=240); plt.close()
pv.round(3).to_csv(f"{X}/loco_by_condition_pivot.csv"); print(pv.round(3)); print("r(High prev, acc) =", round(r, 3))
print("done")
