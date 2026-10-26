import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import reliability_curve
R = os.environ.get("HS_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")) + "/results"; X = f"{R}/extra"; FIG = f"{X}/figs"
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.8, "axes.spines.top": False, "axes.spines.right": False})
lo = pd.read_csv(f"{R}/tables/lopo_per_participant.csv").sort_values("accuracy").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(3.4, 4.7))
ax.barh(np.arange(len(lo)), lo.accuracy, color=["#e8a33d" if a < lo.accuracy.mean() else "#2e9e5b" for a in lo.accuracy], height=0.8)
ax.set_yticks(np.arange(len(lo))); ax.set_yticklabels(lo.participant.astype(int), fontsize=5.5)
ax.axvline(lo.accuracy.mean(), color="black", ls="--", lw=0.9, label=f"mean accuracy {lo.accuracy.mean():.3f}")
ax.axvline(0.489, color="#b43b3b", ls="-.", lw=0.9, label="always-Low, pooled (0.489)"); ax.axvline(1/3, color="gray", ls=":", lw=0.9, label="uniform chance (0.333)")
ax.set_xlabel("LOPO accuracy (XGBoost)"); ax.set_ylabel("Participant ID")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=1); plt.tight_layout(); plt.savefig(f"{FIG}/lopo_real.png", dpi=240); plt.close()
lc = pd.read_csv(f"{R}/tables/loco_condition_results.csv"); pv = lc.pivot_table(index="condition", columns="model", values="accuracy"); xx = np.arange(len(pv)); w = 0.38
fig, ax = plt.subplots(figsize=(3.5, 2.8)); ax.bar(xx - w/2, pv["XGBoost"], w, label="XGBoost", color="#8c8c8c"); ax.bar(xx + w/2, pv["fusion_full"], w, label="Fusion (full)", color="#2e6da4")
ax.axhline(0.489, color="#b43b3b", ls="-.", lw=0.8, label="always-Low, pooled (0.489)"); ax.axhline(1/3, color="gray", ls=":", lw=0.8, label="uniform chance (0.333)")
ax.set_xticks(xx); ax.set_xticklabels([f"C{c}" for c in pv.index]); ax.set_ylabel("Accuracy (held-out scenario)")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2); plt.tight_layout(); plt.savefig(f"{FIG}/loco_real.png", dpi=240); plt.close()
# macro-F1 bars with legend outside
fig, ax = plt.subplots(figsize=(4.4, 3.4))
names = ["LogReg", "Random\nForest", "XGBoost", "Fusion\n(cardiac\nonly)", "Fusion\n(no\nattention)", "Fusion\n(no GRL)", "Fusion\n(full)"]
means = [0.673, 0.692, 0.719, 0.426, 0.660, 0.695, 0.674]; stds = [0.039, 0.040, 0.034, 0.086, 0.056, 0.046, 0.037]
ax.bar(names, means, yerr=stds, capsize=3, color=["#8c8c8c"]*3 + ["#7fb3d5"]*3 + ["#2e6da4"])
ax.axhline(0.319, color="red", ls="--", lw=0.9, label="uniform guess (0.319)"); ax.axhline(0.219, color="red", ls=":", lw=0.9, label="always-Low (0.219)")
ax.set_ylabel("Macro-F1 (mean $\\pm$ SD, 8 folds)"); ax.tick_params(axis="x", labelsize=6.3)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2); plt.tight_layout(); plt.savefig("/home/claude/paper/fig3_macroF1.png", dpi=220); plt.close()
# corrected calibration: reliability diagrams raw / temperature / vector, both models
z = np.load(f"{X}/calibration_fixed_probs.npz"); y = z["y"]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
for ax, m, ttl in zip(axs, ["XGBoost", "Fusion"], ["XGBoost", "Fusion (full)"]):
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=0.8)
    for k, c, l in [("raw", "#e8a33d", "raw"), ("temp", "#2e9e5b", "temperature scaling"), ("vec", "#2e6da4", "vector scaling")]:
        p = z[f"{m}_{k}"]; conf = p.max(1); corr = (p.argmax(1) == y).astype(float)
        bins = np.quantile(conf, np.linspace(0, 1, 9)); xs, ys = [], []
        for i in range(8):
            mk = (conf >= bins[i]) & (conf <= bins[i + 1]) if i == 7 else (conf >= bins[i]) & (conf < bins[i + 1])
            if mk.sum(): xs.append(conf[mk].mean()); ys.append(corr[mk].mean())
        ax.plot(xs, ys, marker="o", ms=3, lw=1, color=c, label=l)
    ax.set_title(ttl); ax.set_xlabel("Mean predicted confidence")
axs[0].set_ylabel("Observed accuracy")
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False); fig.tight_layout(rect=(0, 0.1, 1, 1))
plt.savefig(f"{FIG}/calibration_fixed.png", dpi=240); plt.close(); print("figs ok")
