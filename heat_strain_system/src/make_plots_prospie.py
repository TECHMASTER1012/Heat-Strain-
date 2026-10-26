from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import LABELS

RESULTS_DIR = os.environ.get("HS_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")) + "/results"
PLOTS = f"{RESULTS_DIR}/plots"
os.makedirs(PLOTS, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 130})


def plot_label_distribution():
    df = pd.read_parquet(f"{RESULTS_DIR}/tables/prospie_features.parquet")
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["y"].value_counts().sort_index()
    ax.bar([LABELS[i] for i in counts.index], counts.values, color=["#27ae60", "#f39c12", "#c0392b"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, str(v), ha="center")
    ax.set_ylabel("# windows")
    ax.set_title(f"Real core-temp risk class distribution (n={len(df)})", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/00_label_distribution.png")
    plt.close()


def plot_model_comparison():
    summary = pd.read_csv(f"{RESULTS_DIR}/tables/model_summary.csv", index_col=0)
    order = ["LogReg", "RandomForest", "XGBoost", "ablation_cardio_only",
             "ablation_no_fusion", "ablation_no_grl", "fusion_full"]
    order = [o for o in order if o in summary.index]
    pretty = {"LogReg": "LogReg", "RandomForest": "Random\nForest", "XGBoost": "XGBoost",
              "ablation_cardio_only": "Fusion\n(cardiac only)", "ablation_no_fusion": "Fusion\n(no attention)",
              "ablation_no_grl": "Fusion\n(no GRL)", "fusion_full": "Fusion\n(full)"}
    colors = ["#7f8c8d"] * 3 + ["#5dade2"] * 3 + ["#1f618d"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    means = summary.loc[order, "macro_f1_mean"]
    stds = summary.loc[order, "macro_f1_std"]
    ax.bar([pretty[o] for o in order], means, yerr=stds, capsize=4, color=colors)
    ax.axhline(1 / 3, color="red", ls="--", lw=1, label="chance (3-class)")
    ax.set_ylabel("Macro-F1 (8-fold subject-independent CV)")
    ax.set_title("Model comparison on REAL core-temperature heat-strain labels")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/01_model_comparison_macrof1.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([pretty[o] for o in order], summary.loc[order, "recall_High"], color=colors)
    ax.set_ylabel("Recall on High-risk class (Tre >= 38.5C)")
    ax.set_title("High-risk (real core-temp) recall by model")
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/02_high_risk_recall.png")
    plt.close()


def plot_calibration():
    ece_summary = pd.read_csv(f"{RESULTS_DIR}/tables/calibration_summary.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(ece_summary))
    w = 0.35
    ax.bar(x - w / 2, ece_summary["ece_raw_mean"], width=w, label="raw", color="#e67e22")
    ax.bar(x + w / 2, ece_summary["ece_cal_mean"], width=w, label="temperature-scaled", color="#27ae60")
    ax.set_xticks(x); ax.set_xticklabels(ece_summary.index, rotation=30, ha="right")
    ax.set_ylabel("Expected Calibration Error (lower=better)")
    ax.set_title("Calibration before/after temperature scaling")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/03_ece_before_after.png")
    plt.close()


def plot_lopo():
    df = pd.read_csv(f"{RESULTS_DIR}/tables/lopo_per_participant.csv").sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ["#c0392b" if v < 0.5 else ("#f39c12" if v < 0.7 else "#27ae60") for v in df["accuracy"]]
    ax.barh(df["participant"].astype(str), df["accuracy"], color=colors)
    ax.axvline(df["accuracy"].mean(), color="black", ls="--", lw=1, label=f"mean={df['accuracy'].mean():.2f}")
    ax.axvline(1 / 3, color="red", ls=":", lw=1, label="chance")
    ax.set_xlabel("LOPO accuracy (XGBoost)")
    ax.set_ylabel("Participant ID")
    ax.set_title("True Leave-One-Participant-Out accuracy, all 40 participants")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/04_lopo_per_participant_accuracy.png")
    plt.close()


def plot_loco():
    df = pd.read_csv(f"{RESULTS_DIR}/tables/loco_condition_results.csv")
    piv = df.pivot(index="condition", columns="model", values="accuracy")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(piv))
    w = 0.35
    ax.bar(x - w / 2, piv["XGBoost"], width=w, label="XGBoost", color="#7f8c8d")
    ax.bar(x + w / 2, piv["fusion_full"], width=w, label="Fusion (full)", color="#1f618d")
    ax.axhline(1 / 3, color="red", ls="--", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels([f"C{c}" for c in piv.index])
    ax.set_ylabel("Accuracy (held-out condition)")
    ax.set_title("Leave-One-Condition-Out generalization\n(train on 8 scenarios, test on the 9th, unseen)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/05_loco_condition_generalization.png")
    plt.close()


def plot_variance_correlate():
    df = pd.read_csv(f"{RESULTS_DIR}/tables/lopo_per_participant.csv")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(df["pct_high_windows"], df["accuracy"], alpha=0.7, color="#1f618d")
    r = df[["pct_high_windows", "accuracy"]].corr().iloc[0, 1]
    ax.set_xlabel("Fraction of participant's windows labelled High-risk")
    ax.set_ylabel("LOPO accuracy")
    ax.set_title(f"Per-participant accuracy vs. High-risk prevalence (r={r:.2f})")
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/06_variance_correlate.png")
    plt.close()


if __name__ == "__main__":
    plot_label_distribution()
    plot_model_comparison()
    plot_calibration()
    plot_lopo()
    plot_loco()
    plot_variance_correlate()
    print("done")
