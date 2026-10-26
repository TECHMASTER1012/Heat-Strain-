"""Fusion network: single-modality and leave-one-modality-out runs (same folds/hyper-parameters as the main CV)."""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import class_weights_from, compute_metrics
from evaluate_prospie import prep_fold, RESULTS_DIR
from fusion_model_prospie import train_fusion_model
SEED = int(os.environ.get("HS_SEED", 0)); OUT = f"{RESULTS_DIR}/extra"
df = pd.read_parquet(f"{RESULTS_DIR}/tables/prospie_features.parquet").reset_index(drop=True)
M = ["cardio_resp", "skin_thermal", "environment", "individual"]
configs = {f"only_{m}": [m] for m in M}
configs.update({f"without_{m}": [x for x in M if x != m] for m in M})
rows = []
for k, (tr, te) in enumerate(GroupKFold(8).split(df, groups=df["participant"].values)):
    trd, ted, Xtr, Xte, ytr, yte, Xtr_m, Xte_m = prep_fold(df.iloc[tr].copy(), df.iloc[te].copy())
    cw = class_weights_from(ytr)
    for name, mods in configs.items():
        mdl = train_fusion_model(Xtr_m, ytr, trd["participant"].values, epochs=100, lr=3e-3, batch_size=64, class_weights=cw,
                                 grl_lambda=0.15, use_focal=True, focal_gamma=1.0, seed=SEED, use_fusion_attention=True,
                                 use_grl=True, use_modalities=mods)
        p, _ = mdl.predict_proba(Xte_m, mc_samples=1)
        m = compute_metrics(yte, p.argmax(1), p); m.update(config=name, fold=k, seed=SEED); rows.append(m)
    pd.DataFrame(rows).to_csv(f"{OUT}/fusion_modality_runs_s{SEED}.csv", index=False)
    print("fold", k, flush=True)
