"""Out-of-fold analysis for the fusion network: stores per-window probabilities,
attention (fusion) weights and per-epoch training curves, using exactly the
same 8-fold GroupKFold split, hyper-parameters and seeds as evaluate_prospie.py."""
import os, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(__file__))
from common import class_weights_from, compute_metrics
from evaluate_prospie import prep_fold, RESULTS_DIR, SEED, N_SPLITS
from fusion_model_prospie import AdaptiveFusionNet, MODALITIES
from nn_engine import Adam, cross_entropy_loss, focal_loss

OUT = f"{RESULTS_DIR}/extra"
SEED_RUN = int(os.environ.get("HS_SEED", SEED))

def train_with_history(X, y, subj_ids, epochs=100, lr=3e-3, batch_size=64, cw=None, grl_lambda=0.15,
                       focal_gamma=1.0, seed=0, use_grl=True, use_attn=True, max_restarts=3):
    n = len(y); n_classes = 3
    dims = {k: v.shape[1] for k, v in X.items()}
    su = {s: i for i, s in enumerate(sorted(set(subj_ids)))}
    sidx = np.array([su[s] for s in subj_ids]); ns = len(su)
    y1h = np.eye(n_classes)[y]; s1h = np.eye(ns)[sidx]
    alpha = cw / cw.sum()
    for attempt in range(max_restarts):
        sd = seed + attempt * 1000
        model = AdaptiveFusionNet(dims, n_classes=n_classes, n_subjects=ns, seed=sd,
                                  use_fusion_attention=use_attn, use_grl=use_grl)
        opt = Adam(model.params(), lr=lr, weight_decay=1e-4)
        rng = np.random.default_rng(sd)
        hist = []
        nb = max(1, n // batch_size)
        for ep in range(epochs):
            lam = grl_lambda * min(1.0, ep / max(1, epochs * 0.3))
            perm = rng.permutation(n); tot = 0.0; ncor = 0; nseen = 0
            for bi in range(nb):
                idx = perm[bi*batch_size:(bi+1)*batch_size]
                if len(idx) == 0: continue
                xb = {k: v[idx] for k, v in X.items()}
                opt.zero_grad()
                probs, dom, w, fused = model.forward(xb, training=True, grl_lambda=lam)
                loss = focal_loss(probs, y1h[idx], gamma=focal_gamma, alpha=alpha)
                tl = loss
                if model.use_grl and dom is not None:
                    tl = loss + 0.3 * cross_entropy_loss(dom, s1h[idx])
                tl.backward(); opt.step(clip_norm=5.0); tot += tl.data; ncor += int((probs.data.argmax(1) == y[idx]).sum()); nseen += len(idx)
            hist.append((ep, tot / nb, ncor / max(nseen, 1)))
        p, _ = model.predict_proba(X, mc_samples=1)
        pred = p.argmax(1)
        if len(np.unique(pred)) >= 2 and np.mean(pred == y) > 0.35:
            return model, hist, attempt
    return model, hist, attempt

def attention_weights(model, Xd):
    _, _, w, _ = model.forward(Xd, training=False, grl_lambda=0.0)
    return w.data

if __name__ == "__main__":
    df = pd.read_parquet(f"{RESULTS_DIR}/tables/prospie_features.parquet")
    gkf = GroupKFold(n_splits=N_SPLITS)
    folds = [int(a) for a in sys.argv[1:]] or list(range(N_SPLITS))
    for fold_i, (tr, te) in enumerate(gkf.split(df, groups=df["participant"].values)):
        if fold_i not in folds: continue
        fn = f"{OUT}/fusion_s{SEED_RUN}_fold{fold_i}.npz"
        if os.path.exists(fn): print("skip", fold_i); continue
        t0 = time.time()
        train_df, test_df = df.iloc[tr].copy(), df.iloc[te].copy()
        train_df, test_df, Xf_tr, Xf_te, ytr, yte, Xtr, Xte = prep_fold(train_df, test_df)
        cw = class_weights_from(ytr)
        model, hist, restarts = train_with_history(Xtr, ytr, train_df["participant"].values, cw=cw, seed=SEED_RUN)
        p, _ = model.predict_proba(Xte, mc_samples=1)
        w_te = attention_weights(model, Xte)
        m = compute_metrics(yte, p.argmax(1), p)
        np.savez(fn, idx=te, proba=p, w=w_te, y=yte, hist=np.array(hist), restarts=restarts)
        print(f"fold {fold_i}: macroF1={m['macro_f1']:.3f} acc={m['accuracy']:.3f} restarts={restarts} {time.time()-t0:.0f}s", flush=True)
