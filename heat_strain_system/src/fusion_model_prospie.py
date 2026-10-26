"""Adaptive Fusion Net adapted for PROSPIE's real physiological sensor
modalities. Same core architecture (attention-gated fusion + domain-
adversarial GRL + MC-dropout + random-restart guard) as the wearable-
dataset system, retargeted:
  - GRL now predicts PARTICIPANT ID (subject-invariance), same as before
  - modalities are real distinct physiological sensors, not proxy signals
  - label is real core-temperature heat-strain risk (not a proxy)
"""
from __future__ import annotations
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nn_engine import Tensor, Linear, Adam, cross_entropy_loss, focal_loss
from features_prospie import CARDIO_RESP, SKIN_THERMAL, ENVIRONMENT, INDIVIDUAL

# use current value + rolling mean + rolling slope for the continuous
# sensor groups (cardio_resp, skin_thermal); environment/individual use
# raw values (already slow-moving or static)
CARDIO_RESP_F = CARDIO_RESP + [f"{c}_roll_mean" for c in CARDIO_RESP] + [f"{c}_roll_slope" for c in CARDIO_RESP]
SKIN_THERMAL_F = SKIN_THERMAL + [f"{c}_roll_mean" for c in SKIN_THERMAL] + [f"{c}_roll_slope" for c in SKIN_THERMAL]
ENV_ROLL = ["ta_chamber", "rh_chamber", "wind_chamber", "t_front_cloth", "rh_front_cloth"]
ENVIRONMENT_F = ENVIRONMENT + [f"{c}_roll_mean" for c in ENV_ROLL] + [f"{c}_roll_slope" for c in ENV_ROLL]

MODALITIES = {
    "cardio_resp": CARDIO_RESP_F,
    "skin_thermal": SKIN_THERMAL_F,
    "environment": ENVIRONMENT_F,
    "individual": INDIVIDUAL,
}
ALL_MODEL_FEATURE_COLS = sorted(set(sum(MODALITIES.values(), [])))


class AdaptiveFusionNet:
    def __init__(self, modality_dims, n_classes=3, n_subjects=1, emb_dim=12,
                 hidden=24, dropout=0.2, seed=0, use_fusion_attention=True,
                 use_grl=True, use_modalities=None):
        self.rng = np.random.default_rng(seed)
        self.emb_dim = emb_dim
        self.n_classes = n_classes
        self.n_subjects = max(n_subjects, 1)
        self.dropout_p = dropout
        self.use_fusion_attention = use_fusion_attention
        self.use_grl = use_grl
        self.use_modalities = use_modalities or list(modality_dims.keys())
        self.modality_dims = {k: v for k, v in modality_dims.items() if k in self.use_modalities}
        self.n_parts = len(self.modality_dims)

        self.encoders = {}
        for name, d in self.modality_dims.items():
            self.encoders[name] = [Linear(d, hidden, self.rng), Linear(hidden, emb_dim, self.rng)]

        self.gate = Linear(self.n_parts * emb_dim, self.n_parts, self.rng)
        self.head1 = Linear(emb_dim, hidden, self.rng)
        self.head2 = Linear(hidden, n_classes, self.rng)
        self.dom1 = Linear(emb_dim, hidden, self.rng)
        self.dom2 = Linear(hidden, self.n_subjects, self.rng)

    def params(self):
        ps = []
        for enc in self.encoders.values():
            for lin in enc:
                ps += lin.params()
        ps += self.gate.params() + self.head1.params() + self.head2.params()
        if self.use_grl:
            ps += self.dom1.params() + self.dom2.params()
        return ps

    def _encode(self, x_dict, training):
        embs = []
        for name in self.modality_dims:
            x = Tensor(x_dict[name])
            h = self.encoders[name][0](x).relu().dropout(self.dropout_p, training, self.rng)
            e = self.encoders[name][1](h)
            embs.append(e)
        return embs

    def forward(self, x_dict, training=True, grl_lambda=1.0):
        embs = self._encode(x_dict, training)
        cat = embs[0].hcat(embs[1:]) if len(embs) > 1 else embs[0]

        if self.use_fusion_attention and self.n_parts > 1:
            gate_logits = self.gate(cat)
            weights = gate_logits.softmax()
            fused = cat.weighted_sum(weights, self.n_parts)
        else:
            N = cat.data.shape[0]
            weights = Tensor(np.full((N, self.n_parts), 1.0 / self.n_parts), requires_grad=False)
            fused = cat.weighted_sum(weights, self.n_parts)

        h = self.head1(fused).relu().dropout(self.dropout_p, training, self.rng)
        logits = self.head2(h)
        probs = logits.softmax()

        dom_probs = None
        if self.use_grl and training:
            rev = fused.grl(grl_lambda)
            dh = self.dom1(rev).relu()
            dlogits = self.dom2(dh)
            dom_probs = dlogits.softmax()

        return probs, dom_probs, weights, fused

    def predict_proba(self, x_dict, mc_samples=1):
        all_p = []
        for _ in range(mc_samples):
            probs, _, _, _ = self.forward(x_dict, training=(mc_samples > 1), grl_lambda=0.0)
            all_p.append(probs.data)
        arr = np.stack(all_p, axis=0)
        return arr.mean(axis=0), arr.std(axis=0)


def train_fusion_model(X_by_modality, y, subject_ids, n_classes=3, epochs=100,
                        lr=3e-3, batch_size=64, class_weights=None, grl_lambda=0.15,
                        use_focal=True, focal_gamma=1.0, seed=0,
                        use_fusion_attention=True, use_grl=True, use_modalities=None,
                        verbose=False, max_restarts=3):
    for attempt in range(max_restarts):
        model = _train_once(
            X_by_modality, y, subject_ids, n_classes=n_classes, epochs=epochs, lr=lr,
            batch_size=batch_size, class_weights=class_weights, grl_lambda=grl_lambda,
            use_focal=use_focal, focal_gamma=focal_gamma, seed=seed + attempt * 1000,
            use_fusion_attention=use_fusion_attention, use_grl=use_grl,
            use_modalities=use_modalities, verbose=verbose)
        train_probs, _ = model.predict_proba(X_by_modality, mc_samples=1)
        train_pred = train_probs.argmax(axis=1)
        n_unique_pred = len(np.unique(train_pred))
        train_acc = float(np.mean(train_pred == y))
        if n_unique_pred >= min(2, n_classes) and train_acc > 0.35:
            return model
        if verbose:
            print(f"    [restart] attempt {attempt} collapsed (train_acc={train_acc:.3f}) - retrying")
    return model


def _train_once(X_by_modality, y, subject_ids, n_classes=3, epochs=100,
                 lr=3e-3, batch_size=64, class_weights=None, grl_lambda=0.15,
                 use_focal=True, focal_gamma=1.0, seed=0,
                 use_fusion_attention=True, use_grl=True, use_modalities=None,
                 verbose=False):
    n = len(y)
    modality_dims = {k: v.shape[1] for k, v in X_by_modality.items()}
    subj_unique = {s: i for i, s in enumerate(sorted(set(subject_ids)))}
    subj_idx = np.array([subj_unique[s] for s in subject_ids])
    n_subjects = len(subj_unique)

    model = AdaptiveFusionNet(modality_dims, n_classes=n_classes, n_subjects=n_subjects,
                               seed=seed, use_fusion_attention=use_fusion_attention,
                               use_grl=use_grl, use_modalities=use_modalities)
    opt = Adam(model.params(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    y1h = np.eye(n_classes)[y]
    subj1h = np.eye(n_subjects)[subj_idx]
    alpha = class_weights / class_weights.sum() if class_weights is not None else None

    n_batches = max(1, n // batch_size)
    for epoch in range(epochs):
        lam = grl_lambda * min(1.0, epoch / max(1, epochs * 0.3))
        perm = rng.permutation(n)
        ep_loss = 0.0
        for bi in range(n_batches):
            idx = perm[bi * batch_size:(bi + 1) * batch_size]
            if len(idx) == 0:
                continue
            xb = {k: v[idx] for k, v in X_by_modality.items()}
            opt.zero_grad()
            probs, dom_probs, weights, fused = model.forward(xb, training=True, grl_lambda=lam)
            if use_focal:
                loss = focal_loss(probs, y1h[idx], gamma=focal_gamma, alpha=alpha)
            else:
                loss = cross_entropy_loss(probs, y1h[idx])
            total_loss = loss
            if model.use_grl and dom_probs is not None:
                dloss = cross_entropy_loss(dom_probs, subj1h[idx])
                total_loss = loss + 0.3 * dloss
            total_loss.backward()
            opt.step(clip_norm=5.0)
            ep_loss += total_loss.data
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:3d}  loss={ep_loss / n_batches:.4f}")
    return model
