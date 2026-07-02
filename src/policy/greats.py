"""GREATS greedy Taylor selection (NeurIPS 2024) -- an interaction-aware ④ policy.

Reference: Wang, Wu, Song, Mittal, Jia. "GREATS: Online Selection of High-Quality
Data for LLM Training in Every Iteration." https://github.com/Jiachen-T-Wang/GREATS

GREATS picks, from a candidate batch, the size-``k`` subset that maximizes the
*reduction in validation loss* of the resulting gradient step. That set utility
is approximated to second order (paper Eq. 4) and optimized greedily (Algorithm
1). This module is exactly that greedy optimizer, expressed as a policy over

  * per-sample scores   ``φ⁰_i = <g_i, g_val>``  -- the TracIN alignment of each
    training gradient with the (summed) validation gradient, and
  * per-sample features ``g_i``                  -- the training gradients.

Greedy loop (Algorithm 1, identity-Hessian variant):

    φ ← φ⁰
    repeat k times:
        z* ← argmax_{i not chosen} φ_i           # most useful remaining point
        choose z*
        φ_i ← φ_i − λ · <g_i, g_z*>   ∀ i        # Eq. 4 correction term

The correction discounts every remaining point by its gradient similarity to what
was just chosen, so the kept set is *aligned with the validation set and diverse*
(redundant/duplicate points are demoted after one of them is picked). This is the
'Optimization / submodular' row of axis ④ -- unlike ``hard`` top-k it models the
interactions between selected samples, which is the whole point of GREATS.

λ is the correction strength. Algorithm 1 uses η on the score and η² on the
correction; factoring the common η out of the argmax leaves λ = η (the learning
rate) on the correction. ``--greats-correction none`` drops the term entirely,
recovering plain TracIN top-k (a useful ablation / the 'no diversity' baseline).

Online GREATS (``alg/greats.py``) supplies the per-batch gradients each training
step; the policy itself is timing-agnostic and also runs offline given features.
The exact-Hessian variant (φ_i ← φ_i − η² g_i H_val g_z*) is omitted: it needs the
validation Hessian, and the paper's practical method uses the identity approx.
"""

import numpy as np

from policy.base import BasePolicy


class Policy(BasePolicy):
    needs_features = True

    def __init__(self, cfg):
        super().__init__(cfg)
        sel = cfg.selection
        self.correction = (sel.greats_correction or "identity").lower()
        # λ on the correction term == the learning rate η (see module docstring).
        strength = sel.greats_correction_strength
        if strength is None:
            strength = cfg.train.lr if (cfg.train and cfg.train.lr) else 1.0
        self.strength = float(strength)

    def weights(self, scores, features=None):
        phi = np.array(scores, dtype=np.float64)        # copy: the loop mutates it
        n = phi.shape[0]
        k = self._budget_to_k(n)

        if self.correction == "none":
            G = None                                    # plain TracIN top-k
        elif features is None:
            raise ValueError(
                "GREATS with correction needs per-example gradient features. "
                "alg/greats.py supplies them online; pass --greats-correction "
                "none for plain TracIN top-k without features."
            )
        else:
            G = np.asarray(features)                    # (n, d), keep dtype

        chosen = np.zeros(n, dtype=bool)
        for _ in range(k):
            masked = np.where(chosen, -np.inf, phi)
            zstar = int(np.argmax(masked))
            if not np.isfinite(masked[zstar]):
                break                                   # nothing left worth adding
            chosen[zstar] = True
            if G is not None:                           # Eq. 4 redundancy correction
                phi = phi - self.strength * (G @ G[zstar])

        w = np.zeros(n, dtype=np.float64)
        w[chosen] = 1.0
        return w


def add_args(parser):
    """Register GREATS-policy CLI flags (loaded dynamically by utils.options)."""
    g = parser.add_argument_group("GREATS policy")
    g.add_argument("--greats-correction", choices=["identity", "none"],
                   default="identity", dest="selection.greats_correction",
                   help="Redundancy correction after each greedy pick: 'identity' "
                        "(Eq. 4, Hessian≈I, diversity-aware) or 'none' (plain "
                        "TracIN top-k, no interaction).")
    g.add_argument("--greats-correction-strength", type=float, default=None,
                   dest="selection.greats_correction_strength",
                   help="λ weighting the −λ<g_i,g_z*> correction. Default: the "
                        "learning rate η (per Algorithm 1).")
