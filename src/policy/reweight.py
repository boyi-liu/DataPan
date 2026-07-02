"""Soft per-sample reweighting -- the continuous end of the ④ spectrum.

Where :mod:`policy.hard` emits a ``{0, 1}`` keep/drop mask, this policy maps each
score through a temperature-scaled sigmoid gate to a *continuous* weight and
drops nothing (ADAPT, Eq. 11):

    w(s) = sigmoid( s / max(tau, eps) ),   clipped to [w_min, w_max]

Selection is the binary special case of this map, so ``hard`` and ``reweight``
are siblings behind the same :meth:`weights` primitive.

This is intended for the **online** path: the weights are per-sample
learning-rate multipliers applied to the batch loss each step (see
``alg/adapt.py``). Offline, since the generic Trainer can't consume soft
weights, ``select`` keeps the whole set (every sigmoid weight is > 0) -- i.e. the
weights only bite when applied to a loss.
"""

import numpy as np

from policy.base import BasePolicy


class Policy(BasePolicy):
    def __init__(self, cfg):
        super().__init__(cfg)
        sel = cfg.selection
        self.tau = float(sel.tau if sel.tau is not None else 1.0)
        self.eps = float(sel.eps if sel.eps is not None else 1e-8)
        self.w_min = float(sel.w_min if sel.w_min is not None else 0.0)
        self.w_max = float(sel.w_max if sel.w_max is not None else 10.0)

    def weights(self, scores, features=None):
        s = np.asarray(scores, dtype=np.float64)
        z = s / max(self.tau, self.eps)
        # Branch-stable logistic so large |z| doesn't overflow np.exp.
        w = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                     np.exp(z) / (1.0 + np.exp(z)))
        return np.clip(w, self.w_min, self.w_max)


def add_args(parser):
    """Register reweight-policy CLI flags (loaded dynamically by utils.options)."""
    g = parser.add_argument_group("reweight policy")
    g.add_argument("--reweight-tau", type=float, default=1.0, dest="selection.tau",
                   help="Sigmoid temperature; larger -> flatter weights.")
    g.add_argument("--reweight-w-min", type=float, default=0.0, dest="selection.w_min",
                   help="Lower clip on per-sample weights.")
    g.add_argument("--reweight-w-max", type=float, default=10.0, dest="selection.w_max",
                   help="Upper clip on per-sample weights (prevents LR explosion).")
