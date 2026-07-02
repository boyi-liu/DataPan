"""Coverage-maximizing selection -- an interaction-aware ④ policy.

Where :mod:`policy.hard` scores each example in isolation, this policy makes
*diversity an explicit objective* (the "submodular subset" row of the taxonomy):
it reasons about how the selected examples relate to one another so the kept set
is low-redundancy, not just high-scoring.

Two stages:

  1. **Quality gate.** Restrict to the top ``pool_mult * k`` examples by score,
     dropping unscorable (non-finite) ones. The per-sample signal (axes ①②③)
     still decides who is *eligible*.

  2. **Farthest-point (k-center) cover.** Greedily grow a ``k``-subset of the
     pool, seeding with the highest-scoring example and repeatedly adding the
     pool member *least* similar (cosine) to everything already chosen. This
     spreads the selection across the feature space instead of piling onto one
     high-scoring mode.

Needs per-example ``features`` (an ``(N, d)`` array), so it is usable today with
the representation-based selectors that already expose embeddings (``embedding``,
``miwv``). Score-only methods raise a clear error pointing at ``--policy hard``;
making features available everywhere is the job of the future Scorer-layer cut.

Cost: O(pool * k * d) time, O(pool) memory (similarity rows are streamed, never
the full pool x pool matrix). Tune the pool with ``--diversity-pool-mult``.
"""

import numpy as np

from policy.base import BasePolicy


class Policy(BasePolicy):
    needs_features = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.pool_mult = float(cfg.selection.diversity_pool_mult or 4.0)

    def weights(self, scores, features=None):
        if features is None:
            raise ValueError(
                "The 'diversity' policy needs per-example features, which only "
                "representation-based selectors currently expose (e.g. --method "
                "embedding or --method miwv). Use --policy hard for score-only "
                "methods."
            )
        scores = np.asarray(scores, dtype=np.float64)
        feats = np.asarray(features, dtype=np.float64)
        n = len(scores)
        w = np.zeros(n, dtype=np.float64)
        k = self._budget_to_k(n)
        if k >= n:
            return np.ones(n, dtype=np.float64)

        # Stage 1: quality gate -- top-P scoring examples, finite scores only.
        order = [int(i) for i in np.argsort(scores, kind="stable")[::-1]
                 if np.isfinite(scores[i])]
        pool_size = min(len(order), max(k, int(round(self.pool_mult * k))))
        pool = np.asarray(order[:pool_size], dtype=np.int64)
        if len(pool) <= k:                       # not enough to be picky
            w[pool[:k]] = 1.0
            return w

        # Stage 2: k-center greedy over the (unit-normalized) pool features.
        F = feats[pool]
        F = F / np.clip(np.linalg.norm(F, axis=1, keepdims=True), 1e-12, None)

        selected = [0]                           # pool is score-sorted: 0 == best
        max_sim = F @ F[0]                        # cosine of each pool item to the set
        max_sim[0] = np.inf                       # never re-pick a chosen item
        for _ in range(k - 1):
            nxt = int(np.argmin(max_sim))         # farthest from the current set
            selected.append(nxt)
            max_sim = np.maximum(max_sim, F @ F[nxt])
            max_sim[nxt] = np.inf
        w[pool[selected]] = 1.0
        return w


def add_args(parser):
    """Register diversity-policy CLI flags (loaded dynamically by utils.options)."""
    g = parser.add_argument_group("diversity policy")
    g.add_argument("--diversity-pool-mult", type=float, default=4.0,
                   dest="selection.diversity_pool_mult",
                   help="Candidate pool size as a multiple of the budget k: the "
                        "top pool_mult*k scoring examples are kept, then a "
                        "maximally-spread k-subset is greedily covered.")
