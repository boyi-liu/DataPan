"""Hard top-k filtering -- the ④ baseline (binary weights).

Rank by score, weight the highest ``k`` examples 1 and the rest 0. Per-sample
and diversity-blind; the derived ``select`` returns the top-k indices.
"""

import numpy as np

from policy.base import BasePolicy


class Policy(BasePolicy):
    def weights(self, scores, features=None):
        n = len(scores)
        k = self._budget_to_k(n)
        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
        w = np.zeros(n, dtype=np.float64)
        w[ranked[:k]] = 1.0
        return w
