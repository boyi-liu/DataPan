"""Default selector -- compose any scorer with any policy from config alone.

This is the *modular dynamic-run* path and the **default method**: instead of
writing an ``alg/<name>.py``, pick a scorer and a policy and the pipeline wires
them together. Because it is the default, ``--method`` can be omitted entirely:

    python main.py --scorer bm25      --policy hard
    python main.py --scorer embedding --policy diversity

``--scorer`` chooses ``scorer/<name>.py`` (axes ①②③, via :func:`scorer.get_scorer`)
and ``--policy`` chooses ``policy/<name>.py`` (axis ④, via :func:`policy.get_policy`);
both plugins contribute their own CLI flags (see ``--scorer <s> --policy <p> --help``).
The selector itself is just "score, then apply the policy" offline -- identical to
the thin per-scorer selectors (``alg/bm25.py`` etc.), but with the scorer resolved
by name at runtime rather than hard-imported.

For anything finer-grained -- multiple scorers or custom scoring plumbing --
write a dedicated ``alg/<name>.py`` and select it with ``--method <name>`` (see
``alg/less.py``, ``alg/ifd.py``). Such a method wires
its own scorer *and* policy, so ``--scorer`` and ``--policy`` are both ignored
once ``--method`` is not ``default`` (the method's policy is its module-level
``DEFAULT_POLICY``, else ``hard``). ``--scorer``/``--policy`` only apply here.
"""

from alg.base import BaseSelector
from policy import get_policy
from scorer import get_scorer


class Selector(BaseSelector):
    def __init__(self, cfg, model=None, tokenizer=None):
        super().__init__(cfg, model, tokenizer)
        self.scorer = get_scorer(cfg, model, tokenizer)
        self.policy = get_policy(cfg)

    def select(self, train_dataset, val_dataset=None):
        scores, features = self.scorer.score(train_dataset, val_dataset)
        return self.apply_policy(scores, features=features)
