"""Perplexity baseline -- offline top-k over response perplexity.

A thin selector: the scoring (perplexity the frozen model assigns to each
response, axes ①②③) lives in :mod:`scorer.ppl`; here it is just "score, then
apply the ④ policy" offline. See ``scorer/ppl.py`` for the method and its knobs.
Enable with ``--method ppl``.
"""

from alg.base import BaseSelector
from policy.hard import Policy  # ④ policy loaded directly (get_policy is for `default`)
from scorer.ppl import Scorer, add_args  # add_args re-exported for utils.options


class Selector(BaseSelector):
    def __init__(self, cfg, model=None, tokenizer=None):
        super().__init__(cfg, model, tokenizer)
        self.scorer = Scorer(cfg, model, tokenizer)
        self.policy = Policy(cfg)

    def select(self, train_dataset, val_dataset=None):
        scores, features = self.scorer.score(train_dataset, val_dataset)
        return self.apply_policy(scores, features=features)
