"""Scorer registry: per-example value scoring (the fused ①②③ axes).

Each scorer lives in ``scorer/<name>.py`` and defines a class named ``Scorer``
(a :class:`~scorer.base.BaseScorer` subclass) that maps a dataset to per-example
value scores via ``score(train, val=None) -> (scores, features)``.

Two ways to reach a scorer, mirroring how :mod:`alg` and :mod:`policy` work:

  * **By name** -- :func:`get_scorer` imports ``scorer/<name>.py`` from
    ``cfg.selection.scorer``, exactly like :func:`policy.get_policy`. This powers
    the *default* selector (``alg/default.py``): pick a scorer and a policy in the
    config and they are wired together dynamically, no glue code.
  * **By import** -- a custom ``alg/<name>.py`` can still load a scorer
    *explicitly* (``from scorer.bm25 import Scorer``) when it needs finer control
    than name + policy (multiple scorers, custom plumbing, a bespoke trainer).

"What defines value" (the scorer), "how scores become a subset" (the ④ policy),
and "when it runs" (the ⑤ timing) still vary independently -- the registry just
makes the common "score, then apply a policy" case configuration-only.
"""

import importlib

from scorer.base import BaseScorer

__all__ = ["BaseScorer", "get_scorer"]


def get_scorer(cfg, model=None, tokenizer=None):
    """Instantiate the scorer named by ``cfg.selection.scorer``.

    Mirror of :func:`policy.get_policy`: ``scorer/<name>.py`` must define a
    ``Scorer`` class. ``model``/``tokenizer`` are forwarded so model-based
    scorers (③) get what they need; model-free scorers ignore them.
    """
    name = cfg.selection.scorer
    if not name:
        raise ValueError(
            "No scorer configured: set `selection.scorer` (or pass --scorer) to a "
            "module scorer/<name>.py when using the modular selector."
        )
    try:
        module = importlib.import_module(f"scorer.{name}")
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Unknown scorer {name!r}: expected a module scorer/{name}.py"
        ) from e
    scorer_cls = getattr(module, "Scorer", None)
    if scorer_cls is None:
        raise AttributeError(f"scorer/{name}.py must define a `Scorer` class")
    return scorer_cls(cfg, model, tokenizer)
