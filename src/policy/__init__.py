"""Selection-policy registry (the ④ axis).

Each policy lives in ``policy/<name>.py`` and defines a class named ``Policy``
(a :class:`~policy.base.BasePolicy` subclass). ``cfg.selection.policy`` picks
which one a selector uses to turn its scores into a subset; it defaults to
``hard`` (top-k) so existing methods behave exactly as before.

Mirror of :mod:`alg`'s registry, kept separate because *what to score* (the
selector) and *how scores become a subset* (the policy) vary independently.
"""

import importlib

from policy.base import BasePolicy

__all__ = ["BasePolicy", "get_policy"]


def get_policy(cfg):
    name = cfg.selection.policy or "hard"
    try:
        module = importlib.import_module(f"policy.{name}")
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Unknown selection policy {name!r}: expected a module policy/{name}.py"
        ) from e
    policy_cls = getattr(module, "Policy", None)
    if policy_cls is None:
        raise AttributeError(f"policy/{name}.py must define a `Policy` class")
    return policy_cls(cfg)
