"""Selector registry.

Each selection algorithm lives in ``alg/<method>.py`` and exposes a
``SELECTOR`` attribute pointing at a :class:`~selector.base.BaseSelector`
subclass. ``cfg.selection.method`` picks which one to use.
"""

import importlib

from selector.base import BaseSelector

__all__ = ["BaseSelector", "get_selector"]


def get_selector(cfg, model=None, tokenizer=None):
    method = cfg.selection.method
    try:
        module = importlib.import_module(f"alg.{method}")
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Unknown selection method {method!r}: expected a module alg/{method}.py"
        ) from e
    selector_cls = getattr(module, "SELECTOR", None)
    if selector_cls is None:
        raise AttributeError(f"alg/{method}.py must define a `SELECTOR` class")
    return selector_cls(cfg, model, tokenizer)
