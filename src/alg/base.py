"""Base interface shared by all data-selection algorithms."""

from abc import ABC, abstractmethod


class BaseSelector(ABC):
    """Selects a subset of the training data.

    A selector owns the *scoring* side of the taxonomy -- which signal defines
    data value (①), measured against what (②), with which model (③). Turning
    those scores into a concrete subset (the ④ Selection Policy axis) is
    delegated to a swappable :mod:`policy`. Each hand-written selector wires its
    policy up *explicitly* in ``__init__`` by importing the class it wants, just
    like it imports its scorer (``from policy.hard import Policy`` ->
    ``self.policy = Policy(cfg)``); the dependency is then visible at the call
    site. Score-based subclasses call :meth:`apply_policy` instead of
    re-implementing budget/top-k handling.

    (Only the config-driven ``default`` selector resolves its policy *by name*
    via :func:`policy.get_policy`; concrete methods name theirs directly so the
    ``--policy`` flag stays scoped to ``default``.)

    Subclasses (in ``alg/<method>.py``) implement :meth:`select` directly, so
    methods that don't fit a per-example scoring model (random, clustering,
    coreset, ...) are first-class and may ignore the policy.
    """

    def __init__(self, cfg, model=None, tokenizer=None):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def select(self, train_dataset, val_dataset=None):
        """Return a list of selected example indices into ``train_dataset``."""

    def make_trainer(self, cfg, model, tokenizer, train_dataset, val_dataset):
        """Optionally build a custom trainer for this method, else ``None``.

        Offline selectors return ``None`` and let the generic Trainer fine-tune
        on the selected subset. Online methods (e.g. ADAPT) override this to
        inject per-step reweighting into the optimization loop.
        """
        return None

    # ---- helpers for subclasses -------------------------------------------
    # Reads ``self.policy``, which each selector sets explicitly in its
    # ``__init__`` by importing the policy class it wants (the ``default``
    # selector is the only one that resolves it via ``get_policy(cfg)``).

    def apply_policy(self, scores, features=None):
        """Turn per-example ``scores`` into selected indices via the ④ policy.

        ``scores`` follow the convention *higher == more valuable* (selectors
        that prefer low values negate first; unscorable examples sink to
        ``-inf``). ``features`` are optional ``(N, d)`` per-example vectors that
        interaction-aware policies (e.g. ``diversity``) require and score-only
        policies ignore. Returns sorted indices.
        """
        return self.policy.select(scores, features=features)
