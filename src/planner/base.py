"""Pipeline planners: decide a run's cascade stages, statically or adaptively.

``main.run_pipeline`` no longer walks a fixed list; it asks a :class:`Planner`
for the next stage given the run's live state, applies it, and repeats until the
planner says stop. The default :class:`ListPlanner` simply replays the config's
``pipeline:`` list in order -- so a config- or CLI-driven run behaves exactly as
before -- but the same loop can host an adaptive planner (e.g. an LLM controller)
that *chooses* each stage from the current survivor count, budget and history.

A "stage" is the same dict schema as a ``cfg.pipeline`` entry (see
``main._stage_cfg`` / config.yaml's ``pipeline:`` docs): method/scorer/policy,
model/reference/budget, plus any method hyper-parameter key. The special
``{"_resolved": True}`` stage means "run the fully-resolved base config as-is".
"""


class Planner:
    """Yields the next pipeline stage dict, or ``None`` to stop the cascade.

    ``next_stage(state)`` is called once per step. ``state`` is a dict with the
    run's *structural* signals (no per-step eval):

        original_size  -- len(train_set)
        current_size   -- survivors remaining after the prior stages
        kept_fraction  -- current_size / original_size (~ budget already spent)
        history        -- the per-stage log so far (see ``run_pipeline``)

    It returns a stage dict (a ``cfg.pipeline`` entry) or ``None`` to finish.
    """

    #: Total stage count if known up front (progress display only), else ``None``.
    total = None

    def next_stage(self, state):
        raise NotImplementedError


class ListPlanner(Planner):
    """Replay a static ``pipeline:`` list -- the original, fixed behaviour."""

    def __init__(self, stages):
        self.stages = list(stages)
        self.total = len(self.stages)
        self._i = 0

    def next_stage(self, state):
        if self._i >= len(self.stages):
            return None
        stage = self.stages[self._i]
        self._i += 1
        return stage


def build_planner(cfg):
    """Pick a planner from ``cfg.pipeline_planner.type`` (default: ``list``).

    ``list`` (or unset) replays ``cfg.pipeline`` exactly as before. ``llm`` hands
    orchestration to an LLM controller (see ``planner.llm.LLMPlanner``).
    """
    spec = cfg.get("pipeline_planner")
    kind = (spec.get("type") if isinstance(spec, dict) else None) or "list"
    if kind == "list":
        return ListPlanner(cfg.pipeline or [{"method": "default"}])
    if kind == "llm":
        from planner.llm import LLMPlanner
        return LLMPlanner(cfg)
    raise ValueError(
        f"Unknown pipeline_planner.type {kind!r} (expected 'list' or 'llm')"
    )
