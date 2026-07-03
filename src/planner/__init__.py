"""Pipeline planners — decide a run's cascade stages, statically or adaptively.

``main.run_pipeline`` asks a :class:`Planner` for the next stage given the run's
live state. ``build_planner(cfg)`` picks one from ``cfg.pipeline_planner.type``:
``list`` (default) replays the static ``pipeline:`` list; ``llm`` hands
orchestration to an LLM controller (:class:`planner.llm.LLMPlanner`).
"""

from planner.base import Planner, ListPlanner, build_planner

__all__ = ["Planner", "ListPlanner", "build_planner"]
