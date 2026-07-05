"""Argument parsing and configuration management.

Resolution order (lowest -> highest priority):
    1. defaults in ``config.yaml`` (or the file passed via ``--config``)
    2. method-specific defaults declared by the chosen algorithm's ``add_args``
       (in ``alg/<method>.py``), loaded dynamically based on ``--method``
    3. curated explicit CLI flags (``--model``, ``--lr``, ...)
    4. generic dotted overrides (``-o train.lr=1e-5``)

Each selection algorithm owns its hyper-parameters: it exposes an
``add_args(parser)`` function that registers CLI flags whose ``dest`` is the
dotted config path they populate (e.g. ``selection.warmup_steps``). This keeps
method-specific knobs out of the shared ``config.yaml``.

Usage:
    from utils.options import parse_args
    cfg = parse_args()
    print(cfg.model.name, cfg.selection.warmup_steps)
"""

import argparse
import importlib
import os
import warnings

import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), os.pardir, "config.yaml")

#: The modular selector (alg/default.py), used when no --method is given. It
#: composes a scorer + policy from config, so it needs a scorer to fall back to.
DEFAULT_METHOD = "default"
DEFAULT_SCORER = "bm25"
DEFAULT_BUDGET = 0.05


class Config(dict):
    """A dict with recursive attribute access (``cfg.train.lr``).

    Missing keys return ``None`` instead of raising, which keeps optional
    config fields ergonomic.
    """

    def __init__(self, data=None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = Config(value) if isinstance(value, dict) else value

    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted_key):
        node = self
        for key in dotted_key.split("."):
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return node

    def set_path(self, dotted_key, value):
        """Set a nested value addressed by a dotted path, e.g. ``train.lr``."""
        keys = dotted_key.split(".")
        node = self
        for key in keys[:-1]:
            if not isinstance(node.get(key), Config):
                node[key] = Config()
            node = node[key]
        node[keys[-1]] = value


def load_config(path):
    with open(path, "r") as f:
        return Config(yaml.safe_load(f) or {})


def _coerce(value):
    """Parse a CLI string into a typed value (int/float/bool/list/...)."""
    return yaml.safe_load(value)


def build_parser():
    p = argparse.ArgumentParser(
        description="LLM Data Curator",
        epilog="Method-specific flags are added based on --method; "
               "run with a method to see them, e.g. `--method less --help`.",
    )
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help="Path to the YAML config file.")

    # --- curated overrides: `dest` is the dotted path into the config ---
    p.add_argument("--seed", type=int, dest="seed")
    p.add_argument("--device", dest="device")
    p.add_argument("--output-dir", dest="output_dir")

    p.add_argument("--model", dest="model.name")
    p.add_argument("--max-length", type=int, dest="model.max_length")

    p.add_argument("--dataset", dest="dataset.name")
    p.add_argument("--data-dir", dest="dataset.data_dir")

    p.add_argument("--method", dest="selection.method")
    p.add_argument("--budget", type=float, dest="selection.budget")
    p.add_argument("--scorer", dest="selection.scorer")
    p.add_argument("--policy", dest="selection.policy")

    p.add_argument("--epochs", type=int, dest="train.epochs")
    p.add_argument("--batch-size", type=int, dest="train.batch_size")
    p.add_argument("--lr", type=float, dest="train.lr")

    # --- generic escape hatch for anything not exposed above ---
    p.add_argument("-o", "--override", action="append", default=[],
                   metavar="KEY.PATH=VALUE",
                   help="Override an arbitrary config key, e.g. -o lora.r=16")
    return p


def _add_plugin_args(parser, package, name):
    """Load ``<package>/<name>.add_args`` and register its flags on ``parser``.

    Shared by the selection algorithm (``alg/<method>.py``) and the selection
    policy (``policy/<name>.py``), which both own their hyper-parameters this
    way. Returns the set of ``dest`` names added, so the merge step knows to
    always apply them (their defaults live in the plugin, not config.yaml).
    """
    if not name:
        return set()
    try:
        module = importlib.import_module(f"{package}.{name}")
    except ModuleNotFoundError:
        return set()  # unknown plugin; get_selector()/get_policy() reports it clearly
    add_args = getattr(module, "add_args", None)
    if add_args is None:
        return set()
    before = {id(a) for a in parser._actions}
    add_args(parser)
    return {a.dest for a in parser._actions if id(a) not in before}


def _default_policy(method):
    """A method may declare ``DEFAULT_POLICY`` in ``alg/<method>.py`` (e.g. a
    method that pins 'diversity'), used when neither the CLI nor config.yaml
    pins ``--policy``."""
    if not method:
        return None
    try:
        module = importlib.import_module(f"alg.{method}")
    except ModuleNotFoundError:
        return None
    return getattr(module, "DEFAULT_POLICY", None)


def _apply_overrides(cfg, overrides):
    """Apply ``-o KEY.PATH=VALUE`` generic overrides (the final say)."""
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Malformed override (expected KEY=VALUE): {item!r}")
        cfg.set_path(key.strip(), _coerce(raw))


def parse_args(argv=None):
    """Resolve config for a run, always ending with a ``cfg.pipeline`` stage list.

    A run is a cascade of operators (see ``main.run_pipeline``). There are two
    ways to populate ``cfg.pipeline``:

      * **CLI shortcut** -- naming an operator (``--method``/``--scorer``/
        ``--policy``) or a ``--budget`` builds a *single-stage* pipeline from the
        fully-resolved ``selection.*`` config (so ``--warmup-steps`` etc. still
        work). This overrides any ``pipeline:`` in the config file.
      * **Config-driven** -- otherwise the config file's top-level ``pipeline:``
        list drives the run; per-stage knobs come from each stage dict (see
        ``main._stage_cfg``), not from method-specific CLI flags.
    """
    # Sniff the operator-defining flags without triggering help. Any of them ->
    # the CLI shortcut (a one-stage pipeline built from selection.*).
    sniff = argparse.ArgumentParser(add_help=False)
    sniff.add_argument("--config", default=DEFAULT_CONFIG)
    sniff.add_argument("--method")
    sniff.add_argument("--scorer")
    sniff.add_argument("--policy")
    sniff.add_argument("--budget", type=float)
    pre, _ = sniff.parse_known_args(argv)
    cfg = load_config(pre.config)
    if not isinstance(cfg.get("selection"), Config):
        cfg["selection"] = Config()  # internal per-stage namespace; not user-facing

    single = bool(pre.method or pre.scorer or pre.policy or pre.budget is not None)

    if not single:
        # ---- Config-driven pipeline: the file's `pipeline:` list drives it. ----
        parser = build_parser()
        args = parser.parse_args(argv)
        for dest, value in vars(args).items():
            if dest in ("config", "override") or value is None:
                continue
            cfg.set_path(dest, value)
        _apply_overrides(cfg, args.override)
        if not cfg.get("pipeline"):
            cfg["pipeline"] = [{"method": DEFAULT_METHOD, "budget": DEFAULT_BUDGET}]
        return cfg

    # ---- CLI shortcut: resolve one operator into selection.* (as a 1-stage). ----
    method = pre.method or DEFAULT_METHOD
    user_scorer = pre.scorer or cfg.get_path("selection.scorer")
    user_policy = pre.policy or cfg.get_path("selection.policy")

    if method == DEFAULT_METHOD:
        # The 'default' operator composes scorer + policy from config; both fall
        # back to built-ins so `--budget X` alone still runs.
        scorer = user_scorer or DEFAULT_SCORER
        policy = user_policy or "hard"
    else:
        # A custom method defines its own scorer *and* policy, so a pinned scorer
        # or policy is ignored -- warn rather than let it look applied.
        ignored = []
        if user_scorer:
            ignored.append(f"scorer={user_scorer!r}")
        if user_policy:
            ignored.append(f"policy={user_policy!r}")
        if ignored:
            warnings.warn(
                f"method={method!r} is a custom selector that defines its own "
                f"scorer and policy; the configured {', '.join(ignored)} will be "
                f"ignored. Use method='default' to compose a scorer with a policy.",
                stacklevel=2,
            )
        scorer = None  # custom methods hard-wire their own scorer
        policy = _default_policy(method) or "hard"

    cfg.set_path("selection.method", method)
    cfg.set_path("selection.scorer", scorer)
    cfg.set_path("selection.policy", policy)

    # Let the chosen algorithm + scorer + policy register their CLI flags
    # (defaults live in the plugin, so their dests always apply).
    parser = build_parser()
    plugin_dests = _add_plugin_args(parser, "alg", method)
    plugin_dests |= _add_plugin_args(parser, "scorer", scorer)
    plugin_dests |= _add_plugin_args(parser, "policy", policy)
    args = parser.parse_args(argv)

    # Plugin flags always apply; curated flags only when explicitly set. Skip
    # selection.scorer/policy -- already resolved above (a custom method
    # deliberately overrides what the user passed).
    for dest, value in vars(args).items():
        if dest in ("config", "override", "selection.scorer", "selection.policy"):
            continue
        if dest in plugin_dests or value is not None:
            cfg.set_path(dest, value)
    _apply_overrides(cfg, args.override)

    if cfg.get_path("selection.budget") is None:
        cfg.set_path("selection.budget", DEFAULT_BUDGET)
    # One-stage pipeline that reuses the fully-resolved base config as-is.
    cfg["pipeline"] = [{"_resolved": True}]
    return cfg
