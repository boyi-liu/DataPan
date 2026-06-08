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

import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), os.pardir, "config.yaml")


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

    p.add_argument("--epochs", type=int, dest="train.epochs")
    p.add_argument("--batch-size", type=int, dest="train.batch_size")
    p.add_argument("--lr", type=float, dest="train.lr")

    # --- generic escape hatch for anything not exposed above ---
    p.add_argument("-o", "--override", action="append", default=[],
                   metavar="KEY.PATH=VALUE",
                   help="Override an arbitrary config key, e.g. -o lora.r=16")
    return p


def _add_method_args(parser, method):
    """Load ``alg/<method>.add_args`` and register its flags on ``parser``.

    Returns the set of ``dest`` names the method added, so the merge step knows
    to always apply them (their defaults live in the algorithm, not config.yaml).
    """
    try:
        module = importlib.import_module(f"alg.{method}")
    except ModuleNotFoundError:
        return set()  # unknown method; get_selector() will report it clearly
    add_args = getattr(module, "add_args", None)
    if add_args is None:
        return set()
    before = {id(a) for a in parser._actions}
    add_args(parser)
    return {a.dest for a in parser._actions if id(a) not in before}


def parse_args(argv=None):
    # Phase 1: sniff --config and --method without triggering help / validation.
    sniff = argparse.ArgumentParser(add_help=False)
    sniff.add_argument("--config", default=DEFAULT_CONFIG)
    sniff.add_argument("--method")
    pre, _ = sniff.parse_known_args(argv)
    cfg = load_config(pre.config)
    method = pre.method or cfg.selection.method

    # Phase 2: build the full parser and let the chosen algorithm add its flags.
    parser = build_parser()
    method_dests = _add_method_args(parser, method) if method else set()
    args = parser.parse_args(argv)

    # Apply curated + method flags. Method flags always apply (defaults live in
    # the algorithm); curated flags only when explicitly set (default None).
    for dest, value in vars(args).items():
        if dest in ("config", "override"):
            continue
        if dest in method_dests or value is not None:
            cfg.set_path(dest, value)

    # Generic overrides have the final say.
    for item in args.override:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Malformed override (expected KEY=VALUE): {item!r}")
        cfg.set_path(key.strip(), _coerce(raw))

    return cfg
