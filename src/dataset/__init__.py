"""Dataset registry.

Each dataset lives in its own module ``dataset/load_<name>.py`` exposing a
``load(cfg, tokenizer) -> {"train": Dataset, "validation": Dataset | None}``
function. Selecting a dataset is just ``cfg.dataset.name``.
"""

import importlib


def get_dataset(cfg, tokenizer):
    name = cfg.dataset.name
    try:
        module = importlib.import_module(f"dataset.load_{name}")
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Unknown dataset {name!r}: expected a module dataset/load_{name}.py"
        ) from e
    if not hasattr(module, "load"):
        raise AttributeError(f"dataset/load_{name}.py must define a `load(cfg, tokenizer)` function")
    return module.load(cfg, tokenizer)
