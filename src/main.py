"""Entry point for the data-selection pipeline.

    python main.py --dataset a --method less --budget 0.05
    python main.py --method random --no-train          # baseline, selection only

Flow:
    parse args -> load model+tokenizer -> load dataset -> score & select
               -> (optionally) fine-tune on the selected subset
"""

import argparse
import json
import os

from dataset import get_dataset
from selector import get_selector
from utils.model_utils import load_model_and_tokenizer
from utils.options import parse_args
from utils.train_utils import set_seed, train as run_training


def _save_selection(cfg, indices):
    os.makedirs(cfg.output_dir, exist_ok=True)
    path = os.path.join(cfg.output_dir, "selection.json")
    with open(path, "w") as f:
        json.dump(
            {
                "method": cfg.selection.method,
                "budget": cfg.selection.budget,
                "num_selected": len(indices),
                "selected_indices": indices,
            },
            f,
        )
    return path


def main():
    # A tiny extra flag layered on top of the config-driven options.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--no-train", action="store_true",
                     help="Run selection only; skip the final fine-tuning.")
    known, remaining = pre.parse_known_args()

    cfg = parse_args(remaining)
    set_seed(cfg.seed)

    print(f"[1/4] Loading model & tokenizer: {cfg.model.name}")
    model, tokenizer = load_model_and_tokenizer(cfg)

    print(f"[2/4] Loading dataset: {cfg.dataset.name}")
    data = get_dataset(cfg, tokenizer)
    train_set, val_set = data["train"], data.get("validation")
    print(f"      train={len(train_set)} | "
          f"validation={len(val_set) if val_set is not None else 0}")

    print(f"[3/4] Selecting data with method: {cfg.selection.method}")
    selector = get_selector(cfg, model, tokenizer)
    indices = selector.select(train_set, val_set)
    selected = train_set.select(indices)
    out = _save_selection(cfg, indices)
    print(f"      kept {len(indices)}/{len(train_set)} examples -> {out}")

    if known.no_train:
        print("[4/4] --no-train set; done.")
        return

    print(f"[4/4] Fine-tuning on {len(selected)} selected examples")
    run_training(cfg, model, tokenizer, selected, val_set)
    print(f"Done. Artifacts in {cfg.output_dir}")


if __name__ == "__main__":
    main()
