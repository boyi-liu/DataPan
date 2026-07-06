"""Entry point for the data-selection pipeline.

The three phases -- ``select``, ``train``, ``eval`` -- are fully decoupled and
run ONE AT A TIME via ``--stage``. Each stage force-saves its output at the end
and force-loads the previous stage's output at the start, so a full run is three
separate invocations that hand artifacts to each other through ``--output-dir``:

    python main.py --stage select --dataset alpaca --method less --budget 0.05
    python main.py --stage train                                   # trains the saved subset
    python main.py --stage eval  --benchmark gsm8k --eval-limit 200

Flow per stage:
    select : load model+tokenizer -> load dataset -> run operator cascade
             -> save selected_dataset/ + val_dataset/ + selection.json
    train  : load selected_dataset/ -> fine-tune -> save the model checkpoint
    eval   : load the fine-tuned model -> run a benchmark -> save eval.json

Because the stages share nothing but on-disk artifacts, point every invocation
of one run at the same ``--output-dir`` (and pass the same ``--config``/method
flags, which stay the source of truth for configuration).
"""

import argparse
import importlib
import json
import os
import warnings

from dataset import get_dataset
from alg import get_selector
from planner import build_planner
from utils.model_utils import load_model, load_tokenizer, load_model_and_tokenizer
from utils.options import Config, parse_args, _default_policy
from utils.train_utils import set_seed, train as run_training

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

def _save_json(cfg, name, payload):
    os.makedirs(cfg.output_dir, exist_ok=True)
    path = os.path.join(cfg.output_dir, name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Operator orchestration
# ---------------------------------------------------------------------------
# A run is a *cascade of operators*: every ``alg/<method>.py`` (and every
# ``default`` scorer+policy pairing) is one operator, and a run chains them so
# each stage filters the previous stage's survivors. A single method is just a
# one-stage pipeline -- so there is only this one code path.
#
# The chain lives in ``cfg.pipeline`` (a list of stage dicts), populated by
# ``utils.options.parse_args`` from either the config file's top-level
# ``pipeline:`` list or a CLI shortcut. Each stage overlays the base config for
# that operator only:
#   method/scorer/policy -- which operator (omit method -> `default`)
#   model                -- model.name for this stage (loaded lazily & cached)
#   reference            -- dataset name used as this stage's val_dataset (axis ②)
#   budget               -- fraction/count kept from the CURRENT survivors (cascade)
#   <any other key>      -- a method knob, shorthand for selection.<key>
# The special stage ``{"_resolved": True}`` means "run the fully-resolved base
# config as-is" -- what the CLI single-method shortcut produces.


def _clone_cfg(cfg):
    """Deep-copy a :class:`Config` (``copy.deepcopy`` chokes on its ``__getattr__``)."""
    def plain(node):
        if isinstance(node, dict):
            return {k: plain(v) for k, v in node.items()}
        if isinstance(node, list):
            return [plain(v) for v in node]
        return node
    return Config(plain(cfg))


def _plugin_defaults(package, name):
    """Return ``{dotted_dest: default}`` declared by ``<package>/<name>.add_args``.

    Mirrors ``utils.options._add_plugin_args`` off a throwaway parser, so a stage
    inherits the same defaults it would get from the CLI (e.g. LESS's
    ``warmup_steps=200``) without the YAML repeating them.
    """
    if not name:
        return {}
    try:
        module = importlib.import_module(f"{package}.{name}")
    except ModuleNotFoundError:
        return {}  # unknown plugin; get_selector() reports it clearly downstream
    add_args = getattr(module, "add_args", None)
    if add_args is None:
        return {}
    parser = argparse.ArgumentParser(add_help=False)
    add_args(parser)
    return vars(parser.parse_args([]))


#: Structural stage keys handled explicitly. Any *other* key in a stage dict is
#: shorthand for a ``selection.<key>`` method hyper-parameter (see ``_stage_cfg``).
_STAGE_FIELDS = frozenset({
    "name", "method", "scorer", "policy", "model", "reference", "budget",
    "_resolved",
})


def _stage_cfg(base_cfg, stage):
    """Build a per-stage config: base config + method defaults + stage settings.

    ``scorer`` and ``policy`` only compose the ``default`` operator, so they take
    effect only for ``method: default``. A concrete method wires its own scorer
    *and* policy in code, so a stage's ``scorer``/``policy`` are ignored (with a
    warning) and its policy is forced to the method's ``DEFAULT_POLICY``. The
    orthogonal ``model``/``reference``/``budget`` stay per-stage for every method.

    Any key that isn't a structural field (see ``_STAGE_FIELDS``) is shorthand for
    a ``selection.<key>`` method hyper-parameter, so ``warmup_steps: 100`` in a
    stage is exactly ``--warmup-steps 100`` on the CLI.
    """
    method = stage.get("method") or "default"
    cfg = _clone_cfg(base_cfg)
    if method == "default":
        # `default` composes a scorer+policy; get_scorer has no fallback, so a
        # scorer name is required (-> bm25), and get_policy defaults to hard.
        scorer = stage.get("scorer") or "bm25"
        policy = stage.get("policy") or "hard"
    else:
        ignored = []
        if stage.get("scorer"):
            ignored.append(f"scorer={stage['scorer']!r}")
        if stage.get("policy"):
            ignored.append(f"policy={stage['policy']!r}")
        if ignored:
            warnings.warn(
                f"pipeline stage method={method!r} is a custom selector that wires "
                f"its own scorer and policy; the stage's {', '.join(ignored)} will "
                f"be ignored (use method='default' to compose a scorer with a policy).",
                stacklevel=2,
            )
        scorer = None  # custom methods hard-wire their own scorer
        policy = _default_policy(method) or "hard"  # method's DEFAULT_POLICY, else hard

    # 1) Method/scorer/policy defaults, so a stage's YAML lists only what differs.
    defaults = _plugin_defaults("alg", method)
    if method == "default":
        defaults.update(_plugin_defaults("scorer", scorer))
    defaults.update(_plugin_defaults("policy", policy))
    for dest, value in defaults.items():
        cfg.set_path(dest, value)

    # 2) The operator selection knobs.
    cfg.set_path("selection.method", method)
    cfg.set_path("selection.scorer", scorer)
    cfg.set_path("selection.policy", policy)
    if stage.get("model"):
        cfg.set_path("model.name", stage["model"])
    if stage.get("budget") is not None:
        cfg.set_path("selection.budget", stage["budget"])

    # 3) Method hyper-parameters: any non-structural key -> selection.<key>
    #    (e.g. `warmup_steps: 100` == `--warmup-steps 100`). Applied after the
    #    plugin defaults from step 1, so the stage value wins.
    for key, value in stage.items():
        if key not in _STAGE_FIELDS:
            cfg.set_path(f"selection.{key}", value)
    return cfg


def run_pipeline(cfg, model, tokenizer, train_set, val_set):
    """Run the operator cascade; return (indices, last_selector, log).

    Stages come from a :class:`planner.Planner` (default: replay ``cfg.pipeline``;
    ``pipeline_planner.type: llm`` lets an LLM choose each stage from live state).
    ``indices`` are into the original ``train_set``. ``last_selector`` is the
    final stage's operator.
    """
    planner = build_planner(cfg)
    top_name = cfg.model.name
    # Model cache: the top-level model stays resident (final training reuses it);
    # at most one *extra* model is held at a time so long chains don't OOM.
    model_cache = {top_name: (model, tokenizer)}
    ref_cache = {}

    def resolve_model(name):
        if not name or name == top_name:
            return model_cache[top_name]
        if name in model_cache:
            return model_cache[name]
        for key in [k for k in model_cache if k != top_name]:
            del model_cache[key]
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        stage_cfg = _clone_cfg(cfg)
        stage_cfg.set_path("model.name", name)
        loaded = load_model_and_tokenizer(stage_cfg)
        model_cache[name] = loaded
        return loaded

    def resolve_reference(name, tokenizer_):
        if not name:
            return val_set
        if name not in ref_cache:
            ref_cfg = _clone_cfg(cfg)
            ref_cfg.set_path("dataset.name", name)
            data = get_dataset(ref_cfg, tokenizer_)
            ref_cache[name] = data.get("validation") or data["train"]
        return ref_cache[name]

    original_size = len(train_set)
    current = train_set
    survivors = list(range(original_size))  # positions in current -> original idx
    last_selector = None
    log = []
    stage_num = 0
    while True:
        state = {
            "original_size": original_size,
            "current_size": len(survivors),
            "kept_fraction": len(survivors) / original_size if original_size else 0.0,
            "history": log,
        }
        stage = planner.next_stage(state)
        if stage is None:
            break
        stage_num += 1
        if stage.get("_resolved"):
            stage_cfg, model_i, tok_i, ref = cfg, model, tokenizer, val_set
            name = cfg.selection.method
        else:
            stage_cfg = _stage_cfg(cfg, stage)
            model_i, tok_i = resolve_model(stage.get("model"))
            ref = resolve_reference(stage.get("reference"), tok_i)
            name = (stage.get("name") or stage.get("method")
                    or stage.get("scorer") or "default")

        selector = get_selector(stage_cfg, model_i, tok_i)
        local = selector.select(current, ref)
        survivors = [survivors[j] for j in local]
        current = current.select(local)
        last_selector = selector
        log.append({"name": name, "method": stage_cfg.selection.method,
                    "budget": stage_cfg.selection.budget, "kept": len(survivors)})
        pos = f"{stage_num}/{planner.total}" if planner.total else f"{stage_num}"
        print(f"      [{pos}] {name}: kept {len(survivors)} examples")

    return sorted(survivors), last_selector, log


# ---------------------------------------------------------------------------
# Stages: select / train / eval, each a standalone step
# ---------------------------------------------------------------------------
# The three phases are fully decoupled and run one at a time (``--stage``). They
# communicate ONLY through artifacts under ``cfg.output_dir`` -- each stage
# force-saves its output at the end and force-loads the prior stage's output at
# the start, so a run is three separate invocations:
#
#     python main.py --stage select --method less --budget 0.05
#     python main.py --stage train
#     python main.py --stage eval  --benchmark gsm8k
#
# Artifact layout under ``output_dir``:
#   selection.json      -- provenance: pipeline log, indices, final method   (select ->)
#   selected_dataset/   -- the selected training subset (HF save_to_disk)    (select -> train)
#   val_dataset/        -- validation/anchor split, if any                   (select -> train)
#   <model files>       -- fine-tuned adapter/checkpoint at the dir root     (train  -> eval)
#   eval.json           -- benchmark metrics                                 (eval  ->)
STAGES = ("select", "train", "eval")


def _selected_dir(cfg):
    return os.path.join(cfg.output_dir, "selected_dataset")


def _val_dir(cfg):
    return os.path.join(cfg.output_dir, "val_dataset")


def _save_dataset(dataset, path):
    """Force-save an HF dataset to ``path`` (replacing any prior copy)."""
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)
    dataset.save_to_disk(path)
    return path


def _resolve_trained_model(cfg):
    """Locate the model produced by the ``train`` stage in ``cfg.output_dir``.

    Handles both save layouts of ``train_utils.train`` and returns
    ``(model_path, adapter_dir)``:

    * LoRA adapter (``adapter_config.json``) -> ``(base model name, adapter dir)``
    * full checkpoint (``config.json`` + weights) -> ``(checkpoint dir, None)``

    Shared by the HF loader (``_load_trained_model``) and the vLLM loader so the
    disk-layout logic lives in exactly one place.
    """
    out = cfg.output_dir
    if os.path.exists(os.path.join(out, "adapter_config.json")):
        return cfg.model.name, out
    has_weights = any(
        os.path.exists(os.path.join(out, f))
        for f in ("model.safetensors", "pytorch_model.bin",
                  "model.safetensors.index.json", "pytorch_model.bin.index.json")
    )
    if os.path.exists(os.path.join(out, "config.json")) and has_weights:
        return out, None
    raise SystemExit(
        f"'eval' stage found no trained model in {out} (no adapter_config.json or "
        f"model weights). Run `--stage train` first, or point --output-dir at a run.")


def _load_trained_model(cfg):
    """Load a previously fine-tuned model from ``cfg.output_dir`` for HF eval."""
    model_path, adapter_dir = _resolve_trained_model(cfg)
    if adapter_dir:
        from peft import PeftModel
        base = load_model(cfg)
        model = PeftModel.from_pretrained(base, adapter_dir)
        return model.to(cfg.device)
    full_cfg = _clone_cfg(cfg)
    full_cfg.set_path("model.name", model_path)
    return load_model(full_cfg)


# ---------------------------------------------------------------------------
# The three stages
# ---------------------------------------------------------------------------
def stage_select(cfg):
    """Read the dataset, run the operator cascade, and save the selected subset."""
    print(f"[select] loading model & tokenizer: {cfg.model.name}")
    model, tokenizer = load_model(cfg), load_tokenizer(cfg)

    print(f"[select] loading dataset: {cfg.dataset.name}")
    data = get_dataset(cfg, tokenizer)
    train_set, val_set = data["train"], data.get("validation")
    print(f"         train={len(train_set)} | "
          f"validation={len(val_set) if val_set is not None else 0}")

    planner_type = (cfg.get("pipeline_planner") or {}).get("type") or "list"
    if planner_type == "list":
        print(f"[select] {len(cfg.pipeline or [])}-stage pipeline")
    else:
        print(f"[select] {planner_type}-planned pipeline (adaptive)")
    indices, _last_selector, log = run_pipeline(cfg, model, tokenizer, train_set, val_set)
    selected = train_set.select(indices)

    # Force-save this stage's output: the subset (and val/anchor split) the train
    # stage will consume, plus a provenance record naming the final operator.
    os.makedirs(cfg.output_dir, exist_ok=True)
    _save_dataset(selected, _selected_dir(cfg))
    if val_set is not None:
        _save_dataset(val_set, _val_dir(cfg))
    out = _save_json(cfg, "selection.json", {
        "pipeline": log,
        "num_selected": len(indices),
        "selected_indices": indices,
        "method": log[-1]["method"] if log else None,
    })
    print(f"      kept {len(indices)}/{len(train_set)} examples")
    print(f"      saved subset -> {_selected_dir(cfg)}  (provenance -> {out})")


def stage_train(cfg):
    """Load the selected subset from the ``select`` stage, fine-tune, save the model."""
    from datasets import load_from_disk

    sel_dir = _selected_dir(cfg)
    if not os.path.exists(sel_dir):
        raise SystemExit(
            f"'train' stage needs a selection but {sel_dir} was not found. "
            f"Run `--stage select` first (writing to the same --output-dir).")
    selected = load_from_disk(sel_dir)
    val_set = load_from_disk(_val_dir(cfg)) if os.path.exists(_val_dir(cfg)) else None
    print(f"[train] loaded {len(selected)} selected examples from {sel_dir}")

    print(f"[load] model & tokenizer: {cfg.model.name}")
    model, tokenizer = load_model(cfg), load_tokenizer(cfg)

    print(f"[train] fine-tuning on {len(selected)} selected examples")
    run_training(cfg, model, tokenizer, selected, val_set)
    # ``train`` force-saves the model (trainer.save_model) inside run_training.
    print(f"      model saved -> {cfg.output_dir}")


def stage_eval(cfg, benchmark, limit):
    """Load the fine-tuned model from the ``train`` stage, benchmark it, save metrics.

    ``cfg.eval.backend`` picks the generation engine: ``hf`` loads the model with
    transformers; ``vllm`` loads the same checkpoint (or base + LoRA adapter) into
    a vLLM engine for much faster batched decoding.
    """
    from utils.eval_utils import evaluate

    backend = (cfg.get_path("eval.backend") or "hf").lower()
    tokenizer = load_tokenizer(cfg)
    lora_request = None
    if backend == "vllm":
        from utils.model_utils import load_vllm
        model_path, adapter_dir = _resolve_trained_model(cfg)
        print(f"[eval] loading trained model into vLLM from {cfg.output_dir}")
        model, lora_request = load_vllm(cfg, model_path, adapter_dir)
    else:
        print(f"[eval] loading trained model from {cfg.output_dir}")
        model = _load_trained_model(cfg)

    print(f"[eval] evaluating on {benchmark} (backend={backend})")
    result = evaluate(cfg, model, tokenizer, benchmark, limit=limit,
                      lora_request=lora_request)
    out = _save_json(cfg, "eval.json", result)
    print(f"      {benchmark}: {result['metric']}={result['score']:.4f} "
          f"(n={result['n']}) -> {out}")


def main():
    # Exactly one stage per run; stages hand off through --output-dir artifacts.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--stage", default="select", choices=STAGES,
                     help="Which single phase to run (default: select): 'select' "
                          "(dataset -> subset), 'train' (subset -> model), or 'eval' "
                          "(model -> metrics). Each reads the previous stage's output "
                          "from --output-dir.")
    pre.add_argument("--benchmark", default=None,
                     help="Benchmark to evaluate (e.g. gsm8k). Required by --stage eval.")
    pre.add_argument("--eval-limit", type=int, default=100,
                     help="Number of benchmark examples to evaluate.")
    pre.add_argument("--eval-backend", default=None, choices=["hf", "vllm"],
                     help="Generation engine for --stage eval: 'hf' (transformers, "
                          "default) or 'vllm' (faster batched decoding; needs a CUDA "
                          "GPU + `pip install vllm`). Overrides eval.backend in config.")
    known, remaining = pre.parse_known_args()

    cfg = parse_args(remaining)
    if known.eval_backend:
        cfg.set_path("eval.backend", known.eval_backend)
    set_seed(cfg.seed)
    print(f"[stage] {known.stage}")

    if known.stage == "select":
        stage_select(cfg)
    elif known.stage == "train":
        stage_train(cfg)
    else:  # eval
        if not known.benchmark:
            raise SystemExit("--stage eval requires --benchmark NAME (e.g. "
                             "--benchmark gsm8k).")
        stage_eval(cfg, known.benchmark, known.eval_limit)


if __name__ == "__main__":
    main()
