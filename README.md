# DataPan: Agentic Data Selection for LLM Finetuning

![Logo](./assets/datapan.png)

**DataPan** pans for gold in your data — it sifts large instruction-tuning corpora down to the examples that actually make a LLM better. 
Not all training data is worth its weight: some examples teach the model nothing, some even hurt. 
DataPan gives you a single, principled pipeline to score every example, keep the valuable ones, and fine-tune on that distilled subset.

What sets it apart is its **modular** design. Instead of treating every selection
method as a monolithic black box, DataPan factors data selection into a small set
of orthogonal, swappable pieces — so you can recombine published methods from a
config file, or assemble entirely new ones, without rewriting the pipeline.

The end-to-end flow is always the same:

```
parse args → load model+tokenizer → load dataset → select subset
           → (optionally) fine-tune → (optionally) evaluate a benchmark
```

```bash
python main.py --scorer embedding --policy diversity --budget 0.05   # default method
python main.py --method less --budget 0.05 --benchmark gsm8k
python main.py --method random --no-train          # baseline, selection only
```

### Static → manual → agentic

DataPan supports **three ways to drive data selection**, differing only in *who
decides which operators run* — the operators and the engine (`main.run_pipeline`)
are shared across all three:

| Mode | Who plans the cascade | How |
|------|----------------------|-----|
| **Static** | you — a single operator | `python main.py --method less --budget 0.05` — a one-stage run |
| **Manual pipeline** | you — a hand-authored cascade | a `pipeline:` list of stages, each filtering the previous stage's survivors |
| **Agentic** | an **LLM**, from the run's live state | `pipeline_planner: {type: llm}` — the model picks each stage under guardrails, no fixed list |

The agentic mode is the newest: instead of a pre-written pipeline, an LLM
controller observes the run's *structural* state (survivors left, budget spent,
operators already used) and orchestrates the cascade itself. See
[Four Ways to Use It](#four-ways-to-use-it) for the details of each.

---

## Modular Design

We decompose *"which data is worth training on"* into three independent axes.
Each axis is a directory of interchangeable plugins; a run is just a choice of
one plugin per axis.

| Axis | Question it answers | Lives in | Plugins |
|------|--------------------|----------|---------|
| **Scorer** | What makes an example valuable? (signal × target × model) | `src/scorer/` | `bm25`, `embedding`, `ppl` |
| **Policy** | How do per-example scores become a subset / weights? | `src/policy/` | `hard`, `diversity`, `reweight`, `greats` |
| **Timing** | *When* does selection run? | (selector) | offline subset · online per-step |

- A **Scorer** maps a dataset to per-example scores, `score(train, val) -> (scores, features)`,
  following the convention *higher == more valuable*. It fuses the three things
  that don't vary independently in practice — the **scoring metric** (lexical,
  representation, perplexity, gradient…), the **comparison target** (a validation
  anchor, the corpus itself, or nothing), and the **scoring model** (none, the
  frozen base model, a LoRA influence model…).
- A **Policy** is blind to *what* a score means; it only turns a vector of scores
  (and optional features) into per-sample weights `wᵢ ≥ 0`. **Hard** selection
  (`{0,1}` top-k mask) and **soft** reweighting (continuous weights) are the two
  ends of one primitive — selection is just the binary special case.
- **Timing** is owned by the selector: *offline* methods score once and hand a
  subset to the generic trainer; *online* methods (ADAPT, GREATS) override
  `make_trainer` to score and reweight **each minibatch** inside the training loop.

This is why two methods that look unrelated on paper often share machinery here:
e.g. ADAPT's online reweighting is just `reweight` policy on the online timing,
and GREATS is the gradient-based, diversity-aware sibling of ADAPT.

```
src/
├── scorer/   # ①②③  what defines value   → get_scorer(cfg, model, tokenizer)
├── policy/   # ④     scores → subset/weights → get_policy(cfg)
├── alg/      # ⑤     selectors: glue scorer+policy, decide timing → get_selector(...)
├── dataset/  #       loaders: dataset/load_<name>.py → get_dataset(cfg, tokenizer)
└── main.py   #       the pipeline
```

Each axis has a name-based registry (`get_scorer` / `get_policy` / `get_selector`)
that imports `<axis>/<name>.py` by config, plus a `BaseScorer` / `BasePolicy` /
`BaseSelector` to subclass. Plugins declare their own CLI flags via an
`add_args(parser)` function, loaded dynamically — so method-specific knobs stay
out of the shared `config.yaml`. Discover them with `--help`:

```bash
python main.py --scorer bm25 --policy diversity --help
python main.py --method less --help
```

---

## Four Ways to Use It

Modes 1–3 decide the stages **up front** (a fixed list, even if it's a single
stage); mode 4 lets an **LLM decide them at run time**.

### 1. Modular composition — pick a scorer + a policy (no code)

The `default` selector composes **any** scorer with **any** policy straight from
the config / CLI — it covers the "score the whole set once, then take a subset"
case with zero glue code. It is the **default method**, so `--method` can be
omitted; just choose a scorer and a policy:

```bash
# BM25 lexical relevance, plain top-k
python main.py --scorer bm25 --policy hard --budget 0.05

# perplexity signal, plain top-k
python main.py --scorer ppl --policy hard --budget 1000

# embedding similarity to a validation anchor, kept diverse via k-center coverage
python main.py --scorer embedding --policy diversity
```

Or set it once in `config.yaml`. A run is a **pipeline** — a cascade of
operators — and a single method is just a one-stage pipeline:

```yaml
pipeline:
  - method: default       # the modular selector; alg/default.py
    scorer: embedding     # scorer/<scorer>.py  (omit -> bm25)
    policy: diversity     # policy/<policy>.py
    budget: 0.05
```

The CLI (`--method` / `--scorer` / `--policy` / `--budget`) is a shortcut that
builds a one-stage pipeline and overrides this list.

`--scorer` selects `scorer/<name>.py`, `--policy` selects `policy/<name>.py`, and
both contribute their own tunables (`--bm25-k1`, etc.). Note that interaction-aware
policies like `diversity` need a scorer that exposes `features` (e.g. `embedding`).

> **`--scorer` / `--policy` only apply to the `default` method.** A custom method
> (mode 2 below) wires its own scorer *and* policy, so passing either alongside
> `--method <name>` is ignored and prints a warning.

### 2. DIY — write your own algorithm

When "one scorer → one policy" isn't enough — multiple scorers, gradient
plumbing, a custom reweighting trainer, online timing — drop a file in
`src/alg/<name>.py` that defines a `Selector(BaseSelector)`. You wire the scorer
and policy **in code** by importing the classes you want directly — exactly as
the scorer is imported — so the dependencies are explicit (`get_scorer`/
`get_policy` are reserved for the `default` selector):

```python
# src/alg/my_method.py
from alg.base import BaseSelector
from policy.diversity import Policy           # load the policy directly, like the scorer
from scorer.embedding import Scorer

DEFAULT_POLICY = "diversity"                  # mirror the import so utils.options loads
                                              # the policy's CLI flags; keep the two in sync

class Selector(BaseSelector):
    def __init__(self, cfg, model=None, tokenizer=None):
        super().__init__(cfg, model, tokenizer)
        self.scorer = Scorer(cfg, model, tokenizer)
        self.policy = Policy(cfg)             # your fixed ④ policy

    def select(self, train_dataset, val_dataset=None):
        scores, features = self.scorer.score(train_dataset, val_dataset)
        return self.apply_policy(scores, features=features)

    # optional: override for online per-step reweighting instead of an offline subset
    # def make_trainer(self, cfg, model, tokenizer, train_dataset, val_dataset): ...
```

Run it with `--method my_method`. Because the method owns its wiring, the CLI
`--scorer`/`--policy` don't reach it — the scorer and policy are whatever you
imported. If your policy exposes tunables (e.g. `diversity`, `reweight`), declare
a module-level `DEFAULT_POLICY` matching the import so its `add_args` flags load.
Expose method hyper-parameters by adding an `add_args(parser)` function (its
`dest` is the dotted config path it sets, e.g. `selection.warmup_steps`). The
published methods all live in `alg/` this way and are the best reference — see
`alg/less.py` (offline, gradient influence) and `alg/adapt.py` (online reweighting).

### 3. Orchestrate — chain operators into a pipeline

Every method above is an **operator**, and a run is a *cascade* of them: list
stages under `pipeline:` and each stage filters the previous stage's survivors.
A single method is just a one-stage pipeline — so this is the only execution
model; the `--method`/`--scorer`/`--policy`/`--budget` CLI is a shortcut that
builds a one-stage pipeline and overrides the `pipeline:` list.

```yaml
pipeline:
  - name: coarse-ppl        # cheap perplexity pass on a small model
    method: ppl
    model: Qwen/Qwen2.5-0.5B-Instruct
    budget: 0.5             # keep the top 50% of the full set
  - name: precise-less      # precise influence pass on the survivors
    method: less
    model: Qwen/Qwen2.5-3B-Instruct
    reference: bbh          # target set for LESS's influence match
    budget: 0.1             # keep the top 10% of the survivors -> 5% overall
    warmup_steps: 100       # a LESS knob (see below)
```

Each stage overlays the base config for that operator only. Stage keys:

| Key | Applies to | Meaning |
|-----|-----------|---------|
| `method` | all | Which operator (`alg/<method>.py`); omit → `default`. |
| `scorer` / `policy` | **`default` only** | Compose the `default` operator (`scorer/<s>.py`, `policy/<p>.py`). **Ignored for concrete methods** — they wire their own in code (a warning is printed if you set them, and the policy is forced to the method's `DEFAULT_POLICY`, e.g. `adapt → reweight`). |
| `model` | all | `model.name` for this stage's *selection* pass (see notes below). |
| `reference` | all | Dataset name loaded as this stage's `val_dataset` (the ② target/anchor, e.g. LESS's influence set). `null` reuses the top-level validation split. |
| `budget` | all | Fraction (`≤1`) or int count kept from the **current** survivors. |
| *any other key* | all | A **method hyper-parameter** — shorthand for `selection.<key>`. `warmup_steps: 100` in a stage is exactly `--warmup-steps 100` on the CLI. List a method's knobs with e.g. `--method less --help` (or read the `dest=` in its `add_args`). |

**Budgets compound** down the cascade: `0.5 → 0.1` keeps `0.5 × 0.1 = 5%` of the
original set. Orchestration lives in `main.run_pipeline`.

**Notes on `model`:**

- A stage's `model:` sets **only `model.name`**. Every other `model.*` field
  (`torch_dtype`, `max_length`, `load_in_8bit`, …) and all of `lora.*` always
  come from the top-level config.
- Per-stage models are used **only for selection/scoring**. The **final
  fine-tuning always uses the top-level `model.name`** (loaded once), regardless
  of what any stage used. So a stage can score influence with a 3B model while
  the run still fine-tunes the top-level model.
- Models are loaded lazily and cached; the top-level model stays resident and at
  most **one extra** stage model is held at a time, so long chains don't OOM.

**Final trainer:** offline stages return a subset and the run fine-tunes with the
generic Trainer. If the **last** stage's operator injects a trainer (an online
method like `adapt`), the run uses that instead — so a pipeline can end on
per-step reweighting.

### 4. Agentic — let an LLM plan the cascade

Modes 1–3 fix the stages before the run starts. The **agentic** mode hands that
decision to an **LLM controller**: at each step it sees the run's live
*structural* state — how many examples survive, how much budget is spent, which
operators already ran — and picks the next operator (or stops). No fixed list, no
per-stage planning by you; set an objective and turn it on in `config.yaml`:

```yaml
pipeline_planner:
  type: llm                 # omit / `list` -> the static `pipeline:` below runs as written
  model: gpt-5-mini         # an OpenAI-compatible controller LLM (see env below)
  max_stages: 6             # hard cap on cascade length
  min_keep: 0.01            # stop once survivors fall to this fraction of the original
  goal: "Select a small, high-quality subset for math reasoning."
```

```bash
export OPENAI_API_KEY=...             # controller endpoint credentials
export OPENAI_BASE_URL=...            # optional; defaults to the OpenAI API
python main.py                        # the LLM now orchestrates each stage
```

The controller composes the **same operators** as every other mode, just
adaptively — e.g. a cheap `ppl` pass on a small model to prune, then a precise
`less` pass on the survivors, then stop once the set is small and clean. It plans
from **structural signals only — no per-step evaluation** — so it stays cheap and
never touches the GPU (the controller LLM is separate from the local model being
fine-tuned).

**Guardrails** keep an autonomous planner honest:

| Guardrail | What it enforces |
|-----------|------------------|
| **Step cap** | Never exceed `max_stages` stages. |
| **Budget floor** | Every stage must *shrink* the survivor set; the cascade stops once `kept_fraction ≤ min_keep`. |
| **Operator validity** | A hallucinated `method`/`scorer`/`policy` is rejected and the model is re-prompted; after `max_retries` it stops safely rather than looping. |
| **Decision log** | Every prompt, raw reply and verdict is appended to `<output_dir>/planner_decisions.jsonl`. |

> **Reproducibility.** An LLM planner is non-deterministic. For comparable runs,
> pin `temperature: 0` and keep `planner_decisions.jsonl` — it captures the exact
> sequence of decisions. The seam lives in the `src/planner/` package —
> `planner/base.py` (the pluggable `Planner` + static `ListPlanner` that replays
> `pipeline:`) and `planner/llm.py` (the `LLMPlanner` controller); `main.run_pipeline`
> drives whichever planner `pipeline_planner.type` selects.

---

## Demo — DataPan UI (Streamlit)

A single-file Streamlit app that builds and runs pipelines visually — same engine
as the CLI, no config editing. Everything lives in
[`src/demo/streamlit_app.py`](src/demo/streamlit_app.py); see
[`src/demo/README.md`](src/demo/README.md) for details.

```bash
python3 -m streamlit run src/demo/streamlit_app.py
```

Flow (top → bottom of the page):

1. **Upload** an instruction dataset (`.jsonl` / `.json`; `{instruction, input,
   output}`, chat `messages`, or `prompt/response` schemas are auto-mapped).
2. **Orchestrate selection**, in either of two modes (a toggle at the top):
   - **Manual pipeline** — stack operator **cards**, each one stage (**name ·
     method · proxy model · budget · scorer · policy**, plus optional `reference`
     and method knobs). Add/remove and reorder with ▲/▼; the cascade runs top →
     bottom — the visual counterpart of the `pipeline:` list above.
   - **Agentic** — an **LLM controller** plans the cascade from live state (set
     the controller model / `max_stages` / `min_keep` / `goal`) — the visual
     counterpart of the `pipeline_planner: {type: llm}` mode above.
3. **Run selection**, then **download** the distilled subset as `.jsonl`, or
   **fine-tune** end-to-end on it.

A few things mirror the framework semantics described above:

- **scorer / policy locking** — a card's `scorer`/`policy` are editable **only
  when `method: default`** (the one operator that composes them). Concrete
  methods grey them out, and a method that pins a policy in code (`adapt →
  reweight`, `greats → greats`) shows it read-only — exactly the gating in
  `main._stage_cfg`.
- **real vs. simulated** — with PyTorch + transformers importable, "Run
  selection" runs the genuine cascade via `main.run_pipeline` (badge: **real**,
  uploaded data routed through `dataset/load_custom.py`). Otherwise it falls back
  to a seeded **budget-cascade simulation** — a preview of stage *sizes and
  wiring*, not of *which* examples a model-based scorer would pick.
- Either way, the result panel prints the exact `config.yaml` + CLI to reproduce
  the run on a GPU host.

---

## Supported Algorithms

| Method | `--method` | Venue | Idea |
|--------|-----------|-------|------|
| **IFD** (Cherry LLM) | `ifd` | NAACL 2024 | Self-guided Instruction-Following Difficulty; quality over quantity. |
| **LESS** | `less` | ICML 2024 | Selecting influential data for *targeted* tuning via LoRA gradient influence. |
| **MIWV** | `miwv` | AAAI 2026 | Rank samples by the ICL-based Model Instruction Weakness Value (training-free). |
| **GREATS** | `greats` | NeurIPS 2024 | **Online** batch selection: keep the most useful *and diverse* size-k subset each step. |
| **ADAPT** | `adapt` | ICLR 2026 | **Online** per-sample reweighting instead of offline subset selection. |

Baselines (also usable directly, or via the default method with `--scorer …`):

- **Random** selection (`--method random`)
- **BM25** lexical relevance (`--method bm25` or `--scorer bm25`)
- **Embedding** similarity (`--method embedding` or `--scorer embedding`)
- **Perplexity** (`--method ppl` or `--scorer ppl`)

## Supported Datasets

Pick with `--dataset <name>`; each loads from `dataset/load_<name>.py` and caches
to `{data_dir}/<name>.jsonl` on first use.

- **Alpaca** — Stanford Alpaca 52k
- **WizardLM** — Evol-Instruct
- **LESS** — mixture of Flan V2, CoT, Dolly and Open Assistant
- **GSM8K** — grade-school math word problems
- **BioInstruct** — 25k biomedical instructions
- **DialogSum** — dialogue summarization
