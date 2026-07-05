# DataPan Demo (Streamlit)

A single, self-contained Streamlit app over the selection pipeline —
everything lives in **`streamlit_app.py`** (catalog, dataset parsing, cards →
`cfg.pipeline`, real/simulated execution, and the UI).

## Run

```bash
python3 -m streamlit run src/demo/streamlit_app.py
```

## Flow (top → bottom)

1. **Upload** an instruction dataset (`.jsonl` / `.json`; `{instruction, input,
   output}`, chat `messages`, or `prompt/response` schemas auto-mapped).
2. **Orchestrate selection** — pick one of two modes (a radio at the top of
   step 2):
   - **Manual pipeline** — build a cascade of operator cards; per card:
     **name · method · proxy model · budget · scorer · policy** (+ optional
     `reference` and method knobs, e.g. `warmup_steps=100` → `selection.warmup_steps`).
     Add/remove cards and reorder with the ▲/▼ buttons (the cascade runs top →
     bottom, each stage filtering the previous survivors).
   - **Agentic** — an **LLM controller** plans the cascade itself from the run's
     live structural state (survivors, budget, history), under guardrails. Set the
     controller model / `max_stages` / `min_keep` / `goal` (and optional
     `base URL` / `API key`); this sets `cfg.pipeline_planner` (see
     `planner.llm.LLMPlanner`).
3. **Download** the distilled subset as `.jsonl`.
4. **Fine-tune** end-to-end on the selected subset.

## scorer / policy locking

Matching the framework (`main._stage_cfg`): a card's **scorer** and **policy**
are editable **only when `method: default`** — the one operator that composes a
scorer + policy. Concrete methods (`less`, `ifd`, `miwv`, …) wire their own and
ignore those fields, so they are greyed out and show their fixed policy
(`hard`) read-only.

## Real vs. simulated selection

- With **PyTorch + transformers** importable, "Run selection" runs the genuine
  cascade via `main.run_pipeline` on the uploaded data (validation split off so
  output rows map 1:1 to the input). Badge: **real**. Uploaded data reaches the
  pipeline through `dataset/load_custom.py`
  (`--dataset custom -o dataset.data_files=...`).
- Otherwise it falls back to a labelled **budget-cascade simulation**: each
  stage's budget is applied to the survivors exactly like the real cascade, with
  a deterministic seeded subset — a preview of *sizes and wiring*, not of *which*
  examples a model-based scorer would pick.

Either way the result panel prints the exact `config.yaml` + CLI to reproduce
the run on a GPU host.

**Agentic mode** runs for real only when the ML stack **and** a controller API
key (`OPENAI_API_KEY`, or one entered in the form) are both present — the LLM
plans stages while the operators run on GPU. Without both, "Run selection" skips
execution and hands back the ready-to-run `config.yaml` (with the
`pipeline_planner` block) to run on a GPU host — the API key is never written into
the rendered config.

## Optional: real drag-and-drop

Pure Streamlit can't drag rich cards, so reordering uses ▲/▼ buttons. Install
`streamlit-sortables` and the app auto-adds a drag-to-reorder strip; without it,
that feature is silently skipped.
