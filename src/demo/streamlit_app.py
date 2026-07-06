"""DataPan — single-file Streamlit demo for the data-selection pipeline.

Everything (catalog, dataset parsing, cards → cfg.pipeline, real/simulated
execution, and the whole UI) lives in THIS one file. No other demo module is
required.

Run:

    streamlit run src/demo/streamlit_app.py

Flow (top → bottom): upload an instruction dataset → orchestrate selection either
as a **manual pipeline** (a reorderable cascade of operator cards) or **agentic**
(an LLM controller plans each stage from live state) → run selection → download
the distilled subset → fine-tune end-to-end on it.

Agentic mode sets ``cfg.pipeline_planner`` (see ``planner.llm.LLMPlanner``); it
runs for real only with the ML stack + a controller API key present, otherwise it
hands back the ready-to-run config, like the "reproduce on a GPU host" panel.

Selection runs for real via ``main.run_pipeline`` when PyTorch + transformers are
importable; otherwise it falls back to a clearly-labelled budget-cascade
simulation so the flow works anywhere. Either way the result panel shows the
exact ``config.yaml`` + CLI to reproduce the run on a GPU host.

Reordering: pure Streamlit has no native drag-and-drop of rich widgets, so cards
reorder via ▲/▼ buttons. If the optional ``streamlit-sortables`` package is
installed, a real drag-and-drop strip is offered too.
"""

import json
import os
import random
import sys

import streamlit as st

# Make the framework (main.py, utils/, dataset/, alg/…) importable for the real
# pipeline path: this file is at <src>/demo/streamlit_app.py, so <src> is two up.
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
WORKSPACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_workspace")

# ===========================================================================
# 1) Catalog — mirrors the plugins under alg/ scorer/ policy/ dataset/.
#    method "default" is the ONLY operator that composes a scorer + policy;
#    concrete methods wire their own (scorer/policy greyed out; a pinned
#    default_policy is shown read-only). See main._stage_cfg / utils.options.
# ===========================================================================
MODEL_SUGGESTIONS = [
    "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct", "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]
SCORERS = ["bm25", "embedding", "ppl"]
POLICIES = ["hard", "diversity"]
# method -> (composes_scorer_policy, default_policy_or_None)
METHODS = {
    "default":   (True,  None),
    "random":    (False, None),
    "bm25":      (False, None),
    "ppl":       (False, None),
    "embedding": (False, None),
    "less":      (False, "hard"),
    "ifd":       (False, "hard"),
    "miwv":      (False, "hard"),
}
METHOD_NAMES = list(METHODS)
DEFAULTS = {"method": "default", "scorer": "bm25", "policy": "hard",
            "budget": "0.05", "model": ""}

# ===========================================================================
# 2) Dataset parsing / normalisation (arbitrary schema -> instruction/input/output)
# ===========================================================================
_INSTR_KEYS = ("instruction", "prompt", "question", "query", "input_text")
_OUTPUT_KEYS = ("output", "response", "answer", "completion", "target", "label")
_INPUT_KEYS = ("input", "context", "passage")


def _first(d, keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def normalise_record(obj):
    if not isinstance(obj, dict):
        return {"instruction": str(obj), "input": "", "output": ""}
    msgs = obj.get("messages") or obj.get("conversations")
    if isinstance(msgs, list) and msgs:
        def content(m):
            return (m.get("content") or m.get("value") or "") if isinstance(m, dict) else str(m)
        user = next((content(m) for m in msgs
                     if isinstance(m, dict) and m.get("role") in ("user", "human")), "")
        asst = next((content(m) for m in msgs
                     if isinstance(m, dict) and m.get("role") in ("assistant", "gpt")), "")
        if user or asst:
            return {"instruction": user or content(msgs[0]), "input": "",
                    "output": asst or (content(msgs[-1]) if len(msgs) > 1 else "")}
    return {"instruction": _first(obj, _INSTR_KEYS), "input": _first(obj, _INPUT_KEYS),
            "output": _first(obj, _OUTPUT_KEYS)}


def parse_records(text):
    text = (text or "").strip()
    raw = []
    if not text:
        return [], []
    if text[0] == "[":
        raw = json.loads(text)
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                raw.append(json.loads(line))
    return [normalise_record(r) for r in raw], raw


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


# ===========================================================================
# 3) Cards -> real cfg.pipeline + reproducible config.yaml / CLI
# ===========================================================================
def _num(v):
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return int(f) if f > 1 and f == int(f) else f
    except (TypeError, ValueError):
        return None


def _parse_params(text):
    """Parse ``key=value, ...`` method knobs into a dict (shorthand for
    ``selection.<key>`` — see ``main._stage_cfg``)."""
    parsed = {}
    for part in (text or "").split(","):
        k, sep, val = part.partition("=")
        if sep:
            try:
                import yaml
                parsed[k.strip()] = yaml.safe_load(val.strip())
            except Exception:
                parsed[k.strip()] = val.strip()
    return parsed


def card_to_stage(card):
    method = (card.get("method") or "default").strip()
    stage = {}
    if card.get("name"):
        stage["name"] = card["name"].strip()
    stage["method"] = method
    if method == "default":  # only the composing operator carries scorer/policy
        if card.get("scorer"):
            stage["scorer"] = card["scorer"]
        if card.get("policy"):
            stage["policy"] = card["policy"]
    if card.get("model"):
        stage["model"] = card["model"].strip()
    budget = _num(card.get("budget"))
    if budget is not None:
        stage["budget"] = budget
    if card.get("reference"):
        stage["reference"] = card["reference"].strip()
    # Method knobs are bare keys: any non-structural key -> selection.<key>.
    for k, v in _parse_params(card.get("params")).items():
        stage[k] = v
    return stage


def render_config_yaml(dataset, pipeline, planner=None):
    doc = {"dataset": {"name": dataset}}
    if planner:
        doc["pipeline_planner"] = planner   # agentic: an LLM plans the stages
    if pipeline or not planner:
        doc["pipeline"] = pipeline
    try:
        import yaml
        return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
    except Exception:
        return json.dumps(doc, indent=2)


def render_cli(dataset, pipeline, planner=None):
    if planner:
        return ("# agentic (LLM-planned): save the YAML above as config.yaml, "
                "set OPENAI_API_KEY, then:\npython main.py --config config.yaml")
    if len(pipeline) == 1:
        s = pipeline[0]
        parts = ["python main.py", f"--dataset {dataset}", f"--method {s.get('method','default')}"]
        structural = {"scorer", "policy", "model", "budget"}
        for flag, key in (("--scorer", "scorer"), ("--policy", "policy"),
                          ("--model", "model"), ("--budget", "budget")):
            if s.get(key) is not None and s.get(key) != "":
                parts.append(f"{flag} {s[key]}")
        # Bare method knobs (selection.<key>) -> generic -o override.
        for k, v in s.items():
            if k not in structural and k not in ("name", "method", "reference"):
                parts.append(f"-o selection.{k}={v}")
        return " ".join(parts)
    return "# multi-stage: save the YAML above as config.yaml, then:\npython main.py --config config.yaml"


# ===========================================================================
# 4) Execution — real cascade if the ML stack is present, else simulation
# ===========================================================================
def stack_available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def simulate(records, pipeline, seed):
    rng = random.Random(seed)
    survivors = list(range(len(records)))
    log = []
    for stage in pipeline:
        budget = stage.get("budget")
        current = survivors
        if budget is None:
            kept = current
        else:
            k = int(round(len(current) * budget)) if budget <= 1 else int(budget)
            k = max(1, min(k, len(current))) if current else 0
            kept = sorted(rng.sample(current, k)) if k < len(current) else current
        survivors = kept
        log.append({"name": stage.get("name") or stage.get("method") or "default",
                    "method": stage.get("method", "default"),
                    "budget": budget, "kept": len(survivors)})
    return sorted(survivors), log


def run_real(records, pipeline, seed, run_ws, planner_spec=None):
    from utils.options import load_config
    from utils.model_utils import load_model_and_tokenizer
    from dataset import get_dataset
    import main as pipeline_main

    cfg = load_config(os.path.join(SRC_DIR, "config.yaml"))
    if not isinstance(cfg.get("selection"), dict):
        cfg["selection"] = type(cfg)()
    data_path = write_jsonl(records, os.path.join(run_ws, "input.jsonl"))
    cfg.set_path("seed", seed)
    cfg.set_path("dataset.name", "custom")
    cfg.set_path("dataset.data_files", data_path)
    cfg.set_path("dataset.data_dir", run_ws)
    cfg.set_path("dataset.validation_split", 0.0)  # keep file order == record order
    cfg.set_path("output_dir", run_ws)
    cfg["pipeline"] = pipeline
    if planner_spec:  # agentic: let the LLM controller orchestrate (ignores `pipeline`)
        cfg["pipeline_planner"] = planner_spec
    model, tokenizer = load_model_and_tokenizer(cfg)
    data = get_dataset(cfg, tokenizer)
    indices, _, log = pipeline_main.run_pipeline(
        cfg, model, tokenizer, data["train"], data.get("validation"))
    return indices, log


def run_selection(records, pipeline, seed, run_ws, prefer_real=True):
    if prefer_real and stack_available():
        try:
            indices, log = run_real(records, pipeline, seed, run_ws)
            return {"mode": "real", "indices": indices, "log": log, "note": ""}
        except Exception as e:  # noqa: BLE001
            note = f"real pipeline unavailable ({type(e).__name__}: {e}); showing simulation"
    else:
        note = ("ML stack (transformers) not installed here — showing a budget-cascade "
                "simulation; run the generated CLI on a GPU host for real selection")
    indices, log = simulate(records, pipeline, seed)
    return {"mode": "simulated", "indices": indices, "log": log, "note": note}


# ===========================================================================
# 5) UI
# ===========================================================================
st.set_page_config(page_title="DataPan · Data Selection", page_icon="⛏️", layout="wide")
st.markdown("""
<style>
  .stApp h1 span.gold { color:#e0a83c; }
  div[data-testid="stMetricValue"] { color:#e0a83c; }
  .op-badge { background:#3a2f16; color:#e0a83c; border:1px solid #4a3c1a;
              border-radius:12px; padding:2px 10px; font-size:12px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ---- session state ----
ss = st.session_state
if "cards" not in ss:
    ss.cards = []
    ss.card_seq = 0
if "records" not in ss:
    ss.records = None
    ss.raw = None
    ss.result = None
    ss.selected = None
    ss.run_id = 0


def add_card():
    ss.card_seq += 1
    cid = ss.card_seq
    ss[f"name_{cid}"] = ""
    ss[f"method_{cid}"] = DEFAULTS["method"]
    ss[f"model_{cid}"] = DEFAULTS["model"]
    ss[f"budget_{cid}"] = DEFAULTS["budget"]
    ss[f"scorer_{cid}"] = DEFAULTS["scorer"]
    ss[f"policy_{cid}"] = DEFAULTS["policy"]
    ss[f"ref_{cid}"] = ""
    ss[f"pr_{cid}"] = ""
    ss.cards.append({"id": cid})


if not ss.cards:
    add_card()

# ---- header ----
left, right = st.columns([4, 1])
with left:
    st.markdown("# ⛏️ <span class='gold'>DataPan</span>", unsafe_allow_html=True)
    st.caption("Pan your instruction data down to the gold — then fine-tune on it.")
with right:
    if stack_available():
        st.success("⚙ real selection", icon="✅")
    else:
        st.warning("◐ simulation mode", icon="⚠️")

# ---- STEP 1: upload ----
st.subheader("1 · Upload input dataset")
up = st.file_uploader("Instruction dataset (.jsonl / .json). "
                      "Records of {instruction, input, output}; chat/other schemas auto-mapped.",
                      type=["jsonl", "json", "txt"])
if up is not None:
    try:
        records, raw = parse_records(up.getvalue().decode("utf-8"))
        if records:
            ss.records, ss.raw = records, raw
        else:
            st.error("No records parsed; expected a JSON array or JSONL of objects.")
    except Exception as e:  # noqa: BLE001
        st.error(f"Parse error: {e}")
if ss.records:
    fields = sorted({k for r in ss.raw[:50] if isinstance(r, dict) for k in r})
    st.success(f"**{len(ss.records)}** records loaded · fields: " + ", ".join(f"`{f}`" for f in fields))
    st.dataframe(ss.raw[:5], use_container_width=True, hide_index=True)

# ---- STEP 2: pipeline ----
st.subheader("2 · Data selection pipeline")
mode = st.radio(
    "How should the cascade be orchestrated?",
    ["Manual pipeline", "Agentic — LLM plans the cascade"],
    horizontal=True, key="orchestration_mode",
    help="Manual: you stack operator cards. Agentic: an LLM picks each stage from the "
         "run's live state (survivors, budget, history) under guardrails.")
manual = mode.startswith("Manual")

if not manual:
    st.caption("An **LLM controller** picks each stage from the run's live *structural* state "
               "— survivors left, budget spent, operators already used — no fixed list. "
               "Guardrails: hard step cap, budget floor, operator-name validation, decision log.")
    pc = st.columns(3)
    ss.planner_model = pc[0].text_input("controller model", value=ss.get("planner_model") or "gpt-5-mini")
    ss.planner_max = pc[1].number_input("max stages", value=int(ss.get("planner_max", 6)),
                                        min_value=1, step=1)
    ss.planner_minkeep = pc[2].text_input("min keep (fraction)",
                                          value=str(ss.get("planner_minkeep") or "0.01"))
    pc2 = st.columns(2)
    ss.planner_base = pc2[0].text_input("base URL (optional)", value=ss.get("planner_base") or "",
                                        placeholder="blank → OpenAI API / $OPENAI_BASE_URL")
    ss.planner_key = pc2[1].text_input("API key (optional)", value=ss.get("planner_key") or "",
                                       type="password", placeholder="blank → $OPENAI_API_KEY")
    ss.planner_goal = st.text_area("goal", height=68,
                                   value=ss.get("planner_goal") or "Select a small, high-quality subset.")
    if not (stack_available() and (ss.planner_key or os.environ.get("OPENAI_API_KEY"))):
        st.info("Agentic execution needs the ML stack **and** a controller API key. Without both, "
                "*Run selection* shows the ready-to-run config instead of executing here.")

if manual:
    st.caption("Each card is one operator. The cascade runs top → bottom; each stage filters "
               "the previous survivors. scorer & policy are editable **only** for method `default`.")

n = len(ss.cards)
for i, card in (enumerate(ss.cards) if manual else []):
    cid = card["id"]
    method_now = ss.get(f"method_{cid}", "default")
    with st.container(border=True):
        head = st.columns([0.5, 4, 0.5, 0.5, 0.6])
        head[0].markdown(f"<span class='op-badge'>#{i+1}</span>", unsafe_allow_html=True)
        head[1].markdown(f"**{ss.get(f'name_{cid}') or method_now}**")
        if head[2].button("▲", key=f"up_{cid}", disabled=(i == 0), help="move up"):
            ss.cards[i - 1], ss.cards[i] = ss.cards[i], ss.cards[i - 1]
            st.rerun()
        if head[3].button("▼", key=f"dn_{cid}", disabled=(i == n - 1), help="move down"):
            ss.cards[i + 1], ss.cards[i] = ss.cards[i], ss.cards[i + 1]
            st.rerun()
        if head[4].button("✕", key=f"rm_{cid}", disabled=(n == 1), help="remove"):
            ss.cards.pop(i)
            st.rerun()

        r1 = st.columns(4)
        r1[0].text_input("name", key=f"name_{cid}", placeholder="stage name (optional)")
        method = r1[1].selectbox("method", METHOD_NAMES, key=f"method_{cid}")
        r1[2].text_input("proxy model", key=f"model_{cid}", placeholder="e.g. Qwen/Qwen2.5-0.5B-Instruct")
        r1[3].text_input("budget (fraction ≤1 or count)", key=f"budget_{cid}", placeholder="0.05 or 500")

        composes, forced_policy = METHODS[method]
        r2 = st.columns(4)
        r2[0].selectbox("scorer", SCORERS, key=f"scorer_{cid}", disabled=not composes,
                        help=None if composes else f"ignored: {method} wires its own scorer")
        if composes:
            r2[1].selectbox("policy", POLICIES, key=f"policy_{cid}")
        else:
            shown = forced_policy or "(built-in)"
            opts = [shown] if shown not in POLICIES else POLICIES
            r2[1].selectbox("policy", opts, index=opts.index(shown), disabled=True,
                            key=f"policy_disp_{cid}",
                            help=f"{method} wires its own policy" + (f" → {forced_policy}" if forced_policy else ""))
        r2[2].text_input("reference (optional)", key=f"ref_{cid}", placeholder="e.g. bbh")
        r2[3].text_input("params (optional)", key=f"pr_{cid}",
                         placeholder="warmup_steps=100, projection_dim=4096")

if manual:
    c1, _ = st.columns([1.4, 5])
    if c1.button("➕ Add operator", use_container_width=True):
        add_card()
        st.rerun()

    # optional real drag-and-drop if the component is installed
    try:
        from streamlit_sortables import sort_items  # type: ignore

        labels = []
        for i, c in enumerate(ss.cards):
            cid = c["id"]
            name = ss.get(f"name_{cid}") or ss.get(f"method_{cid}")
            labels.append(f"#{i + 1} {name}")
        with st.expander("↕ Drag to reorder (optional component detected)"):
            new_order = sort_items(labels, direction="vertical")
            if new_order != labels:
                order = [labels.index(lbl) for lbl in new_order]
                ss.cards = [ss.cards[i] for i in order]
                st.rerun()
    except ImportError:
        pass

seed = st.number_input("seed", value=42, step=1)

run = st.button("▶ Run selection", type="primary", disabled=ss.records is None)
if ss.records is None:
    st.caption("Upload a dataset first (step 1) to enable selection.")

def make_planner_spec(include_key):
    """Build a `pipeline_planner` dict from the agentic form. ``include_key`` is
    False for the rendered config.yaml so the API key is never shown."""
    try:
        mk = float(ss.get("planner_minkeep") or 0.01)
    except (TypeError, ValueError):
        mk = 0.01
    spec = {"type": "llm",
            "model": (ss.get("planner_model") or "gpt-5-mini").strip(),
            "max_stages": int(ss.get("planner_max") or 6),
            "min_keep": mk,
            "goal": (ss.get("planner_goal") or "").strip()}
    if (ss.get("planner_base") or "").strip():
        spec["base_url"] = ss.planner_base.strip()
    if include_key and (ss.get("planner_key") or "").strip():
        spec["api_key"] = ss.planner_key.strip()
    return spec


if run and ss.records:
    ss.run_id += 1
    run_ws = os.path.join(WORKSPACE, f"run{ss.run_id:04d}")
    if manual:
        pipeline = [card_to_stage({
            "name": ss.get(f"name_{c['id']}"), "method": ss.get(f"method_{c['id']}"),
            "model": ss.get(f"model_{c['id']}"), "budget": ss.get(f"budget_{c['id']}"),
            "scorer": ss.get(f"scorer_{c['id']}"), "policy": ss.get(f"policy_{c['id']}"),
            "reference": ss.get(f"ref_{c['id']}"), "params": ss.get(f"pr_{c['id']}"),
        }) for c in ss.cards]
        with st.spinner("Running selection…"):
            res = run_selection(ss.records, pipeline, int(seed), run_ws, prefer_real=True)
        ss.selected = [ss.records[i] for i in res["indices"]]
        ss.result = {**res, "pipeline": pipeline, "planner": None, "num_total": len(ss.records)}
    else:
        exec_spec = make_planner_spec(include_key=True)
        disp_spec = make_planner_spec(include_key=False)   # rendered config — no secret
        has_key = bool((ss.get("planner_key") or "").strip() or os.environ.get("OPENAI_API_KEY"))
        if stack_available() and has_key:
            with st.spinner("LLM is planning & running the cascade…"):
                try:
                    indices, log = run_real(ss.records, [{"method": "default", "budget": 0.5}],
                                            int(seed), run_ws, planner_spec=exec_spec)
                    res = {"mode": "real", "indices": indices, "log": log, "note": ""}
                except Exception as e:  # noqa: BLE001
                    res = {"mode": "config", "indices": [], "log": [],
                           "note": f"agentic run failed ({type(e).__name__}: {e}); "
                                   "showing the config to run on a GPU host instead"}
        else:
            res = {"mode": "config", "indices": [], "log": [],
                   "note": "Agentic execution needs the ML stack + a controller API key here — "
                           "save the config below and run it on a GPU host."}
        ss.selected = [ss.records[i] for i in res["indices"]] if res["mode"] == "real" else None
        ss.result = {**res, "pipeline": [], "planner": disp_spec, "num_total": len(ss.records)}

# ---- STEP 3: result ----
if ss.result:
    res = ss.result
    st.subheader("3 · Result & download")
    if res["mode"] == "config":
        # Agentic run couldn't execute here (no ML stack / API key) — hand over the
        # ready-to-run config instead, matching the demo's "reproduce on a GPU host".
        st.info(res["note"])
        st.markdown("**Run this on a GPU host**")
        st.caption("config.yaml")
        st.code(render_config_yaml("custom", res["pipeline"], planner=res.get("planner")), language="yaml")
        st.caption("CLI")
        st.code(render_cli("custom", res["pipeline"], planner=res.get("planner")), language="bash")
    else:
        badge = st.success if res["mode"] == "real" else st.warning
        badge(f"mode: **{res['mode']}** — kept {len(ss.selected)} / {res['num_total']} examples")
        if res["note"]:
            st.info(res["note"])

        for i, s in enumerate(res["log"]):
            cols = st.columns([0.5, 2, 2, 3, 1.4])
            cols[0].markdown(f"<span class='op-badge'>#{i+1}</span>", unsafe_allow_html=True)
            cols[1].markdown(f"**{s['name']}**")
            cols[2].caption(f"method: {s['method']} · budget: {s.get('budget', '—')}")
            cols[3].progress(min(1.0, s["kept"] / max(1, res["num_total"])))
            cols[4].metric("kept", s["kept"], label_visibility="collapsed")

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Selected preview**")
            st.dataframe(ss.selected[:8], use_container_width=True, hide_index=True)
            st.download_button(
                "⬇ Download selected dataset (.jsonl)",
                data="\n".join(json.dumps(r, ensure_ascii=False) for r in ss.selected),
                file_name=f"selected_{res['num_total']}to{len(ss.selected)}.jsonl",
                mime="application/x-ndjson", type="primary")
        with g2:
            st.markdown("**Reproduce this run**")
            st.caption("config.yaml")
            st.code(render_config_yaml("custom", res["pipeline"], planner=res.get("planner")), language="yaml")
            st.caption("CLI")
            st.code(render_cli("custom", res["pipeline"], planner=res.get("planner")), language="bash")

# ---- STEP 4: train ----
if ss.selected:
    st.subheader("4 · End-to-end fine-tuning")
    st.caption("`select`, `train` and `eval` are fully decoupled — each runs on its own "
               "with `--stage` and hands artifacts to the next through `--output-dir`.")
    t = st.columns([2, 1, 1, 1])
    tmodel = t[0].selectbox("base model", MODEL_SUGGESTIONS)
    tepochs = t[1].number_input("epochs", value=3, min_value=1)
    tbatch = t[2].number_input("batch size", value=4, min_value=1)
    tlr = t[3].text_input("learning rate", value="2e-5")
    tbench = t[0].selectbox("eval benchmark", ["(none)", "gsm8k", "dialogsum"])
    tbackend = t[1].selectbox("eval backend", ["hf", "vllm"],
                              help="vllm = faster batched decoding (CUDA GPU + `pip install vllm`)")
    # The exported selected.jsonl is *already* the curated subset, so the select
    # stage keeps all of it (--method random --budget 1.0) rather than filtering
    # again. train then reads that saved subset; eval reads the trained model.
    common = f"--output-dir ./run --dataset custom -o dataset.data_files=./selected.jsonl"
    lines = [
        f"python main.py --stage select {common} --method random --budget 1.0",
        f"python main.py --stage train  {common} --model {tmodel} "
        f"--epochs {tepochs} --batch-size {tbatch} --lr {tlr}",
    ]
    if tbench != "(none)":
        lines.append(f"python main.py --stage eval   {common} --benchmark {tbench} "
                     f"--eval-backend {tbackend}")
    cli = "\n".join(lines)
    if st.button("🚀 Start training"):
        n_sel = len(ss.selected)
        if stack_available():
            st.info(f"Ready to fine-tune **{n_sel}** selected examples with `{tmodel}`. "
                    "Real multi-billion-param training is a long GPU job — launch the commands below.")
        else:
            st.warning("ML stack (transformers) not installed here. The selected examples are "
                       "ready — run the commands below on a machine with GPUs + transformers.")
        st.code(cli, language="bash")
        st.caption("Run the stages in order against the same `--output-dir`; each reads the "
                   "previous stage's output (subset → model → metrics) from there.")
