"""LLM controller that orchestrates the operator cascade from *structural* state.

Instead of a fixed ``pipeline:`` list, an LLM decides each stage: it sees only
cheap structural signals (survivors remaining, budget spent, which operators
already ran) -- no per-step evaluation -- and returns the next stage dict, or
stops. Enable with::

    pipeline_planner:
      type: llm
      model: gpt-5-mini          # controller LLM (an OpenAI-compatible endpoint;
      base_url: null             #   NOT the local model being fine-tuned)
      max_stages: 6              # hard cap on cascade length
      min_keep: 0.01             # stop once survivors drop to this fraction
      goal: "..."                # optional free-text objective for the controller

Four guardrails keep an adaptive planner honest (see ``next_stage``):
  1. hard step cap                -- never exceed ``max_stages``
  2. budget floor / monotonicity  -- survivors must keep shrinking toward a floor
  3. operator validity            -- reject hallucinated method/scorer/policy
  4. decision logging             -- every prompt/response/verdict is appended to
                                     ``<output_dir>/planner_decisions.jsonl`` so a
                                     non-deterministic run stays reproducible.
"""

import importlib
import json
import os
import pkgutil
import urllib.request

from planner.base import Planner


def _list_modules(package):
    """Names of the operator modules in ``package`` (drops ``base``/dunder)."""
    module = importlib.import_module(package)
    return sorted(
        name for _, name, _ in pkgutil.iter_modules(module.__path__)
        if name != "base" and not name.startswith("_")
    )


def _extract_json(text):
    """Pull the first balanced ``{...}`` object out of an LLM reply."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON in reply")


def _chat(model, messages, base_url, api_key, temperature):
    """One OpenAI-compatible chat completion; returns the message content."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": model, "messages": messages, "temperature": temperature,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"]


_SYSTEM = """\
You orchestrate a DATA-SELECTION CASCADE for LLM fine-tuning. Each step you pick
ONE operator that filters the CURRENT survivors down to a smaller subset, or you
STOP. The goal is to end with a small, high-quality training subset.

You choose from these operators:
- method: one of {methods}. Omit or use "default" to compose a scorer+policy.
- when method is "default": scorer one of {scorers}; policy one of {policies}.
  Custom methods (everything except "default") wire their own scorer/policy, so
  do NOT set scorer/policy for them.
- budget: REQUIRED. A fraction in (0,1) of the CURRENT survivors to KEEP, or an
  integer count < current_size. The cascade multiplies: two 0.5 stages keep 25%.
- Optional: "model" (a smaller model for a cheap early stage), "reference" (a
  dataset name used as this stage's comparison anchor), and any method
  hyper-parameter as an extra key.

Reply with STRICT JSON only, no prose. Either a stage:
  {{"method": "ppl", "budget": 0.5}}
or, to finish the cascade:
  {{"stop": true}}

Guidance: cheap coarse filters (e.g. ppl on a small model) first, precise/
expensive operators (e.g. less) on the survivors, then stop once the subset is
small and clean. Do not repeat an operator that cannot improve the set."""


class LLMPlanner(Planner):
    def __init__(self, cfg):
        spec = cfg.get("pipeline_planner") or {}
        self.model = spec.get("model") or os.environ.get("PLANNER_MODEL", "gpt-5-mini")
        self.base_url = (spec.get("base_url")
                         or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        self.api_key = spec.get("api_key") or os.environ.get("OPENAI_API_KEY")
        self.max_stages = spec.get("max_stages", 6)
        self.min_keep = spec.get("min_keep", 0.01)
        self.temperature = spec.get("temperature", 0.0)
        self.max_retries = spec.get("max_retries", 2)
        self.goal = spec.get("goal")

        self.methods = _list_modules("alg")
        self.scorers = _list_modules("scorer")
        self.policies = _list_modules("policy")
        self.total = None  # cascade length is decided at run time

        self.log_path = os.path.join(cfg.output_dir or ".", "planner_decisions.jsonl")
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)

    # -- guardrail 3: reject a stage that names an unknown/ill-formed operator --
    def _validate(self, stage, state):
        if not isinstance(stage, dict):
            return "stage is not a JSON object"
        method = stage.get("method") or "default"
        if method not in self.methods:
            return f"unknown method {method!r}; choose from {self.methods}"
        if method == "default":
            scorer = stage.get("scorer")
            if scorer is not None and scorer not in self.scorers:
                return f"unknown scorer {scorer!r}; choose from {self.scorers}"
            policy = stage.get("policy")
            if policy is not None and policy not in self.policies:
                return f"unknown policy {policy!r}; choose from {self.policies}"
        budget = stage.get("budget")
        if budget is None:
            return "budget is required"
        if not isinstance(budget, (int, float)) or isinstance(budget, bool):
            return "budget must be a number"
        # guardrail 2: the stage must actually shrink the survivor set.
        if isinstance(budget, float):
            if not 0 < budget < 1:
                return "fractional budget must be in (0,1) to shrink survivors"
        else:  # integer count
            if not 0 < budget < state["current_size"]:
                return (f"integer budget must be in (0,{state['current_size']}) "
                        "to shrink survivors")
        return None

    def _record(self, entry):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def next_stage(self, state):
        step = len(state["history"])
        # guardrail 1: hard cap on cascade length.
        if step >= self.max_stages:
            self._record({"step": step, "verdict": "stop", "reason": "max_stages"})
            return None
        # guardrail 2: stop once survivors reach the budget floor.
        if state["kept_fraction"] <= self.min_keep:
            self._record({"step": step, "verdict": "stop", "reason": "min_keep"})
            return None

        system = _SYSTEM.format(methods=self.methods, scorers=self.scorers,
                                policies=self.policies)
        user = {
            "goal": self.goal or "Select a small, high-quality fine-tuning subset.",
            "original_size": state["original_size"],
            "current_size": state["current_size"],
            "kept_fraction": round(state["kept_fraction"], 4),
            "min_keep": self.min_keep,
            "steps_used": step,
            "max_stages": self.max_stages,
            "history": state["history"],
        }
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user)}]

        for attempt in range(self.max_retries + 1):
            raw = _chat(self.model, messages, self.base_url, self.api_key,
                        self.temperature)
            try:
                stage = _extract_json(raw)
            except ValueError as e:
                err = f"could not parse JSON: {e}"
                stage = None
            else:
                if stage.get("stop"):
                    self._record({"step": step, "raw": raw, "verdict": "stop",
                                  "reason": "llm"})
                    return None
                err = self._validate(stage, state)

            if err is None:
                self._record({"step": step, "raw": raw, "stage": stage,
                              "verdict": "accept"})
                return stage

            self._record({"step": step, "raw": raw, "stage": stage,
                          "verdict": "reject", "error": err, "attempt": attempt})
            # feed the error back so the model can correct itself.
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user",
                          "content": f"Invalid: {err}. Reply with corrected JSON only."}]

        # guardrail 2 (fallback): give up safely rather than loop forever.
        self._record({"step": step, "verdict": "stop", "reason": "retries_exhausted"})
        return None
