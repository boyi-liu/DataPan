"""GREATS: online selection of high-quality data in every iteration (NeurIPS 2024).

Reference: Wang, Wu, Song, Mittal, Jia. "GREATS: Online Selection of High-Quality
Data for LLM Training in Every Iteration."
Official code: https://github.com/Jiachen-T-Wang/GREATS

GREATS is an *online batch selection* method: at every training step it samples a
large batch B_t, scores each example by how much its gradient step would reduce
the validation loss, and updates the model on only the most useful and *diverse*
size-k subset (the paper uses k = 50% of the batch). It is the gradient-based,
interaction-aware sibling of ADAPT in this framework:

  * Scoring (this file). Per training step, compute each example's TracIN score
    ``φ⁰_i = <g_i, g_val>`` -- the inner product of its LoRA gradient with the
    (summed) validation gradient, refreshed every R steps under the current model
    (like ADAPT's anchor refresh).
  * Selection (``policy/greats.py``). The greedy Taylor optimizer turns those
    scores + the gradients ``g_i`` into a size-k subset, demoting redundancy after
    each pick (Eq. 4 correction). Selection is applied as a {0,1} mask on the
    batch loss, so "select k of the batch" == the binary case of reweighting --
    the same online machinery as ADAPT, with a gradient-based policy.

Timing-wise ``select`` returns *all* indices (nothing is dropped offline);
``cfg.selection.budget`` is reinterpreted by the policy as the per-*batch* keep
fraction, and the per-step selection lives in :class:`GREATSTrainer`.

Per-sample gradients
--------------------
Like LESS, this computes per-example LoRA gradients directly (one backward per
example), which is the paper's "direct" variant -- correct and simple on LoRA
(the base weights stay frozen, so the model handed to training is pristine). The
paper's headline efficiency trick, the **ghost inner-product** (Appendix A), gets
the same pairwise inner products from per-layer activations + output gradients in
a single backward, without materializing per-sample gradients; it is the natural
drop-in optimization here but is not implemented. For large LoRA ranks the
gradient feature dimension can be reduced with a random projection à la LESS.

Requires LoRA (operates on adapter gradients) and a small validation/anchor set.
Enable with ``--method greats``; defaults to ``--policy greats``.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DataCollatorForSeq2Seq, Trainer

from alg.base import BaseSelector
from policy.greats import Policy  # ④ policy loaded directly (get_policy is for `default`)
from utils.model_utils import maybe_wrap_lora
from utils.selector_utils import model_inputs, tqdm

# GREATS's selection *is* the greedy Taylor optimizer, so it loads the 'greats'
# policy directly above. DEFAULT_POLICY mirrors that choice so utils.options loads
# the right policy's CLI flags; keep the two in sync.
DEFAULT_POLICY = "greats"


# --------------------------------------------------------------------------- #
# Scorer: per-batch TracIN gradients + the validation gradient
# --------------------------------------------------------------------------- #
class GREATSScorer:
    """Produces the per-sample TracIN scores and gradient features for a batch.

    Owns the trainable (LoRA) parameter list, the cached validation gradient
    ``g_val = sum_v grad ell(w_t, z_v)``, and its periodic refresh under the
    current model state. The score -> subset mapping is the ④ policy's job.
    """

    def __init__(self, cfg, params, val_dataset):
        self.device = cfg.device
        self.params = params
        self.val_dataset = val_dataset
        sel = cfg.selection
        self.refresh_interval = int(sel.refresh_interval or 20)
        self.grad_dtype = np.float32
        self.val_grad = None          # (d,) numpy, summed validation gradient
        self._last_refresh = -1

    def _flat_grad(self):
        """Concatenate the current ``.grad`` of every trainable param into (d,)."""
        chunks = []
        for p in self.params:
            if p.grad is None:
                chunks.append(torch.zeros(p.numel(), device=self.device))
            else:
                chunks.append(p.grad.detach().reshape(-1))
        return torch.cat(chunks)

    @torch.enable_grad()
    def _grad_of(self, inputs, model):
        """Flat LoRA gradient of the loss on a single example -> (d,) tensor."""
        model.zero_grad(set_to_none=True)
        loss = model(**inputs).loss
        loss.backward()
        g = self._flat_grad()
        model.zero_grad(set_to_none=True)
        return g

    def maybe_refresh(self, model, step):
        """Recompute g_val under the current model every ``refresh_interval`` steps."""
        due = self.val_grad is None or (
            self.refresh_interval > 0
            and step % self.refresh_interval == 0
            and step != self._last_refresh
        )
        if not due:
            return
        was_training = model.training
        model.eval()                                  # deterministic (no dropout)
        acc = None
        for v in range(len(self.val_dataset)):
            g = self._grad_of(model_inputs(self.val_dataset[v], self.device), model)
            acc = g.clone() if acc is None else acc + g   # sum over val points
        if was_training:
            model.train()
        self.val_grad = acc.to(torch.float32).cpu().numpy()
        self._last_refresh = step

    @torch.enable_grad()
    def per_sample(self, inputs, model):
        """Per-example LoRA gradients for the collated batch -> (B, d) numpy.

        Each padded row is sliced out and run on its own; pad tokens carry
        ``attention_mask == 0`` / ``labels == -100`` so they don't affect the loss.
        """
        was_training = model.training
        model.eval()
        B = inputs["input_ids"].shape[0]
        feats = None
        for b in range(B):
            single = {k: v[b:b + 1] for k, v in inputs.items()}
            g = self._grad_of(single, model).to(torch.float32).cpu().numpy()
            if feats is None:
                feats = np.empty((B, g.shape[0]), dtype=self.grad_dtype)
            feats[b] = g
        if was_training:
            model.train()
        return feats


# --------------------------------------------------------------------------- #
# Trainer: select k of each batch via the greedy policy, then update on them
# --------------------------------------------------------------------------- #
class GREATSTrainer(Trainer):
    """Selects a subset of every batch with GREATS, then steps on the selection.

    ``compute_loss`` (1) scores the batch by per-sample gradient alignment with
    the validation gradient, (2) asks the ④ policy for a size-k subset mask, and
    (3) returns the masked-mean LM loss -- so the gradient step uses only the kept
    examples (Algorithm 1, line 14). Selection is the binary case of reweighting,
    so this reuses the same masked-loss application as ADAPT.
    """

    def __init__(self, *args, scorer=None, policy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scorer = scorer
        self.policy = policy

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("idx", None)
        labels = inputs["labels"]

        # (1) score the batch by gradient alignment with the validation set.
        self.scorer.maybe_refresh(model, self.state.global_step)
        G = self.scorer.per_sample(inputs, model)          # (B, d)
        scores = G @ self.scorer.val_grad                  # (B,) TracIN scores

        # (2) greedy Taylor selection -> {0, 1} keep mask over the batch.
        mask = self.policy.weights(scores, features=G)     # (B,)

        # (3) update step on the selected subset: masked-mean per-sample LM loss.
        model.zero_grad(set_to_none=True)
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        B, Tm1, V = logits.shape
        tok_loss = F.cross_entropy(
            logits.reshape(-1, V), shift_labels.reshape(-1),
            ignore_index=-100, reduction="none",
        ).view(B, Tm1)
        valid = (shift_labels != -100).to(tok_loss.dtype)
        per_sample = (tok_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        w = torch.as_tensor(mask, device=per_sample.device, dtype=per_sample.dtype)
        loss = (w * per_sample).sum() / w.sum().clamp(min=1.0)   # mean over selected
        return (loss, outputs) if return_outputs else loss


# --------------------------------------------------------------------------- #
# Selector: keeps the full dataset, wires up the online selection trainer
# --------------------------------------------------------------------------- #
class Selector(BaseSelector):
    def __init__(self, cfg, model=None, tokenizer=None):
        super().__init__(cfg, model, tokenizer)
        if model is None or tokenizer is None:
            raise ValueError("GREATS needs a model and tokenizer.")
        if not (cfg.lora and cfg.lora.enable):
            raise ValueError(
                "GREATS requires LoRA (set lora.enable: true). It scores LoRA "
                "adapter gradients so the base model is left untouched."
            )
        self.policy = Policy(cfg)

    # Online method: "selection" happens per batch in the trainer, so select()
    # returns every index and `budget` is the per-batch keep fraction.
    def select(self, train_dataset, val_dataset=None):
        if val_dataset is None or len(val_dataset) == 0:
            raise ValueError(
                "GREATS requires a non-empty validation/anchor set (set "
                "dataset.validation_split > 0)."
            )
        return list(range(len(train_dataset)))

    def make_trainer(self, cfg, model, tokenizer, train_dataset, val_dataset):
        from utils.train_utils import TRAINER_TOKENIZER_KW, build_training_args

        if val_dataset is None or len(val_dataset) == 0:
            raise ValueError("GREATS requires a non-empty validation/anchor set.")

        # Operate on LoRA adapters; base weights stay frozen and unchanged.
        model = maybe_wrap_lora(cfg, model)
        if getattr(model, "config", None) is not None:
            model.config.use_cache = False
        if cfg.train.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable parameters found after wrapping with LoRA.")

        scorer = GREATSScorer(cfg, params, val_dataset)
        args = build_training_args(cfg, len(train_dataset))
        collator = DataCollatorForSeq2Seq(
            tokenizer, model=model, padding="longest", label_pad_token_id=-100
        )
        return GREATSTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            scorer=scorer,
            policy=self.policy,           # greats (default) or e.g. hard / reweight
            **{TRAINER_TOKENIZER_KW: tokenizer},
        )


def add_args(parser):
    """Register GREATS-specific CLI arguments (loaded dynamically by utils.options).

    Scoring knobs only -- the score -> subset mapping (greedy correction strength)
    lives with policy/greats.py.
    """
    g = parser.add_argument_group("GREATS")
    g.add_argument("--greats-refresh-interval", type=int, default=20,
                   dest="selection.refresh_interval",
                   help="Recompute the validation gradient every R steps under the "
                        "current model (0 = compute once).")
