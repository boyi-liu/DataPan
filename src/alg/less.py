"""LESS: Selecting Influential Data for Targeted Instruction Tuning (ICML 2024).

Reference: Xia, Malladi, Gururangan, Arora, Chen. https://arxiv.org/abs/2402.04333
Official code: https://github.com/princeton-nlp/LESS

Pipeline implemented here:

    1. Warmup. Train LoRA adapters on a small random subset of the data for a
       few hundred steps, snapshotting `num_checkpoints` reference models along
       the way (each with its Adam optimizer state and the learning rate at that
       point).

    2. Gradient features. For every checkpoint, compute a per-example gradient
       of the LoRA parameters and reduce it with a fixed random projection:
         - train examples use the **Adam** gradient (the raw gradient passed
           through one Adam update using the checkpoint's (m, v) moments), which
           matches how the data would actually move the model;
         - target/validation examples use the plain **SGD** gradient.
       Each projected feature is L2-normalized so inner products are cosines.

    3. Influence matching. For each checkpoint, influence(train, val) is the
       cosine between their features; we average over the target set, weight by
       the checkpoint's learning rate, and sum across checkpoints.

`BaseSelector.apply_policy` then keeps the highest-influence examples up to
`cfg.selection.budget`.

Notes
-----
* LESS operates on LoRA adapters, so the base model weights are never modified
  by selection -- the model handed back to the final training run is pristine.
* The random projector builds a dense matrix when `input_dim * projection_dim`
  is small enough (fast path) and otherwise projects in blocks. For very large
  models the official `trak` CUDA projector is faster; this implementation
  trades speed for being dependency-free and device-agnostic.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from alg.base import BaseSelector
from policy.hard import Policy  # ④ policy loaded directly (get_policy is for `default`)
from utils.model_utils import maybe_wrap_lora
from utils.selector_utils import model_inputs, tqdm


@dataclass
class _Checkpoint:
    """A warmup snapshot: trainable params + Adam moments + learning rate."""
    params: List[torch.Tensor]      # CPU clones, in trainable-param order
    exp_avg: torch.Tensor           # CPU, concatenated first moments
    exp_avg_sq: torch.Tensor        # CPU, concatenated second moments
    lr: float


class _GradientProjector:
    """Fixed random (Rademacher) projection D -> proj_dim, reused for all inputs.

    Builds a single dense projection matrix when it fits in memory, otherwise
    projects in blocks, regenerating each block's matrix deterministically.
    """

    def __init__(self, proj_dim, seed, device, max_dense=10 ** 8):
        self.proj_dim = int(proj_dim)
        self.seed = int(seed)
        self.device = device
        self.max_dense = int(max_dense)
        self.input_dim = None
        self.mode = None
        self.matrix = None
        self.block_size = None
        self._scale = self.proj_dim ** 0.5

    def _rademacher(self, rows, gen_seed):
        gen = torch.Generator(device="cpu").manual_seed(gen_seed)
        bits = torch.randint(0, 2, (rows, self.proj_dim), generator=gen, dtype=torch.int8)
        return (bits.to(torch.float32) * 2 - 1) / self._scale

    def _build(self, input_dim):
        self.input_dim = int(input_dim)
        if self.input_dim * self.proj_dim <= self.max_dense:
            self.mode = "dense"
            self.matrix = self._rademacher(self.input_dim, self.seed).to(self.device)
        else:
            self.mode = "blocked"
            self.block_size = max(1, self.max_dense // self.proj_dim)

    def project(self, vec):
        """vec: 1-D tensor (D,) on any device/dtype -> (proj_dim,) float32 on device."""
        if self.input_dim is None:
            self._build(vec.numel())
        vec = vec.to(self.device, torch.float32)
        if self.mode == "dense":
            return vec @ self.matrix
        out = torch.zeros(self.proj_dim, device=self.device, dtype=torch.float32)
        for block, start in enumerate(range(0, self.input_dim, self.block_size)):
            end = min(start + self.block_size, self.input_dim)
            mat = self._rademacher(end - start, self.seed + block + 1).to(self.device)
            out += vec[start:end] @ mat
            del mat
        return out


class Selector(BaseSelector):
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.999
    ADAM_EPS = 1e-8

    def __init__(self, cfg, model=None, tokenizer=None):
        super().__init__(cfg, model, tokenizer)
        if not (cfg.lora and cfg.lora.enable):
            raise ValueError(
                "LESS requires LoRA (set lora.enable: true). It operates on adapter "
                "gradients so the base model is left untouched for final training."
            )
        if model is None or tokenizer is None:
            raise ValueError("LESS needs a model and tokenizer.")
        self.policy = Policy(cfg)

        self.device = cfg.device
        sel = cfg.selection
        self.warmup_steps = int(sel.warmup_steps or 200)
        self.warmup_subset = float(sel.warmup_subset or 0.05)
        self.num_checkpoints = int(sel.num_checkpoints or 4)
        self.proj_dim = int(sel.projection_dim or 8192)
        self.val_aggregation = sel.val_aggregation or "mean"
        self.max_dense_proj = int(sel.max_dense_proj or 10 ** 8)

        # Operate on LoRA adapters; base weights stay frozen, hence unchanged.
        self.model = maybe_wrap_lora(cfg, model)
        if getattr(self.model, "config", None) is not None:
            self.model.config.use_cache = False
        self._params = [p for p in self.model.parameters() if p.requires_grad]
        if not self._params:
            raise ValueError("No trainable parameters found after wrapping with LoRA.")

        self.projector = _GradientProjector(
            self.proj_dim, cfg.seed, self.device, max_dense=self.max_dense_proj
        )

    # ---- BaseSelector API --------------------------------------------------

    def select(self, train_dataset, val_dataset=None):
        if val_dataset is None or len(val_dataset) == 0:
            raise ValueError("LESS requires a non-empty validation/target set.")

        checkpoints = self._warmup(train_dataset)
        weights = self._checkpoint_weights(checkpoints)

        influence = np.zeros(len(train_dataset), dtype=np.float64)
        for ckpt, weight in zip(checkpoints, weights):
            self._load_checkpoint(ckpt)
            adam = (ckpt.exp_avg.to(self.device), ckpt.exp_avg_sq.to(self.device))
            train_feats = self._projected_grads(train_dataset, adam_state=adam, tag="train grads")
            val_feats = self._projected_grads(val_dataset, adam_state=None, tag="val grads")

            sim = train_feats @ val_feats.T            # (N, V), unit-normalized rows
            if self.val_aggregation == "max":
                agg = sim.max(axis=1)
            else:
                agg = sim.mean(axis=1)
            influence += weight * agg

        return self.apply_policy(influence)

    # ---- step 1: warmup ----------------------------------------------------

    def _warmup(self, train_dataset):
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from transformers import (
            DataCollatorForSeq2Seq,
            get_linear_schedule_with_warmup,
        )

        n = len(train_dataset)
        sub_n = max(1, int(self.warmup_subset * n))
        rng = np.random.default_rng(self.cfg.seed)
        subset = train_dataset.select(sorted(rng.choice(n, size=sub_n, replace=False).tolist()))

        collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model, padding="longest", label_pad_token_id=-100
        )
        loader = DataLoader(
            subset, batch_size=self.cfg.train.batch_size, shuffle=True, collate_fn=collator
        )

        optimizer = AdamW(
            self._params,
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay or 0.0,
            betas=(self.ADAM_BETA1, self.ADAM_BETA2),
            eps=self.ADAM_EPS,
        )
        total = self.warmup_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int((self.cfg.train.warmup_ratio or 0.0) * total), total
        )
        ckpt_steps = sorted({
            max(1, round(total * (j + 1) / self.num_checkpoints))
            for j in range(self.num_checkpoints)
        })

        self.model.train()
        checkpoints, step = [], 0
        pbar = tqdm(total=total, desc="LESS warmup")
        while step < total:
            for batch in loader:
                if step >= total:
                    break
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                loss = self.model(**batch).loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1
                pbar.update(1)
                if step in ckpt_steps:
                    checkpoints.append(self._snapshot(optimizer, scheduler))
        pbar.close()
        return checkpoints

    def _snapshot(self, optimizer, scheduler):
        params = [p.detach().to("cpu", copy=True) for p in self._params]
        exp_avg = torch.cat([
            optimizer.state[p]["exp_avg"].detach().reshape(-1).cpu() for p in self._params
        ])
        exp_avg_sq = torch.cat([
            optimizer.state[p]["exp_avg_sq"].detach().reshape(-1).cpu() for p in self._params
        ])
        return _Checkpoint(params, exp_avg, exp_avg_sq, scheduler.get_last_lr()[0])

    def _load_checkpoint(self, ckpt):
        with torch.no_grad():
            for p, value in zip(self._params, ckpt.params):
                p.copy_(value.to(self.device))

    def _checkpoint_weights(self, checkpoints):
        manual = self.cfg.selection.ckpt_weights
        if manual:
            weights = np.asarray(list(manual), dtype=np.float64)
        else:
            weights = np.asarray([c.lr for c in checkpoints], dtype=np.float64)
        total = weights.sum()
        if total <= 0:
            return np.full(len(checkpoints), 1.0 / len(checkpoints))
        return weights / total

    # ---- step 2: projected per-example gradients ---------------------------

    def _adam_transform(self, grad, exp_avg, exp_avg_sq):
        """One Adam update applied to the fresh gradient using stored moments."""
        grad = grad.to(torch.float32)
        m = self.ADAM_BETA1 * exp_avg + (1 - self.ADAM_BETA1) * grad
        v = self.ADAM_BETA2 * exp_avg_sq + (1 - self.ADAM_BETA2) * grad ** 2
        return m / (torch.sqrt(v) + self.ADAM_EPS)

    def _flat_grad(self):
        chunks = []
        for p in self._params:
            if p.grad is None:
                chunks.append(torch.zeros(p.numel(), device=self.device))
            else:
                chunks.append(p.grad.detach().reshape(-1))
        return torch.cat(chunks)

    @torch.enable_grad()
    def _projected_grads(self, dataset, adam_state=None, tag="grads"):
        self.model.eval()  # deterministic (no dropout); grads still flow
        feats = np.empty((len(dataset), self.proj_dim), dtype=np.float32)
        for i in tqdm(range(len(dataset)), desc=f"LESS {tag}"):
            self.model.zero_grad(set_to_none=True)
            loss = self.model(**model_inputs(dataset[i], self.device)).loss
            loss.backward()

            grad = self._flat_grad()
            if adam_state is not None:
                grad = self._adam_transform(grad, adam_state[0], adam_state[1])

            proj = self.projector.project(grad)
            proj = proj / (proj.norm() + 1e-10)
            feats[i] = proj.detach().cpu().numpy()
        self.model.zero_grad(set_to_none=True)
        return feats


def add_args(parser):
    """Register LESS-specific CLI arguments.

    Loaded dynamically by utils.options when ``--method less`` is selected.
    Defaults live here (not in config.yaml); every value lands under the
    ``selection.*`` namespace of the resolved config.
    """
    group = parser.add_argument_group("LESS")
    group.add_argument("--warmup-steps", type=int, default=200,
                       dest="selection.warmup_steps",
                       help="Steps to train the reference LoRA model.")
    group.add_argument("--warmup-subset", type=float, default=0.05,
                       dest="selection.warmup_subset",
                       help="Fraction of the train set used for the warmup run.")
    group.add_argument("--num-checkpoints", type=int, default=4,
                       dest="selection.num_checkpoints",
                       help="Number of gradient-feature checkpoints taken during warmup.")
    group.add_argument("--projection-dim", type=int, default=8192,
                       dest="selection.projection_dim",
                       help="Dimension of the random gradient projection.")
    group.add_argument("--val-aggregation", choices=["mean", "max"], default="mean",
                       dest="selection.val_aggregation",
                       help="How to aggregate influence over the target set.")
    group.add_argument("--max-dense-proj", type=int, default=10 ** 8,
                       dest="selection.max_dense_proj",
                       help="Build a dense projection matrix below this many elements.")
