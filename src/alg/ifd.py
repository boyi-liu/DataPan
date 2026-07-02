"""IFD: Instruction-Following Difficulty (NAACL 2024).

Reference: Li, Zhang, Do, Yue, Chen. "From Quantity to Quality: Boosting LLM
Performance with Self-Guided Data Selection for Instruction Tuning."
https://arxiv.org/abs/2308.12032
Official code: https://github.com/tianyi-lab/Cherry_LLM

Self-guided, model-only selection: the model scores its own training data and
keeps the "cherry" samples. For a sample (instruction Q, answer A) the paper
defines two answer scores from the model's cross-entropy loss on the *response*
tokens:

    * Conditioned Answer Score  s(A|Q): the loss of A given the full prompt Q.
      This is exactly the standard instruction-tuning loss (prompt masked to
      -100), so `model(**inputs).loss` gives it directly.
    * Direct Answer Score       s(A):   the loss of A fed to the model *alone*,
      with no instruction context.

The Instruction-Following Difficulty score is their ratio (paper Eq. 3):

    IFD(Q, A) = s(A|Q) / s(A)

Intuition: a low IFD means the instruction already makes the answer easy to
predict (little is learned); a high IFD means the answer stays hard even with
the instruction -- the sample genuinely challenges the model's instruction
following. We therefore keep the *highest*-IFD samples. Scores >= 1 are dropped:
there the instruction *raised* the answer's loss, a signature of misaligned or
noisy pairs the paper explicitly filters out (Sec. 4.2).

Pipeline (paper Sec. 3):

    1. Learning from Brief Experience (optional). Briefly LoRA-tune the model on
       a small subset so the scores reflect an instruction-aware model rather
       than the raw pretrained one. The adapters are reset to their init state
       afterwards, so the model handed to the final run is pristine -- the brief
       model is used *only* for scoring.
    2. Evaluating based on Experience. Compute IFD over the full dataset with
       that model.
    3. Retraining from Self-Guided Experience. `BaseSelector.apply_policy`
       keeps the highest-IFD samples up to `cfg.selection.budget`, and the
       generic Trainer fine-tunes on them.

Notes
-----
* Brief experience runs on LoRA adapters (like LESS), so the base weights are
  never modified by selection. With ``lora.enable: false`` the warmup is skipped
  and IFD is computed on the pretrained model (the paper's ablation variant).
* The warmup subset is random by default; pass ``--ifd-warmup-cluster`` to pick
  it by k-means over instruction embeddings (the paper's diversity sampling).
"""

import numpy as np
import torch

from alg.base import BaseSelector
from policy.hard import Policy  # ④ policy loaded directly (get_policy is for `default`)
from utils.model_utils import maybe_wrap_lora
from utils.selector_utils import batched, mean_pool, model_inputs, tqdm


class Selector(BaseSelector):
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.999
    ADAM_EPS = 1e-8

    def __init__(self, cfg, model=None, tokenizer=None):
        super().__init__(cfg, model, tokenizer)
        if model is None or tokenizer is None:
            raise ValueError("IFD needs a model and tokenizer.")
        self.policy = Policy(cfg)

        self.device = cfg.device
        sel = cfg.selection
        self.warmup_steps = int(sel.warmup_steps or 0)
        self.warmup_subset = float(sel.warmup_subset or 0.01)
        self.warmup_cluster = bool(sel.warmup_cluster)
        self.n_clusters = int(sel.n_clusters or 100)
        self.ifd_max = float(sel.ifd_max if sel.ifd_max is not None else 1.0)
        self.encode_batch = int(sel.encode_batch or 16)

        self.lora_enabled = bool(cfg.lora and cfg.lora.enable)
        if self.warmup_steps > 0 and not self.lora_enabled:
            print(
                "[IFD] warmup_steps > 0 but LoRA is disabled; skipping brief "
                "experience to avoid mutating the base model. Scoring with the "
                "pretrained model instead."
            )
            self.warmup_steps = 0

        # Operate on LoRA adapters when enabled; base weights stay frozen so the
        # model handed to the final run is unchanged by selection.
        self.model = maybe_wrap_lora(cfg, model) if self.lora_enabled else model
        if getattr(self.model, "config", None) is not None:
            self.model.config.use_cache = False
        self._params = [p for p in self.model.parameters() if p.requires_grad]

    # ---- BaseSelector API --------------------------------------------------

    def select(self, train_dataset, val_dataset=None):
        if self.warmup_steps > 0 and self._params:
            self._brief_experience(train_dataset)

        ifd = self._ifd_scores(train_dataset)

        valid = np.isfinite(ifd) & (ifd > 0) & (ifd < self.ifd_max)
        n_valid = int(valid.sum())
        print(f"[IFD] {n_valid}/{len(ifd)} samples with 0 < IFD < {self.ifd_max}")

        # Filtered samples (IFD >= max, or undefined) sink below every valid one
        # so they are only ever picked when the budget exceeds the valid pool.
        scores = np.where(valid, ifd, -np.inf)
        return self.apply_policy(scores.tolist())

    # ---- step 1: learning from brief experience ----------------------------

    def _brief_experience(self, train_dataset):
        """Briefly LoRA-tune the model, then restore the init adapter weights.

        Mirrors the LESS warmup loop. The trained adapters shape the IFD scores
        but are rolled back afterwards, so the (frozen-base + init-adapter) model
        that flows into final training is identical to the one we started with.
        """
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from transformers import (
            DataCollatorForSeq2Seq,
            get_linear_schedule_with_warmup,
        )

        init_state = {n: p.detach().clone() for n, p in self.model.named_parameters()
                      if p.requires_grad}

        subset = self._warmup_subset(train_dataset)
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

        self.model.train()
        step = 0
        pbar = tqdm(total=total, desc="IFD brief experience")
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
        pbar.close()

        # Roll the adapters back to their initialization (keep base untouched).
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in init_state:
                    p.copy_(init_state[n])

    def _warmup_subset(self, train_dataset):
        n = len(train_dataset)
        sub_n = max(1, min(n, int(round(self.warmup_subset * n))))
        if self.warmup_cluster:
            idx = self._cluster_subset(train_dataset, sub_n)
        else:
            rng = np.random.default_rng(self.cfg.seed)
            idx = rng.choice(n, size=sub_n, replace=False).tolist()
        return train_dataset.select(sorted(idx))

    def _cluster_subset(self, train_dataset, sub_n):
        """Diversity sampling: k-means on instruction embeddings, evenly per cluster."""
        emb = self._embed(train_dataset)                       # (N, d)
        k = min(self.n_clusters, sub_n, len(emb))
        labels = self._kmeans(emb, k)
        rng = np.random.default_rng(self.cfg.seed)
        chosen, c = [], 0
        # Round-robin across clusters so the subset stays balanced.
        buckets = [list(rng.permutation(np.where(labels == j)[0])) for j in range(k)]
        while len(chosen) < sub_n and any(buckets):
            b = buckets[c % k]
            if b:
                chosen.append(int(b.pop()))
            c += 1
        return chosen[:sub_n]

    @torch.no_grad()
    def _embed(self, dataset):
        """Mean-pooled last-layer features for each example (for clustering)."""
        self.model.eval()
        from transformers import DataCollatorForSeq2Seq
        collate = DataCollatorForSeq2Seq(
            self.tokenizer, padding="longest", label_pad_token_id=-100
        )
        keys = ("input_ids", "attention_mask")
        feats = []
        for i in tqdm(range(0, len(dataset), self.encode_batch), desc="IFD embed"):
            rows = [dataset[j] for j in range(i, min(i + self.encode_batch, len(dataset)))]
            enc = collate([{k: r[k] for k in ("input_ids", "attention_mask", "labels")}
                           for r in rows])
            enc = {k: enc[k].to(self.device) for k in keys}
            out = self.model(**enc, output_hidden_states=True)
            pooled = mean_pool(out.hidden_states[-1], enc["attention_mask"])
            feats.append(pooled.float().cpu().numpy())
        return np.concatenate(feats, axis=0)

    def _kmeans(self, x, k, iters=25):
        """Tiny dependency-free Lloyd's k-means; returns a cluster label per row."""
        rng = np.random.default_rng(self.cfg.seed)
        centers = x[rng.choice(len(x), size=k, replace=False)]
        labels = np.zeros(len(x), dtype=np.int64)
        for _ in range(iters):
            d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            new = d.argmin(1)
            if np.array_equal(new, labels):
                break
            labels = new
            for j in range(k):
                members = x[labels == j]
                if len(members):
                    centers[j] = members.mean(0)
        return labels

    # ---- step 2: IFD scoring -----------------------------------------------

    @torch.no_grad()
    def _ifd_scores(self, dataset):
        """IFD = s(A|Q) / s(A) per example, computed with two forward passes."""
        self.model.eval()
        scores = np.empty(len(dataset), dtype=np.float64)
        for i in tqdm(range(len(dataset)), desc="IFD scoring"):
            example = dataset[i]
            cond = self._conditioned_loss(example)   # s(A|Q)
            direct = self._direct_loss(example)       # s(A)
            if cond is None or direct is None or direct <= 0:
                scores[i] = np.nan
            else:
                scores[i] = cond / direct
        return scores

    def _conditioned_loss(self, example):
        """s(A|Q): loss on the response given the full prompt (prompt masked)."""
        if all(l == -100 for l in example["labels"]):
            return None
        out = self.model(**model_inputs(example, self.device))
        return float(out.loss)

    def _direct_loss(self, example):
        """s(A): loss on the response tokens fed alone, with no instruction."""
        response_ids = [tok for tok, lab in zip(example["input_ids"], example["labels"])
                        if lab != -100]
        if len(response_ids) < 2:  # need >=2 tokens for a next-token loss
            return None
        ids = batched(response_ids, self.device)
        out = self.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            labels=ids,
        )
        return float(out.loss)


def add_args(parser):
    """Register IFD-specific CLI arguments (loaded dynamically by utils.options)."""
    g = parser.add_argument_group("IFD")
    g.add_argument("--ifd-warmup-steps", type=int, default=0,
                   dest="selection.warmup_steps",
                   help="Brief-experience LoRA steps before scoring (0 = score the "
                        "pretrained model). Adapters are reset afterwards.")
    g.add_argument("--ifd-warmup-subset", type=float, default=0.01,
                   dest="selection.warmup_subset",
                   help="Fraction of the train set used for brief experience.")
    g.add_argument("--ifd-warmup-cluster", action="store_true",
                   dest="selection.warmup_cluster",
                   help="Pick the warmup subset by k-means over instruction "
                        "embeddings (diversity sampling) instead of at random.")
    g.add_argument("--ifd-n-clusters", type=int, default=100,
                   dest="selection.n_clusters",
                   help="Number of clusters for diversity sampling.")
    g.add_argument("--ifd-max", type=float, default=1.0,
                   dest="selection.ifd_max",
                   help="Drop samples with IFD >= this (misaligned/noisy pairs).")
    g.add_argument("--ifd-encode-batch", type=int, default=16,
                   dest="selection.encode_batch",
                   help="Batch size for embedding extraction (clustering only).")
