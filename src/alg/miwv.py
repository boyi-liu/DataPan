"""MIWV: Model Instruction Weakness Value (AAAI 2026).

Reference: Jiang, Li, Song, Zhang, Zhu, Zhao, Xu, Taura, Wang. "Importance-Aware
Data Selection for Efficient LLM Instruction Tuning." https://arxiv.org/abs/2511.07074

A model-only, training-free selector that ranks instruction samples by how much
they expose a model's *weakness* under In-Context Learning (ICL). The idea: if
showing the model a similar solved example (a one-shot context) sharply lowers
its loss on a sample's answer, the model was previously weak there and the sample
carries something genuinely worth learning. Such high-weakness samples form the
high-quality subset.

The method has three steps (paper Sec. 3):

  1. One-Shot Example Retrieval. Embed every sample's *prompt* (instruction, plus
     input if any -- the response is excluded, matching the paper's
     ``x = map(Instruction, [Input])``). For each sample x_i find its nearest
     neighbour x_k by cosine similarity (excluding itself); the full pair
     ``C = Prompt(x_k, y_k)`` becomes its one-shot context.

  2. Computation of Sample Importance. With two forward passes per sample compute
     the mean cross-entropy on the *response* tokens of (x_i, y_i):

         L(y_i | x_i)     -- the plain instruction-tuning loss (prompt masked).
         L(y_i | x_i, C)  -- the same loss, but with the one-shot pair C
                             prepended to the input as context.

     and define the Model Instruction Weakness Value (paper Eq. 8):

         MIWV(x_i, y_i) = L(y_i | x_i, C) - L(y_i | x_i)

     A *high* MIWV means the one-shot context helped a lot (the model was weak on
     its own) -- a valuable, capability-enhancing sample. A low/negative MIWV
     means the context didn't help (or an irrelevant neighbour hurt), so the
     sample teaches little; a high value there also keeps diversity, since an
     irrelevant neighbour that fails to help still surfaces the sample.

  3. High-Quality Data Selection. ``BaseSelector.apply_policy`` keeps the
     highest-MIWV samples up to ``cfg.selection.budget`` and the generic Trainer
     fine-tunes on them.

Notes
-----
* No warmup, no gradients, no validation set: the model is used purely for
  inference and is never mutated by selection.
* Retrieval embeddings default to the fine-tuning model's own mean-pooled
  last-layer hidden states (no extra encoder), matching ``alg/embedding.py``.
  Pass ``--miwv-embed-model <name>`` to instead embed with a sentence-transformer
  (e.g. ``BAAI/bge-large-en-v1.5``, the paper's choice) when it is installed.
* Both losses average over the *same* response tokens of x_i, so their difference
  is a clean per-token quantity regardless of answer length.

Enable with ``--method miwv``.
"""

import numpy as np
import torch
import torch.nn.functional as F

from alg.base import BaseSelector
from policy.hard import Policy  # ④ policy loaded directly (get_policy is for `default`)
from utils.selector_utils import batched, mean_pool, tqdm


class Selector(BaseSelector):
    def __init__(self, cfg, model=None, tokenizer=None):
        super().__init__(cfg, model, tokenizer)
        if model is None or tokenizer is None:
            raise ValueError("MIWV needs a model and tokenizer.")
        self.policy = Policy(cfg)

        self.device = cfg.device
        sel = cfg.selection
        self.embed_model_name = sel.embed_model or None
        self.encode_batch = int(sel.encode_batch or 16)
        self.nn_chunk = int(sel.nn_chunk or 512)
        self.max_length = int(cfg.model.max_length)

        if getattr(self.model, "config", None) is not None:
            self.model.config.use_cache = False

    # ---- BaseSelector API --------------------------------------------------

    def select(self, train_dataset, val_dataset=None):
        if "text" not in train_dataset.column_names:
            raise ValueError(
                "MIWV needs a 'text' field on the dataset (the raw prompt); use "
                "a loader built on dataset.formatting (it keeps it)."
            )

        # Step 1: retrieve each sample's nearest neighbour as its one-shot example.
        emb = self._embed(train_dataset["text"])          # (N, d)
        nn_idx = self._nearest_neighbors(emb)             # (N,) int

        # Step 2: MIWV = L(y|x, C) - L(y|x) per sample.
        miwv = self._miwv_scores(train_dataset, nn_idx)

        valid = np.isfinite(miwv)
        print(f"[MIWV] scored {int(valid.sum())}/{len(miwv)} samples; "
              f"keeping the highest-MIWV (weakest-response) examples")

        # Unscorable examples (no response tokens) sink to the bottom.
        scores = np.where(valid, miwv, -np.inf)
        # Reuse the retrieval embeddings as policy features, so interaction-aware
        # policies (e.g. --policy diversity) can cover the feature space.
        return self.apply_policy(scores.tolist(), features=emb.numpy())

    # ---- step 1: one-shot retrieval ---------------------------------------

    @torch.no_grad()
    def _embed(self, texts):
        """Prompt embeddings -> (N, d) CPU float tensor (model pooling by default)."""
        if self.embed_model_name:
            return self._embed_sentence_transformer(texts)

        self.model.eval()
        chunks = []
        for i in tqdm(range(0, len(texts), self.encode_batch), desc="MIWV embed"):
            batch = texts[i:i + self.encode_batch]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            out = self.model(**enc, output_hidden_states=True)
            pooled = mean_pool(out.hidden_states[-1], enc["attention_mask"])
            chunks.append(pooled.float().cpu())
        return torch.cat(chunks)

    def _embed_sentence_transformer(self, texts):
        """Embed with a dedicated sentence encoder (paper uses BAAI/bge-large-en)."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "--miwv-embed-model needs sentence-transformers "
                "(`pip install sentence-transformers`)."
            ) from e
        encoder = SentenceTransformer(self.embed_model_name, device=self.device)
        emb = encoder.encode(
            list(texts), batch_size=self.encode_batch, convert_to_numpy=True,
            show_progress_bar=True, normalize_embeddings=False,
        )
        return torch.from_numpy(emb).float()

    @torch.no_grad()
    def _nearest_neighbors(self, emb):
        """Index of the most cosine-similar other sample for each row (chunked)."""
        emb = F.normalize(emb.to(self.device), dim=1)
        n = emb.shape[0]
        nn = torch.empty(n, dtype=torch.long)
        for start in tqdm(range(0, n, self.nn_chunk), desc="MIWV retrieval"):
            end = min(start + self.nn_chunk, n)
            sims = emb[start:end] @ emb.T                 # (chunk, N) cosine sim
            # Exclude self-matches before taking the argmax.
            rows = torch.arange(end - start, device=self.device)
            sims[rows, torch.arange(start, end, device=self.device)] = -float("inf")
            nn[start:end] = sims.argmax(dim=1).cpu()
        return nn.numpy()

    # ---- step 2: MIWV scoring ----------------------------------------------

    @torch.no_grad()
    def _miwv_scores(self, dataset, nn_idx):
        """MIWV = L(y|x, C) - L(y|x), two forward passes per example."""
        self.model.eval()
        scores = np.empty(len(dataset), dtype=np.float64)
        for i in tqdm(range(len(dataset)), desc="MIWV scoring"):
            example = dataset[i]
            base = self._response_loss(example["input_ids"], example["labels"])
            if base is None:
                scores[i] = np.nan
                continue
            shot = dataset[int(nn_idx[i])]
            ids, labels = self._one_shot_inputs(shot, example)
            oneshot = self._response_loss(ids, labels)
            scores[i] = np.nan if oneshot is None else oneshot - base
        return scores

    def _one_shot_inputs(self, shot, example):
        """Prepend the neighbour's full pair C as context to x_i (response masked).

        ``shot["input_ids"]`` already holds C = Prompt(x_k, y_k) (prompt + response,
        terminated by EOS), so it serves directly as the one-shot context. If the
        concatenation would exceed ``max_length`` the context is trimmed from the
        left, keeping x_i and its response intact so the loss stays comparable.
        """
        ctx = list(shot["input_ids"])
        tgt_ids = list(example["input_ids"])
        tgt_labels = list(example["labels"])

        budget = self.max_length - len(tgt_ids)
        if budget <= 0:
            return tgt_ids, tgt_labels          # no room for context
        if len(ctx) > budget:
            ctx = ctx[len(ctx) - budget:]

        ids = ctx + tgt_ids
        labels = [-100] * len(ctx) + tgt_labels  # only x_i's response is scored
        return ids, labels

    def _response_loss(self, input_ids, labels):
        """Mean cross-entropy on the response tokens (everything else is -100)."""
        if all(l == -100 for l in labels):
            return None
        ids = batched(input_ids, self.device)
        out = self.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            labels=batched(labels, self.device),
        )
        return float(out.loss)


def add_args(parser):
    """Register MIWV-specific CLI arguments (loaded dynamically by utils.options)."""
    g = parser.add_argument_group("MIWV")
    g.add_argument("--miwv-embed-model", default=None, dest="selection.embed_model",
                   help="Sentence-transformer used for one-shot retrieval (e.g. "
                        "BAAI/bge-large-en-v1.5). Default: the model's own pooled "
                        "hidden states (no extra encoder).")
    g.add_argument("--miwv-encode-batch", type=int, default=16,
                   dest="selection.encode_batch",
                   help="Batch size for retrieval embedding extraction.")
    g.add_argument("--miwv-nn-chunk", type=int, default=512,
                   dest="selection.nn_chunk",
                   help="Query rows per chunk in the nearest-neighbour search "
                        "(controls peak memory of the similarity matrix).")
