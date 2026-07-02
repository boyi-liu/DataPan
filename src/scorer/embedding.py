"""Embedding-similarity scorer -- representation proximity to an anchor (①②③).

A naive scorer that rates every training example by the cosine similarity of its
embedding to a *reference* embedding. Embeddings are mean-pooled last-layer
hidden states from the (frozen) model, so no extra encoder is required:

    emb(x) = meanpool_t  h_L(x)_t
    score(d) = cos( emb(d), reference )

The reference (② Comparison Target) is the centroid (mean embedding) of the
validation set when one is available -- selecting training data that looks like
the target task. With no validation set it falls back to the centroid of the
training set itself, turning the score into a representativeness measure. The
direction is a knob (``--embedding-select``):

    * ``near`` (default): the most similar examples score highest (on-target /
      representative).
    * ``far``: the least similar examples score highest (outliers / diversity).

The model weights stay frozen (a single forward pass per batch, no gradients).
Besides the scalar score this scorer also returns the per-example embeddings as
``features``, so interaction-aware policies (e.g. ``diversity``) can cover the
representation space. Operates on the raw prompt ``text`` field. Used by
``alg/embedding.py``.
"""

import torch
import torch.nn.functional as F

from scorer.base import BaseScorer
from utils.selector_utils import mean_pool, tqdm


class Scorer(BaseScorer):
    needs_model = True

    def __init__(self, cfg, model=None, tokenizer=None):
        super().__init__(cfg, model, tokenizer)
        self.device = cfg.device
        sel = cfg.selection
        self.direction = sel.embedding_select or "near"
        if self.direction not in ("near", "far"):
            raise ValueError("embedding_select must be 'near' or 'far'.")
        self.encode_batch = int(sel.encode_batch or 16)

        if getattr(self.model, "config", None) is not None:
            self.model.config.use_cache = False

    # ---- BaseScorer API ----------------------------------------------------

    def score(self, train_dataset, val_dataset=None):
        if "text" not in train_dataset.column_names:
            raise ValueError(
                "Embedding needs a 'text' field on the dataset; use a loader "
                "built on dataset.formatting (it keeps the raw prompt text)."
            )

        emb = self._encode(train_dataset["text"], desc="encode pool")  # (N, d)

        if val_dataset is not None and len(val_dataset) > 0 \
                and "text" in val_dataset.column_names:
            ref = self._encode(val_dataset["text"], desc="encode val").mean(0)
            source = "validation centroid"
        else:
            ref = emb.mean(0)
            source = "train centroid"
        print(f"[Embedding] scoring {emb.shape[0]} docs vs {source}; "
              f"keeping {self.direction} examples")

        sim = F.cosine_similarity(emb, ref.unsqueeze(0), dim=1)  # (N,)
        scores = sim if self.direction == "near" else -sim
        # Return the embeddings as features so interaction-aware policies (e.g.
        # --policy diversity) can cover the feature space; score-only policies
        # ignore them.
        return scores.numpy(), emb.numpy()

    # ---- encoding ----------------------------------------------------------

    @torch.no_grad()
    def _encode(self, texts, desc="encode"):
        """Mean-pooled last-layer hidden states as CPU float tensors -> (N, d)."""
        self.model.eval()
        chunks = []
        for i in tqdm(range(0, len(texts), self.encode_batch), desc=f"Embedding {desc}"):
            batch = texts[i:i + self.encode_batch]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=self.cfg.model.max_length,
            ).to(self.device)
            out = self.model(**enc, output_hidden_states=True)
            pooled = mean_pool(out.hidden_states[-1], enc["attention_mask"])
            chunks.append(pooled.float().cpu())
        return torch.cat(chunks)


def add_args(parser):
    """Register Embedding-specific CLI arguments (loaded dynamically by utils.options)."""
    g = parser.add_argument_group("Embedding")
    g.add_argument("--embedding-select", choices=["near", "far"], default="near",
                   dest="selection.embedding_select",
                   help="Keep examples most similar (near, default) or least "
                        "similar (far) to the reference centroid.")
    g.add_argument("--embedding-encode-batch", type=int, default=16,
                   dest="selection.encode_batch",
                   help="Batch size for embedding extraction.")
