"""BM25 scorer -- lexical relevance to a target set (axes ①②③, model-free).

A naive, model-free scorer that rates every training example by its lexical
relevance to a *target* set using Okapi BM25 -- the classic retrieval ranking
function. This mirrors retrieval-based data selection (e.g. BM25 retrieval in
DSIR): documents whose words best match the task we care about score highest.

    score(d) = aggregate over queries q in target of  BM25(q, d)

with the standard formula over the training corpus statistics:

    BM25(q, d) = sum_{t in q} IDF(t) * f(t,d)*(k1+1)
                                       / ( f(t,d) + k1*(1 - b + b*|d|/avgdl) )

The target/query set (② Comparison Target) is the validation set when one is
available; otherwise we fall back to a random sample of the training set itself,
so the score becomes a self-similarity / centrality measure (documents that look
like the rest of the corpus rank highest).

Operates purely on the raw prompt ``text`` field (see dataset/formatting.py), so
no model and no gradients are involved (③ = none). Used by ``alg/bm25.py``.
"""

import math
import re
from collections import Counter

import numpy as np

from scorer.base import BaseScorer
from utils.selector_utils import tqdm

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())


class _BM25:
    """Okapi BM25 over a fixed document corpus (dependency-free)."""

    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = corpus_tokens
        self.doc_freqs = [Counter(toks) for toks in corpus_tokens]
        self.doc_len = np.array([len(toks) for toks in corpus_tokens], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0

        n = len(corpus_tokens)
        df = Counter()
        for toks in corpus_tokens:
            df.update(set(toks))
        # Okapi IDF with the +1 form so weights stay non-negative.
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query_tokens):
        """BM25 of one query against every corpus document -> (N,) array."""
        out = np.zeros(len(self.docs), dtype=np.float64)
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue  # term unseen in the corpus -> zero contribution
            for i, freqs in enumerate(self.doc_freqs):
                f = freqs.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out


class Scorer(BaseScorer):
    def __init__(self, cfg, model=None, tokenizer=None):
        super().__init__(cfg, model, tokenizer)
        sel = cfg.selection
        self.k1 = float(sel.bm25_k1 or 1.5)
        self.b = float(sel.bm25_b if sel.bm25_b is not None else 0.75)
        self.aggregation = sel.bm25_aggregation or "mean"
        self.max_queries = int(sel.bm25_max_queries or 256)

    # ---- BaseScorer API ----------------------------------------------------

    def score(self, train_dataset, val_dataset=None):
        if "text" not in train_dataset.column_names:
            raise ValueError(
                "BM25 needs a 'text' field on the dataset; use a loader built on "
                "dataset.formatting (it keeps the raw prompt text)."
            )

        corpus = [_tokenize(t) for t in tqdm(train_dataset["text"], desc="BM25 tokenize")]
        bm25 = _BM25(corpus, k1=self.k1, b=self.b)

        queries = self._query_tokens(train_dataset, val_dataset, corpus)
        print(f"[BM25] scoring {len(corpus)} docs against {len(queries)} queries "
              f"({self.aggregation} aggregation)")

        agg = np.zeros(len(corpus), dtype=np.float64) if self.aggregation == "mean" \
            else np.full(len(corpus), -np.inf)
        for q in tqdm(queries, desc="BM25 scoring"):
            s = bm25.scores(q)
            agg = agg + s if self.aggregation == "mean" else np.maximum(agg, s)
        if self.aggregation == "mean" and queries:
            agg /= len(queries)

        return agg, None

    # ---- query set ---------------------------------------------------------

    def _query_tokens(self, train_dataset, val_dataset, corpus):
        """Validation prompts as queries, else a random sample of train docs."""
        if val_dataset is not None and len(val_dataset) > 0 \
                and "text" in val_dataset.column_names:
            return [_tokenize(t) for t in val_dataset["text"]]

        n = len(corpus)
        rng = np.random.default_rng(self.cfg.seed)
        m = min(self.max_queries, n)
        idx = rng.choice(n, size=m, replace=False)
        return [corpus[i] for i in idx]


def add_args(parser):
    """Register BM25-specific CLI arguments (loaded dynamically by utils.options)."""
    g = parser.add_argument_group("BM25")
    g.add_argument("--bm25-k1", type=float, default=1.5, dest="selection.bm25_k1",
                   help="BM25 term-frequency saturation parameter.")
    g.add_argument("--bm25-b", type=float, default=0.75, dest="selection.bm25_b",
                   help="BM25 length-normalization parameter.")
    g.add_argument("--bm25-aggregation", choices=["mean", "max"], default="mean",
                   dest="selection.bm25_aggregation",
                   help="How to aggregate BM25 relevance over the query set.")
    g.add_argument("--bm25-max-queries", type=int, default=256,
                   dest="selection.bm25_max_queries",
                   help="Random train docs used as queries when no validation "
                        "set is available (self-similarity fallback).")
