"""Retrieval pipeline.

**This is the file you change.** Every other module is plumbing —
loading data, building an index, writing run files, running ``pt.Experiment``.
:func:`build_pipeline` returns a single PyTerrier Transformer that gets
handed a set of topics and is expected to produce a ranking.

The default implementation is plain BM25 with the parameters from
``config.yaml``. Anything you build on top of that (query expansion,
re-ranking, dense retrieval, a longitudinal approach, …) goes here.

A few PyTerrier idioms worth knowing:

* **Composition.**   ``a >> b`` chains transformers — the output of ``a``
  becomes the input of ``b``.
* **Rank cutoff.**   ``retriever % 100`` keeps only the top-100 per query.
* **Union.**         ``a + b`` sums scores of two rankers (trivial hybrid).
* **Fetching text.** ``pt.text.get_text(dataset, "text")`` hydrates the
  pipeline with the actual document text — useful before a re-ranker.

See https://pyterrier.readthedocs.io/ for the full reference.
"""

from __future__ import annotations

import logging

from typing import Any

import pyterrier as pt

from longeval_starter.config import Config


log = logging.getLogger(__name__)

COLBERT_MODEL_PATH = "./colbert-longeval"


def build_pipeline(cfg: Config, index_ref: Any, snapshot: str = "snapshot-1") -> pt.Transformer:
    """Return the retrieval pipeline used for a snapshot.

    Parameters
    ----------
    cfg
        Parsed config — use it to read BM25 controls, cutoffs etc.
    index_ref
        The Terrier index for the *current* snapshot.

    Returns
    -------
    A PyTerrier transformer that maps a topics DataFrame
    (``qid``, ``query``) to a ranking DataFrame
    (``qid``, ``docno``, ``score``, ``rank``).
    """
    wmodel = cfg.raw["retrieval"]["wmodel"]
    num_results = int(cfg.raw["retrieval"]["num_results"])
    controls = dict(cfg.raw["retrieval"].get("controls") or {})

    log.info(
        "Building pipeline for %s: %s (num_results=%d, controls=%s)",
        snapshot, wmodel, num_results, controls,
    )

    bm25 = pt.terrier.Retriever(  # type: ignore[attr-defined]
        index_ref,
        wmodel=wmodel,
        num_results=num_results,
        controls={str(k): str(v) for k, v in controls.items()},
    )

    # ------------------------------------------------------------------
    # Pipeline 1 (active): BM25 + Bo1 query expansion
    # ------------------------------------------------------------------
    bm25_qe = bm25 >> pt.rewrite.Bo1QueryExpansion(index_ref) >> bm25
    # return bm25_qe

    # ------------------------------------------------------------------
    # Pipeline 2: BM25 first-stage + ColBERT re-ranker
    # Train the model first:  python train_colbert.py
    # Then swap the return above for the one below.
    # ------------------------------------------------------------------
    from longeval_starter.data import load_snapshot
    from longeval_starter.rerank import ColBERTReranker, LazyTextFetcher
    dataset = load_snapshot(cfg, snapshot)
    text_fields = list(cfg.raw["index"].get("text_fields", ["title", "abstract", "text"]))
    return (
        bm25 % 100
        >> LazyTextFetcher(dataset, text_fields)
        >> ColBERTReranker(COLBERT_MODEL_PATH)
    )
