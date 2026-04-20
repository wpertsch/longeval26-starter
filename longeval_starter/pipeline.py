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


def build_pipeline(cfg: Config, index_ref: Any) -> pt.Transformer:
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
        "Building pipeline: %s (num_results=%d, controls=%s)",
        wmodel, num_results, controls,
    )

    bm25 = pt.terrier.Retriever(  # type: ignore[attr-defined]
        index_ref,
        wmodel=wmodel,
        num_results=num_results,
        controls={str(k): str(v) for k, v in controls.items()},
    )

    # ------------------------------------------------------------------
    # TODO for students: extend the pipeline below.
    # ------------------------------------------------------------------
    # Some inspirations:
    #
    #   # Query expansion with Bo1
    #   bm25_qe = bm25 >> pt.rewrite.Bo1QueryExpansion(index_ref) >> bm25
    #   return bm25_qe
    #
    #   # Re-ranking the top-100 of BM25
    #   from longeval_starter.data import load_snapshot
    #   dataset = load_snapshot(cfg, snapshot)           # pass snapshot in!
    #   text_pipeline = bm25 % 100 >> pt.text.get_text(dataset, "text")
    #   return text_pipeline >> YourReranker()
    #
    #   # Longitudinal boosting from prior snapshots
    #   dataset = load_snapshot(cfg, snapshot)
    #   for prior in dataset.get_prior_datasets():
    #       ...  # fetch prior qrels and boost
    #
    # Keep the return type as a Transformer with output columns
    # (qid, docno, score, rank) — the CLI and evaluator rely on that.
    # ------------------------------------------------------------------

    return bm25
