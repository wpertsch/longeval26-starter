"""Evaluate the current pipeline on a snapshot's qrels.

Used mostly during development on ``snapshot-1`` (where train qrels are
released). Can also be pointed at snapshot-2 / snapshot-3 once pseudo
qrels are available.
"""

from __future__ import annotations

import logging

import ir_measures
import pyterrier as pt

from longeval_starter.config import Config
from longeval_starter.data import load_qrels, load_snapshot_with_qrels, load_topics
from longeval_starter.index import build_or_load_index
from longeval_starter.pipeline import build_pipeline


log = logging.getLogger(__name__)


def evaluate(cfg: Config, snapshot: str):
    dataset = load_snapshot_with_qrels(cfg, snapshot)
    topics = load_topics(dataset)
    qrels = load_qrels(dataset)

    if qrels.empty:
        raise RuntimeError(
            f"No qrels available for {snapshot!r} — try a snapshot with "
            "released relevance judgments (typically snapshot-1 during "
            "development)."
        )

    index_ref = build_or_load_index(cfg, snapshot)
    pipeline = build_pipeline(cfg, index_ref)

    metrics = [ir_measures.parse_measure(m) for m in cfg.raw["evaluation"]["metrics"]]

    log.info("Running pt.Experiment on %s (%d topics, %d qrels)",
             snapshot, len(topics), len(qrels))

    result = pt.Experiment(
        [pipeline],
        topics,
        qrels,
        eval_metrics=metrics,
        names=[cfg.run_name],
        # Only evaluate queries for which we actually have qrels —
        # otherwise metrics are diluted by unjudged queries.
        filter_by_qrels=True,
    )
    print(result.to_string(index=False))
    return result
