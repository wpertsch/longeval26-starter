"""Run the retrieval pipeline on a snapshot and write a TREC run file.

Output layout matches what TIRA expects for LongEval submissions::

    runs/
    ├── snapshot-1/run.txt.gz
    ├── snapshot-2/run.txt.gz
    └── snapshot-3/run.txt.gz

Just zip ``runs/`` together with ``submission/ir-metadata.yml`` and
upload it on TIRA.
"""

from __future__ import annotations

import logging

import pyterrier as pt

from longeval_starter.config import Config
from longeval_starter.data import load_snapshot, load_topics
from longeval_starter.index import build_or_load_index
from longeval_starter.pipeline import build_pipeline


log = logging.getLogger(__name__)


def retrieve(cfg: Config, snapshot: str):
    dataset = load_snapshot(cfg, snapshot)
    topics = load_topics(dataset)

    index_ref = build_or_load_index(cfg, snapshot)
    pipeline = build_pipeline(cfg, index_ref)

    log.info("Retrieving %d topics on %s", len(topics), snapshot)
    results = pipeline.transform(topics)

    out_path = cfg.run_path(snapshot)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Writing TREC run → %s (run_name=%s)", out_path, cfg.run_name)
    pt.io.write_results(
        results,
        str(out_path),
        format="trec",
        run_name=cfg.run_name,
    )

    # Tiny sanity summary so students see at a glance that the run isn't empty.
    n_q = results["qid"].nunique()
    n_docs = len(results)
    avg = n_docs / max(n_q, 1)
    log.info(
        "Wrote %d rankings for %d queries (avg %.1f docs/query).",
        n_docs, n_q, avg,
    )
    return out_path
