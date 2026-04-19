"""Build (or load) a Terrier index for one LongEval snapshot."""

from __future__ import annotations

import logging
from pathlib import Path

import pyterrier as pt
from tqdm import tqdm

from longeval_starter.config import Config
from longeval_starter.data import iter_docs, load_snapshot


log = logging.getLogger(__name__)


def build_or_load_index(cfg: Config, snapshot: str) -> "pt.IndexRef":
    """Build a Terrier index for ``snapshot``; re-use it on subsequent runs.

    Indexes live under ``cfg.indexes_dir / <snapshot>``. If that directory
    already contains a ``data.properties`` file the index is just loaded —
    re-indexing a ~million-document snapshot is not something you want
    to do by accident.
    """
    index_path = cfg.index_path(snapshot)
    props = index_path / "data.properties"

    if props.exists():
        log.info("Re-using existing index at %s", index_path)
        return pt.IndexRef.of(str(props.resolve()))

    index_path.mkdir(parents=True, exist_ok=True)

    dataset = load_snapshot(cfg, snapshot)
    text_fields = cfg.raw["index"]["text_fields"]
    threads = int(cfg.raw["index"].get("threads", 1))

    log.info("Indexing %s → %s (threads=%d)", snapshot, index_path, threads)

    indexer = pt.IterDictIndexer(
        str(index_path.resolve()),
        threads=threads,
        meta={"docno": 32},
        # We only store the docno in meta — documents themselves stay on
        # disk inside ir-datasets. If you want to re-rank with document
        # text later, see the hint in pipeline.py.
    )

    # Try to pass docs_count so tqdm shows a real progress bar.
    try:
        total = dataset.docs_count()
    except Exception:
        total = None

    docs = iter_docs(dataset, text_fields)
    if total:
        docs = tqdm(docs, total=total, desc=f"index {snapshot}", unit="doc")

    index_ref = indexer.index(docs)
    log.info("Done. Index: %s", index_ref)
    return index_ref
