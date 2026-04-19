"""Thin wrapper around ``ir_datasets_longeval``.

This module is the only place the rest of the code talks to the
LongEval corpus — everything downstream (indexing, retrieval, eval)
operates on plain dicts / pandas DataFrames.

If you want to use the LongEval-specific extras (``get_timestamp``,
``get_prior_datasets``), call :func:`load_snapshot` and work with the
returned ``dataset`` object directly.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Iterator

import pandas as pd

from ir_datasets_longeval import load as longeval_load

from longeval_starter.config import Config


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_snapshot(cfg: Config, snapshot: str):
    """Return the raw LongEval ``ir_dataset`` object for a snapshot.

    The returned object is a standard ``ir_datasets`` dataset with two
    extra methods provided by the LongEval extension:

    - ``dataset.get_timestamp()``
    - ``dataset.get_prior_datasets()``

    Use those to build longitudinal approaches (e.g. boost documents
    that were clicked in a previous snapshot).
    """
    dsid = cfg.dataset_id(snapshot)
    log.info("Loading dataset %s", dsid)
    return longeval_load(dsid)


def load_snapshot_with_qrels(cfg: Config, snapshot: str):
    """Same as :func:`load_snapshot` but forces the qrels variant from config."""
    dsid = cfg.dataset_id(snapshot, with_qrels=True)
    log.info("Loading dataset %s", dsid)
    return longeval_load(dsid)


# ---------------------------------------------------------------------------
# Iterators that speak the PyTerrier / pandas dialect
# ---------------------------------------------------------------------------

def iter_docs(dataset, text_fields: Iterable[str]) -> Iterator[dict[str, str]]:
    """Yield ``{'docno': ..., 'text': ...}`` dicts for PyTerrier's IterDictIndexer.

    Uses the document's ``default_text()`` method if it exists (the
    ir-datasets convention), otherwise concatenates the given fields.
    """
    fields = list(text_fields)
    first = True
    for doc in dataset.docs_iter():
        if first:
            log.info("Document type: %s", type(doc).__name__)
            log.info("Document fields: %s", getattr(doc, "_fields", "n/a"))
            first = False

        text = _extract_text(doc, fields)
        yield {"docno": str(doc.doc_id), "text": text}


def _extract_text(doc: Any, fields: list[str]) -> str:
    if hasattr(doc, "default_text"):
        try:
            return str(doc.default_text() or "")
        except Exception:  # pragma: no cover -- be forgiving on odd doc types
            pass
    parts: list[str] = []
    for field in fields:
        value = getattr(doc, field, None)
        if value:
            parts.append(str(value))
    return " ".join(parts)


def load_topics(dataset) -> pd.DataFrame:
    """Return the queries as a PyTerrier topics DataFrame.

    Columns: ``qid``, ``query``.

    The query string is stripped of Terrier-unfriendly punctuation so
    that the BM25 pipeline does not crash on parentheses, colons etc.
    If you pre-process queries differently, do it here.
    """
    rows = [
        {"qid": str(q.query_id), "query": _sanitize_query(q.text)}
        for q in dataset.queries_iter()
    ]
    df = pd.DataFrame(rows)
    # Drop empty queries defensively — Terrier errors out on empty input.
    df = df[df["query"].str.strip().astype(bool)].reset_index(drop=True)
    log.info("Loaded %d topics", len(df))
    return df


def load_qrels(dataset) -> pd.DataFrame:
    """Return the qrels as a PyTerrier qrels DataFrame.

    Columns: ``qid``, ``docno``, ``label``.
    """
    rows = [
        {
            "qid": str(q.query_id),
            "docno": str(q.doc_id),
            "label": int(q.relevance),
        }
        for q in dataset.qrels_iter()
    ]
    df = pd.DataFrame(rows)
    log.info("Loaded %d qrels", len(df))
    return df


_TERRIER_UNSAFE = str.maketrans({c: " " for c in "()[]{}:\"'/"})


def _sanitize_query(text: str) -> str:
    return " ".join(text.translate(_TERRIER_UNSAFE).split())
