"""ColBERT re-ranker and text fetcher as PyTerrier Transformers.

Usage in pipeline.py
---------------------
    from longeval_starter.rerank import ColBERTReranker, LazyTextFetcher

    pipeline = (
        bm25 % 100
        >> LazyTextFetcher(dataset, text_fields)
        >> ColBERTReranker("./colbert-longeval")
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd
import pyterrier as pt
import torch
from pylate.models import ColBERT as ColBERTModel

log = logging.getLogger(__name__)


class LazyTextFetcher(pt.Transformer):
    """Fetch document text from an ir_datasets corpus by docno.

    Iterates the corpus once on the first ``transform()`` call and caches
    a docno→text map in memory for all subsequent calls.

    Parameters
    ----------
    dataset:
        Raw ir_datasets dataset (as returned by ``load_snapshot``).
    text_fields:
        Fallback field names if the document has no ``default_text()`` method.
    """

    def __init__(self, dataset, text_fields: list[str]) -> None:
        self._dataset = dataset
        self._text_fields = text_fields
        self._doc_map: dict[str, str] | None = None

    def _build_map(self) -> None:
        log.info("LazyTextFetcher: building docno→text map (one-time corpus scan)…")
        doc_map: dict[str, str] = {}
        for doc in self._dataset.docs_iter():
            if hasattr(doc, "default_text"):
                text = doc.default_text() or ""
            else:
                text = " ".join(
                    str(getattr(doc, f, "") or "") for f in self._text_fields
                )
            doc_map[str(doc.doc_id)] = text.strip()
        self._doc_map = doc_map
        log.info("LazyTextFetcher: cached %d documents", len(doc_map))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._doc_map is None:
            self._build_map()
        df = df.copy()
        df["text"] = df["docno"].map(self._doc_map).fillna("")
        return df


class ColBERTReranker(pt.Transformer):
    """Re-rank candidates using a PyLate ColBERT model.

    Expects the input DataFrame to have a ``text`` column (document text).
    Use ``pt.text.get_text(dataset, "text")`` upstream to hydrate it.

    Parameters
    ----------
    model_path:
        HuggingFace model name or local path to a saved ColBERT model.
    batch_size:
        Number of documents encoded per forward pass.
    device:
        ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        *,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        log.info("Loading ColBERT model from %s on %s", self.model_path, self.device)
        self.model = ColBERTModel(model_name_or_path=self.model_path, device=self.device)
        self.model.eval()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if "text" not in df.columns:
            raise ValueError(
                "ColBERTReranker requires a 'text' column. "
                "Add `>> pt.text.get_text(dataset, 'text')` before this transformer."
            )

        results: list[dict] = []

        for qid, group in df.groupby("qid", sort=False):
            query = str(group["query"].iloc[0])
            doc_texts = group["text"].tolist()
            docnos = group["docno"].tolist()

            scores = self._score(query, doc_texts)

            for docno, score in zip(docnos, scores):
                results.append({"qid": qid, "docno": docno, "score": score})

        if not results:
            return df.assign(score=float("nan"), rank=0)

        out = pd.DataFrame(results)
        out = out.merge(
            df.drop(columns=["score"], errors="ignore"),
            on=["qid", "docno"],
            how="left",
        )
        out = out.sort_values(["qid", "score"], ascending=[True, False])
        out["rank"] = out.groupby("qid").cumcount()
        return out.reset_index(drop=True)

    @torch.inference_mode()
    def _score(self, query: str, doc_texts: list[str]) -> list[float]:
        # encode returns list[Tensor] with variable seq lengths
        q_embs: list[torch.Tensor] = self.model.encode(
            [query],
            is_query=True,
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=False,
        )
        d_embs: list[torch.Tensor] = self.model.encode(
            doc_texts,
            is_query=False,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=False,
        )

        q = q_embs[0]  # [q_len, dim]
        return [_maxsim(q, d).item() for d in d_embs]


def _maxsim(q_emb: torch.Tensor, d_emb: torch.Tensor) -> torch.Tensor:
    """Compute ColBERT MaxSim score for one query–document pair.

    Parameters
    ----------
    q_emb: ``[q_len, dim]``
    d_emb: ``[d_len, dim]``
    """
    sim = torch.einsum("qd,ld->ql", q_emb, d_emb)  # [q_len, d_len]
    return sim.max(dim=1).values.sum()              # scalar
