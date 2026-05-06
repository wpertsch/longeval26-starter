"""Train a ColBERT model on LongEval 2026 training data using PyLate.

Pipeline
--------
1. Load training qrels from ``snapshot-1/train/dctr``
2. Mine hard negatives via BM25 against the snapshot-1 index
3. Build training triples (query_text, positive_text, negative_text)
4. Fine-tune a ColBERT model with PyLate
5. Save model to ``./colbert-longeval``

Usage
-----
    python train_colbert.py --max-steps 200  # quick smoke test with 200 training steps
    python train_colbert.py --negs 10 --epochs 3   # number of negatives / training lenght
    python train_colbert.py --model-name colbert-ir/colbertv2.0 --negs 10 --epochs 3  # start from ColBERT v2 weights
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import pandas as pd
import pyterrier as pt
from datasets import Dataset
from pylate import losses, models
from pylate.utils.collator import ColBERTCollator
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

from longeval_starter.config import load_config
from longeval_starter.data import load_snapshot, load_snapshot_with_qrels, load_topics, load_qrels

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hard-negative mining
# ---------------------------------------------------------------------------

def mine_hard_negatives(
    topics: pd.DataFrame,
    qrels: pd.DataFrame,
    index_ref,
    *,
    depth: int = 100,
    negs_per_query: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Return a list of {query, positive, negative} dicts.

    Positives are randomly sampled from docs with qrels label >= 1.
    Negatives are sampled from the BM25 top-``depth`` that are NOT in qrels
    (or have label 0), giving harder negatives than random corpus samples.
    """
    rng = random.Random(seed)

    relevant = qrels[qrels["label"] >= 1].groupby("qid")["docno"].apply(set).to_dict()
    irrelevant_or_unknown = qrels[qrels["label"] == 0].groupby("qid")["docno"].apply(set).to_dict()

    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25", num_results=depth)
    retrieved = bm25.transform(topics)

    doc_text_map: dict[str, str] = {}

    triples: list[dict] = []
    for _, row in topics.iterrows():
        qid = str(row["qid"])
        query = str(row["query"])

        pos_docnos = list(relevant.get(qid, set()))
        if not pos_docnos:
            continue

        neg_candidates = (
            retrieved[retrieved["qid"] == qid]["docno"]
            .loc[lambda s: ~s.isin(relevant.get(qid, set()))]
            .tolist()
        )
        if not neg_candidates:
            neg_candidates = list(irrelevant_or_unknown.get(qid, set()))
        if not neg_candidates:
            continue

        num_neg = min(negs_per_query, len(neg_candidates))
        sampled_neg = rng.sample(neg_candidates, num_neg)

        for neg_docno in sampled_neg:
            pos_docno = rng.choice(pos_docnos)  # sample with replacement
            triples.append({
                "query": query,
                "_pos_docno": pos_docno,
                "_neg_docno": neg_docno,
            })

    log.info("Mined %d raw triples from %d queries", len(triples), len(topics))
    return triples


def hydrate_triples(
    triples: list[dict],
    dataset,
    text_fields: list[str],
) -> list[dict]:
    """Fetch document text for each triple's pos/neg docno."""
    log.info("Building docno→text map (iterating corpus)…")
    needed = {t["_pos_docno"] for t in triples} | {t["_neg_docno"] for t in triples}
    doc_map: dict[str, str] = {}

    for doc in dataset.docs_iter():
        docno = str(doc.doc_id)
        if docno not in needed:
            continue
        if hasattr(doc, "default_text"):
            text = doc.default_text() or ""
        else:
            text = " ".join(str(getattr(doc, f, "") or "") for f in text_fields)
        doc_map[docno] = text.strip()
        if len(doc_map) == len(needed):
            break

    hydrated = []
    for t in triples:
        pos_text = doc_map.get(t["_pos_docno"], "")
        neg_text = doc_map.get(t["_neg_docno"], "")
        if pos_text and neg_text:
            hydrated.append({"query": t["query"], "positive": pos_text, "negative": neg_text})

    log.info("%d triples after hydration", len(hydrated))
    return hydrated


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    output_dir: str,
    *,
    model_name: str = "bert-base-uncased",
    negs_per_query: int = 5,
    depth: int = 100,
    epochs: int = 1,
    batch_size: int = 16,
    lr: float = 3e-5,
    max_steps: int = -1,
    seed: int = 42,
) -> None:
    cfg = load_config()

    # --- data ----------------------------------------------------------------
    log.info("Loading training dataset (snapshot-1/train/dctr)…")
    train_ds = load_snapshot_with_qrels(cfg, "snapshot-1", train=True)
    topics = load_topics(train_ds)
    qrels = load_qrels(train_ds)
    log.info("%d topics, %d qrels entries", len(topics), len(qrels))

    # --- index ---------------------------------------------------------------
    index_path = cfg.index_path("snapshot-1")
    props = index_path / "data.properties"
    if not props.exists():
        raise FileNotFoundError(
            f"Index not found at {index_path}. "
            "Run `python -m longeval_starter index --snapshot snapshot-1` first."
        )
    pt.java.init()
    index_ref = pt.IndexRef.of(str(props.resolve()))

    # --- hard negatives ------------------------------------------------------
    text_fields = list(cfg.raw["index"].get("text_fields", ["title", "abstract", "text"]))
    raw_triples = mine_hard_negatives(
        topics, qrels, index_ref,
        depth=depth,
        negs_per_query=negs_per_query,
        seed=seed,
    )

    snap_ds = load_snapshot(cfg, "snapshot-1")
    triples = hydrate_triples(raw_triples, snap_ds, text_fields)

    if not triples:
        raise RuntimeError("No training triples could be built — check qrels and index.")

    hf_dataset = Dataset.from_list(triples)
    log.info("HuggingFace dataset: %s", hf_dataset)

    # --- model ---------------------------------------------------------------
    log.info("Loading ColBERT model: %s", model_name)
    model = models.ColBERT(model_name_or_path=model_name)

    # --- training ------------------------------------------------------------
    loss = losses.Contrastive(model=model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        fp16=False,
        bf16=False,
        seed=seed,
        save_strategy="epoch",
        logging_steps=50,
        **({"max_steps": max_steps} if max_steps > 0 else {}),
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset,
        loss=loss,
        data_collator=ColBERTCollator(tokenize_fn=model.tokenize),
    )

    log.info("Starting training…")
    trainer.train()

    log.info("Saving model to %s", output_dir)
    model.save_pretrained(output_dir)
    log.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a ColBERT model on LongEval 2026")
    p.add_argument("--output-dir", default="./colbert-longeval")
    p.add_argument("--model-name", default="bert-base-uncased",
                   help="HuggingFace model name or path (e.g. colbert-ir/colbertv2.0)")
    p.add_argument("--negs", dest="negs_per_query", type=int, default=5,
                   help="Hard negatives per query")
    p.add_argument("--depth", type=int, default=100,
                   help="BM25 retrieval depth for neg mining")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Stop after N steps (useful for quick smoke-tests, -1 = full run)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        output_dir=args.output_dir,
        model_name=args.model_name,
        negs_per_query=args.negs_per_query,
        depth=args.depth,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        seed=args.seed,
    )
