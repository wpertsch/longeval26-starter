# LongEval 2026 — Task 1 (Sci-Retrieval) — BM25 Starter

A minimal **PyTerrier + BM25** baseline for
[LongEval 2026 Task 1 (LongEval-Sci)](https://clef-longeval.github.io/tasks/).
Built as a teaching skeleton for the *Advanced Information Retrieval* course.

The code is intentionally small. It walks through the full pipeline once
(loading → indexing → retrieving → evaluating → writing TREC runs) and
leaves a single, clearly marked **extension point** where you build your
own contribution on top of BM25.

---

## 1. What LongEval-Sci asks of you

You are given three snapshots of a scientific-document corpus (CORE),
growing over time:

| Snapshot          | Period                   | Queries  | Qrels (pseudo)          |
|-------------------|--------------------------|----------|--------------------------|
| `snapshot-1`      | March – May 2025         | 100 train + test queries | click-based (`raw`, `dctr`) |
| `snapshot-2`      | June – August 2025       | test queries             | click-based              |
| `snapshot-3`      | September – November 2025| test queries             | click-based              |

Your job: for **each** snapshot, rank its documents for each test query
and submit one TREC run file per snapshot to
[TIRA.io](https://www.tira.io/task-overview/longeval-2026).

---

## 2. Prerequisites

- Python 3.10+
- Java 11+ (PyTerrier/Terrier needs a JVM — `java -version` to check)
- ~50 GB free disk space for the full corpora + indexes

## 3. Install

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verify:

```bash
python -c "import pyterrier as pt; print(pt.__version__)"
ir_datasets_longeval list | grep longeval-sci-2026
```

## 4. Quick start — run the whole pipeline

```bash
# 1. Build an index for each snapshot (first run downloads the corpus — this is slow!)
python -m longeval_starter index --snapshot snapshot-1
python -m longeval_starter index --snapshot snapshot-2
python -m longeval_starter index --snapshot snapshot-3

# 2. Sanity-check on the training qrels of snapshot-1
python -m longeval_starter evaluate --snapshot snapshot-1

# 3. Produce the TREC run files that go into the TIRA submission
python -m longeval_starter retrieve --snapshot snapshot-1
python -m longeval_starter retrieve --snapshot snapshot-2
python -m longeval_starter retrieve --snapshot snapshot-3

# Or do everything in one go:
make all
```

Run files land in `runs/<snapshot>/run.txt.gz`, which is the exact layout
TIRA expects (see `submission/` for the `ir-metadata.yml` template).

## 5. Where to make your changes

Everything data-related (loading, iterating, building the index) is
already implemented — you should **not** need to touch it.

The only file you are expected to modify is:

> **`longeval_starter/pipeline.py`**

It defines `build_pipeline(index) -> pt.Transformer`. Right now it
returns a plain BM25 retriever. Replace or wrap it with whatever you want
to evaluate: stemming variants, query expansion (Bo1, RM3, …), a
learned sparse model, a dense retriever, a re-ranker, a hybrid, a
longitudinal approach that uses `dataset.get_prior_datasets()`, …

Ideas to get started:

- **Query expansion.** `bm25 >> pt.rewrite.Bo1QueryExpansion(index) >> bm25`
- **Re-ranking.** `bm25 % 100 >> pt.text.get_text(dataset, "text") >> your_reranker`
- **Longitudinal re-use.** Use qrels from an earlier snapshot to boost
  documents that have been clicked before (see `data.load_snapshot`
  — each snapshot exposes its prior snapshots via
  `dataset.get_prior_datasets()`).

When you change `pipeline.py`, give your approach a name in
`config.yaml` (`run.name`) so your run files and `ir-metadata.yml`
stay in sync.

## 6. Project layout

```
longeval26-starter/
├── README.md                     ← you are here
├── requirements.txt
├── config.yaml                   ← dataset IDs, BM25 params, output paths
├── Makefile                      ← convenience targets
├── longeval_starter/
│   ├── __init__.py
│   ├── __main__.py               ← enables `python -m longeval_starter ...`
│   ├── config.py                 ← loads config.yaml
│   ├── data.py                   ← wraps ir_datasets_longeval
│   ├── index.py                  ← builds a Terrier index from a snapshot
│   ├── pipeline.py               ← *** THE EXTENSION POINT ***
│   ├── evaluate.py               ← pt.Experiment on train qrels
│   └── cli.py                    ← argparse entry point
├── indexes/                      ← created on first `index` call
├── runs/                         ← TREC run files end up here
└── submission/
    └── ir-metadata.yml           ← fill in and upload to TIRA
```

## 7. FAQ

**PyTerrier complains that Java is missing.**
Install a JDK (`sudo apt install openjdk-17-jdk` or `brew install openjdk`)
and make sure `JAVA_HOME` is set.

**Indexing runs out of memory.**
Lower `index.threads` in `config.yaml` to 1–2 and/or pass
`-Xmx4g` via the `TERRIER_HEAP` environment variable.

**Where does ir-datasets store the corpus?**
Default is `~/.ir_datasets/`. Set `IR_DATASETS_HOME` if you need to
put it on another disk.

**Do I need to resubmit for every snapshot change?**
No — TIRA wants one submission containing run files for all three
snapshots. Just zip `runs/` together with `submission/ir-metadata.yml`.

---

Happy hacking, and may your nDCG@10 be ever increasing.
