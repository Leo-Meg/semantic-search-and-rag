# Semantic search, from the inverted index to RAG

Taking up my first NLP program, a search engine, and giving it the evaluation it lacked.

**Léo Mégret**, MSc Computational Linguistics, Université Paris Cité

> **Repository status, version 2.** I am doing this work in stages, each in its
> own folder. I publish as I go rather than once everything is finished.

---

## Why this repository

In my third undergraduate year, my first real NLP program was a search engine,
with an inverted index, a few preprocessing variants and TF-IDF weighting. In my
final Master's year, my last lab was a complete RAG pipeline, with a vector store
and sentence embeddings.

Three years separate those two pieces of work and the problem is the same,
finding the relevant information in a collection. What changed between them is the
representation.

I am rebuilding the chain from the beginning, measuring at each step what is
gained and what is lost.

---

## Published versions

| | Folder | Contents | Tests |
|---|---|---|---:|
| **1** | `1.rag_python_projet` | Lexical search, inverted index, TF-IDF, BM25 | 24 |
| **2** | `2.rag_python_projet` | Latent semantic analysis, the first search by meaning | 18 |

That is **42 tests** in total. Each folder contains everything the previous
one had, plus one step.

---

## Running the latest version

```bash
cd 2.rag_python_projet
python -m src.semantique
python -m tests.test_semantique
```

---

## What is still open

LSA recovers part of the failures, but its query projection stays lexical, and
two queries out of six have no known term at all.

Going further would need a representation learned on a corpus much larger than
mine. I do not yet know whether that is enough.

---

## How I work

Four rules I set myself at the start, and intend to keep across the whole
repository.

**Nothing to download.** The corpus is written into the code. All of my Master's
notebooks began with a `wget` to a university server or a Google Drive mount. Two
years later, half of them no longer run.

**Nothing is claimed without a measurement.** Every figure in this file
corresponds to a command you can re-run.

**Mistakes in my coursework are quoted, not erased.** Where a result I handed in
was wrong or incomplete, I say so and give the correct one.

**Negative results stay.** When an experiment shows the opposite of what I
expected, I write down what I found.

**The code is commented in French.**

---

---

*French version, which I wrote first, [README_FR.md](README_FR.md).*
