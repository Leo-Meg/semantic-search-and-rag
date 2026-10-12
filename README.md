# Semantic search, from the inverted index to RAG

Taking up a first NLP program, a search engine, and giving it the evaluation it lacked.

**Léo Mégret** — implementation, French technical notes.  
**Haeeul Hwang** — English documentation, Korean documentation, review.

> **Repository status** — step 2 of 3, landed by Léo Mégret.

---

## Why this repository

In his third undergraduate year, Léo's first real NLP program was a search engine,
with an inverted index, a few preprocessing variants and TF-IDF weighting. In his
final Master's year, his last lab — written with Haeeul — was a complete RAG
pipeline, with a vector store and sentence embeddings.

Three years separate those two pieces of work and the problem is the same:
finding the relevant information in a collection. What changed between them is
the representation.

We rebuild the chain from the beginning, measuring at each step what is gained
and what is lost.

---

## Steps

| | Step | Modules | Tests | Landed by | Detailed note (French) |
|---|---|---|---:|---|---|
| **1** | Lexical search: inverted index, TF-IDF, BM25 | `src/corpus.py`, `src/recherche.py` | 24 | Léo | [`index-inverse-et-bm25`](notes/index-inverse-et-bm25.md) |
| **2** | Latent semantic analysis, the first search by meaning | `src/semantique.py` | 18 | Léo | [`analyse-semantique-latente`](notes/analyse-semantique-latente.md) |

**42** tests in total.

---

## What each step found

**1. Lexical search: inverted index, TF-IDF, BM25**  
Overall both engines score about 78 % and look equivalent. Split the queries by difficulty and they are not: the average was hiding the only case that mattered.

**2. Latent semantic analysis, the first search by meaning**  
On lexically disjoint queries, the lexical engine scores 1.000 where terms overlap and 0.333 where they do not. SVD is where a search engine stops matching strings.

---

## Running it

```bash
python -m src.semantique

python -m tests.test_recherche
python -m tests.test_semantique
```

---

## How we work

**Nothing to download.** The corpus is written into the code. All of
Léo's Master's notebooks began with a `wget` to a university server or a Google
Drive mount. Two years later, half of them no longer run.

**Nothing is claimed without a measurement.** Every figure in this file
corresponds to a command you can re-run.

**Mistakes in the original coursework are quoted, not erased.** Where a result
that was handed in was wrong or incomplete, we say so and give the correct one.

**Negative results stay.** When an experiment shows the opposite of what we
expected, we write down what we found.

**The code is commented in French**, because that is the language it was thought
out in. The documentation is in English, and in Korean where Haeeul has caught
up.

---

*Français : [README_FR.md](README_FR.md) · [CONTRIBUTORS.md](CONTRIBUTORS.md)*
