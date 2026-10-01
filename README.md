# Semantic search — from the inverted index to RAG

Taking up my very first NLP program, a search engine, and finally giving it the evaluation it lacked.

**Léo Mégret** — MSc Computational Linguistics, Université Paris Cité

> **Repository status: version 1.** This is the first step of a piece of work I
> am doing in stages, each in its own folder. Only version 1 exists so far. I am
> publishing as I go rather than once everything is finished, because the point
> of this work is precisely the way one question leads to the next.

---

## Why this repository

In my third undergraduate year, my first real NLP program was a search engine: an
inverted index, a few preprocessing variants, TF-IDF weighting. In my final
Master's year, my last lab was a complete RAG pipeline, with a vector store and
sentence embeddings.

Three years separate those two pieces of work, and it is the same problem:
finding the relevant information in a collection. What changed between them is
not the evaluation metric, it is the representation.

I am rebuilding the chain from the beginning, measuring at each step what is
gained and what is lost.

---

## What exists today

### Version 1 — Lexical search: inverted index, TF-IDF, BM25

| File | What I do in it |
|---|---|
| `src/corpus.py` | 20 documents about NLP, 18 annotated queries sorted by difficulty, Unicode normalisation, justified stop words, a French stemmer written by hand. |
| `src/recherche.py` | `IndexInverse` with two IDF formulas, `RechercheTFIDF` with logarithmic TF and cosine, `RechercheBM25` with saturation and length normalisation, and the Hit Rate, MRR and MAP metrics. |
| `tests/test_recherche.py` | 24 tests, including a check that BM25 really does saturate frequency. |

Academic origin: a third-year undergraduate project, a search engine over a
corpus of articles, and the Master's lab *RAG: semantic search* (Florent Storme),
evaluated with Hit Rate and MAP on Quora Question Pairs.

---

## Running the code

```bash
cd 1.rag_python_projet
python -m src.corpus
python -m src.recherche
python -m tests.test_recherche
```

No dependencies, not even NumPy.

---

## What I take from this step

**Lexical search is perfect as long as the query shares words with the document,
and collapses as soon as it does not.** On my semantic queries, the ones that
share no term with the right answer, I measure 0.333.

**Sorting queries by difficulty changes everything.** As long as I was measuring
a global average, I saw nothing. It is by separating lexical queries from
semantic ones that the failure becomes legible, and locatable.

**Stemming has its own limits, and they are instructive.** The case of
`apprentissage` against `apprendre` cannot be solved by any truncation rule: it
would need dictionary-based lemmatisation.

---

## What is still open

Two thirds of the semantic queries fail, and the reason is always the same: no
shared term between the query and the document.

No amount of BM25 tuning can answer that. It is not the weighting that needs to
change, it is the representation. I do not yet know what to replace it with.

---

## How I work

Four rules I set myself at the start, and intend to keep across the whole
repository.

**Nothing to download.** The corpus lives in the code. All of my Master's
notebooks began with a `wget` to a university server or a Google Drive mount.
Two years later, half of them no longer run.

**Nothing is claimed without a measurement.** Every figure in this file
corresponds to a command you can re-run.

**Mistakes in my submitted coursework are quoted, not erased.** Where a result I
handed in was wrong or incomplete, I say so and give the correct one.

**Negative results stay.** When an experiment shows the opposite of what I
expected, I change the conclusion, not the experiment.

**The code is commented in French.** This is a repository meant to be read as
much as run.

---

*French version, and the one I wrote first: [README_FR.md](README_FR.md).*
