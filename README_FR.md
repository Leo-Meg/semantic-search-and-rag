# Recherche sémantique, de l'index inversé au RAG

Reprendre un premier programme de TAL, un moteur de recherche, et lui donner l'évaluation qui lui manquait.

**Léo Mégret** — implémentation, notes techniques en français.  
**Haeeul Hwang** — documentation anglaise, documentation coréenne, relecture.

> **État du dépôt** — étape 2 sur 3, publiée par Léo Mégret.

---

## Pourquoi ce dépôt

En L3, mon premier programme sérieux de TAL était un moteur de recherche, avec un
index inversé, quelques variantes de prétraitement et une pondération TF-IDF. En
M2, mon TP final — écrit avec Haeeul — était un RAG complet, avec base
vectorielle et plongements de phrases.

Trois ans séparent ces deux travaux et le problème est le même : trouver
l'information pertinente dans une collection. Ce qui a changé entre les deux,
c'est la représentation.

Je reconstruis la chaîne depuis le début, en mesurant à chaque étape ce que l'on
gagne et ce que l'on perd.

---

## Les étapes

| | Étape | Modules | Tests | Publiée par | Note détaillée (français) |
|---|---|---|---:|---|---|
| **1** | Recherche lexicale : index inversé, TF-IDF, BM25 | `src/corpus.py`, `src/recherche.py` | 24 | Léo | [`index-inverse-et-bm25`](notes/index-inverse-et-bm25.md) |
| **2** | Analyse sémantique latente, la première recherche par le sens | `src/semantique.py` | 18 | Léo | [`analyse-semantique-latente`](notes/analyse-semantique-latente.md) |

**42** tests au total.

---

## Ce que chaque étape a trouvé

**1. Recherche lexicale : index inversé, TF-IDF, BM25**  
Globalement les deux moteurs font environ 78 % et semblent équivalents. En séparant les requêtes par difficulté, ils ne le sont pas : la moyenne cachait le seul cas qui comptait.

**2. Analyse sémantique latente, la première recherche par le sens**  
Sur des requêtes lexicalement disjointes, le moteur lexical obtient 1,000 quand les termes se recouvrent et 0,333 quand ils ne se recouvrent pas. La SVD est le moment où un moteur cesse de comparer des chaînes.

---

## Lancer le code

```bash
python -m src.semantique

python -m tests.test_recherche
python -m tests.test_semantique
```

---

## Comment nous travaillons

**Rien à télécharger.** Le corpus est écrit dans le code. Tous mes
notebooks de master commençaient par un `wget` vers un serveur universitaire ou
un montage de Google Drive ; deux ans plus tard, la moitié ne s'exécutent plus.

**Rien n'est affirmé sans mesure.** Chaque chiffre de ce fichier correspond à
une commande qu'on peut relancer.

**Les erreurs de mes rendus sont citées, pas effacées.** Quand un résultat que
j'avais rendu était faux ou incomplet, je le dis et je donne le bon.

**Les résultats négatifs restent.** Quand une expérience montre l'inverse de ce
que j'attendais, j'écris ce que j'ai trouvé.

**Le code est commenté en français**, parce que c'est la langue dans laquelle je
l'ai pensé. La documentation est en anglais, et en coréen quand Haeeul est
passée dessus.

---

*English: [README.md](README.md) · [CONTRIBUTORS.md](CONTRIBUTORS.md)*
