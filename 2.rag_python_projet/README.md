# Version 2, analyse sémantique latente. La première recherche par le sens

> **Où j'en suis.** La version 1 a mesuré l'échec. **0,333** de hit@5 sur les
> requêtes qui ne partagent aucun mot avec le document attendu. Cette version
> applique une SVD à la matrice termes-documents, et découvre à la fois ce que
> ça règle et ce que ça ne peut pas régler.

---

## Nouveautés par rapport à la version 1

| Fichier | Ce que j'ajoute |
|---|---|
| `src/semantique.py` | Matrice termes × documents (trois pondérations), `RechercheLSA` avec *folding-in* des requêtes, interprétation des dimensions latentes, balayage de `k`, et `RechercheHybride` par fusion de rangs réciproques. |
| `tests/test_semantique.py` | 18 tests, dont la reconstruction exacte de la matrice à rang plein. |

## Origine universitaire

- **Devoir « Topic analysis »** (Benoît Crabbé, bases formelles du TAL).
  matrice termes × documents en NumPy, TF-IDF, SVD, similarités cosinus dans
  l'espace réduit. L'énoncé insistait. *« On veillera à ne pas utiliser nltk ou
  scikit-learn. »*, contrainte que je respecte encore ici.
- **TP « Topic modelling, LSA and LDA »** (Timothée Bernard, M1). LSA et LDA
  sur les sous-titres de *Game of Thrones*, avec un jeu artificiel de contrôle.

## Lancer le code

```bash
python -m src.semantique
```

```bash
python -m tests.test_semantique
```

Dépendance. NumPy (pour `np.linalg.svd`).

---

## L'idée, en trois lignes

Au lieu de comparer les documents dans l'espace des **termes** (une dimension
par mot, matrice remplie à moins de 10 %), on les compare dans un espace réduit
obtenu par décomposition en valeurs singulières.

La SVD trouve les directions qui expliquent le plus de variance. Ces directions
regroupent les termes qui **co-occurrent**. Si « attention » et « transformer »
apparaissent souvent ensemble, ils sont projetés dans la même direction latente, même si aucun document ne les contient tous les deux.

C'est l'hypothèse distributionnelle appliquée à l'algèbre linéaire au lieu des
réseaux de neurones. LSA (Deerwester et al., 1990) précède word2vec de
**vingt-trois ans** et repose sur la même intuition.

Les dimensions apprises sont d'ailleurs lisibles.

```
dimension 1 : attention (+0.23), mécan (+0.19) | fine (−0.17), tuning (−0.17), lora (−0.17)
dimension 2 : transfert (−0.33), lingu (−0.27), cros (−0.27) | attention (+0.21)
```

---

## Résultat n°1, la variance expliquée est un mauvais critère

| k | var. expliquée | hit@5 | MRR | lexicale | partielle | sémantique |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 13,3 % | 0,389 | 0,243 | 0,667 | 0,333 | 0,167 |
| 4 | 25,4 % | 0,778 | 0,457 | 1,000 | 1,000 | 0,333 |
| 8 | 47,9 % | 0,778 | 0,676 | 1,000 | 1,000 | 0,333 |
| 12 | 67,7 % | 0,778 | 0,750 | 1,000 | 1,000 | 0,333 |
| **20** | **100 %** | **0,833** | 0,764 | 1,000 | 1,000 | **0,500** |

**La variance expliquée croît toujours avec `k`. La performance, non.**

On lit souvent qu'il faut choisir `k` pour capturer « 90 % de la variance ». Ce
critère ne dit rien de la qualité de la recherche, c'est un critère de
compression, pas de pertinence. Vérifié par
`test_la_variance_expliquee_est_un_mauvais_critere`.

On voit aussi que `k = 2` est catastrophique, la réduction écrase des
distinctions bien réelles.

---

## Résultat n°2, LSA règle une partie du problème

| moteur | hit@5 | MRR | lexicale | partielle | **sémantique** |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 0,778 | 0,750 | 1,000 | 1,000 | 0,333 |
| BM25 | 0,778 | 0,778 | 1,000 | 1,000 | 0,333 |
| **LSA (k=20)** | **0,833** | 0,764 | 1,000 | 1,000 | **0,500** |
| hybride BM25+LSA | 0,833 | 0,764 | 1,000 | 1,000 | 0,500 |

Un cas concret.

```
« savoir si tel terme est un nom un verbe ou un adjectif »
attendu : Étiquetage morphosyntaxique

BM25 : Recherche documentai | BM25 | Analyse sémantique l
LSA  : Recherche documentai | BM25 | Analyse sémantique l | >>Étiquetage morphosyn | Transfert cross-ling
```

LSA le trouve au rang 4. BM25 ne le trouve pas du tout, la requête ne contient
aucun de ses termes.

---

## Résultat n°3, la limite que LSA ne franchit pas

C'est le résultat que je trouve le plus intéressant de cette version, et je ne
l'attendais pas.

| difficulté | termes de la requête présents dans le vocabulaire |
|---|---:|
| lexicale | 18 / 21, **85,7 %** |
| partielle | 22 / 31 à 71,0 % |
| **sémantique** | 7 / 38, **18,4 %** |

Et surtout. **2 requêtes sur 6 n'ont aucun terme dans le vocabulaire.**

```
« pourquoi un score automatique juge mal une bonne reformulation »
« ajuster progressivement les coefficients pour réduire l'erreur »
```

Or **la projection d'une requête dans LSA reste lexicale**. `projeter_requete`
construit d'abord un vecteur de termes, puis le multiplie par `U`. Si le vecteur
de départ est nul, la projection est nulle et LSA ne renvoie **rien**.

Vérifié par `test_requete_hors_vocabulaire_renvoie_vide`.

### La cause profonde

Sur 20 documents, il n'y a pas assez de co-occurrences pour apprendre que
« coefficients » et « paramètres » sont liés. **LSA ne peut pas inventer une
sémantique absente de son corpus.**

C'est exactement ce qu'apportent les plongements de phrases pré-entraînés,
`all-MiniLM-L6-v2` dans mon TP de M2.

1. une sémantique apprise **ailleurs**, sur des milliards de mots,
2. une tokenisation en **sous-mots**, qui garantit qu'aucun terme n'est jamais
   hors-vocabulaire.

Ce sont deux choses distinctes, et les deux sont nécessaires.

---

## L'hybride n'est pas un compromis mou

`RechercheHybride` fusionne les **rangs** et non les scores, par *Reciprocal
Rank Fusion* (Cormack et al., 2009).

```
score(d) = Σ_moteurs  1 / (60 + rang_moteur(d))
```

Pourquoi les rangs, les scores de BM25 (non bornés) et de LSA (cosinus dans
[−1, 1]) ne sont pas commensurables. Les normaliser demanderait de connaître
leurs distributions. Les rangs, eux, sont directement comparables, c'est
pourquoi la RRF est la méthode standard en production, elle ne demande **aucun
calibrage**.

Le paramètre `k = 60` amortit l'écart entre les premiers rangs, sans lui, un
document classé 1er par un seul moteur écraserait un document classé 2e par les
deux. Vérifié par `test_rrf_favorise_le_consensus`.

---

## Ce que j'aurais dû voir dans mon TP de M2

J'avais conclu « les sentence transformers sont meilleurs ». Mes propres
chiffres disaient quelque chose de plus précis.

| pipeline | Hit Rate | MAP |
|---|---:|---:|
| TF-IDF + k-NN | 0,960 | 0,883 |
| Sentence Transformers | 0,996 | 0,958 |
| **BM25** | **0,982** | **0,500** |

BM25 avait le **deuxième meilleur Hit Rate**, devant TF-IDF, et le **pire MAP**,
de très loin. Ces deux chiffres, lus ensemble, disent. *BM25 trouve le bon
document mais le classe mal.*

Un reclassement l'aurait sauvé. C'est l'architecture retriever puis reranker des
systèmes RAG modernes, que je n'ai pas encore mise en place.

---

## Ce qui reste ouvert

LSA rattrape une partie des échecs, mais sa projection de requête reste
lexicale, et deux requêtes sur six n'ont aucun terme connu.

Pour aller plus loin il faudrait une représentation apprise sur un corpus bien
plus grand que le mien. Je ne sais pas encore si cela suffit.
