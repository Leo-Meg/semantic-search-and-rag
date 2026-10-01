# Version 1 — Recherche lexicale : index inversé, TF-IDF, BM25

> **Où j'en suis.** Je pars du tout premier programme sérieux que j'ai écrit en
> TAL — un moteur de recherche, en L3 — et je lui donne enfin ce qui lui
> manquait : une évaluation. Puis je mesure précisément **où** la recherche
> lexicale échoue, et je m'arrête sur ce constat.

---

## Ce que contient cette version

| Fichier | Ce que j'y fais |
|---|---|
| `src/corpus.py` | 20 documents sur le TAL, 18 requêtes annotées et **classées par difficulté**, normalisation Unicode, mots vides justifiés, racinisation française écrite à la main. |
| `src/recherche.py` | `IndexInverse` (avec deux formules d'IDF), `RechercheTFIDF` (TF logarithmique, cosinus), `RechercheBM25` (saturation + normalisation de longueur), et les métriques Hit Rate / MRR / MAP. |
| `tests/test_recherche.py` | 24 tests, dont la vérification que BM25 sature bien la fréquence. |

## Origine universitaire

- **Projet de L3** (« moteur de recherche ») : index inversé sur un corpus
  d'articles, quatre variantes de prétraitement, pondération TF-IDF.
- **TP de M2** — *RAG : semantic search* (Florent Storme) : TF-IDF + k-NN, BM25
  et plongements de phrases sur Quora Question Pairs, évalués en Hit Rate et MAP.

Trois ans séparent ces deux travaux. Ce qui a changé entre les deux n'est pas la
métrique d'évaluation — c'est la **représentation**.

## Lancer le code

```bash
python -m src.corpus
```

```bash
python -m src.recherche
```

```bash
python -m tests.test_recherche
```

**Aucune dépendance** — pas même NumPy.

---

## Le protocole : des requêtes classées par difficulté

C'est la seule décision de conception qui compte vraiment ici. Mes 18 requêtes
sont réparties en trois niveaux :

| niveau | définition | exemple |
|---|---|---|
| **lexicale** | partage plusieurs termes avec le document | « comment fonctionne la tokenisation en sous-mots » → *Tokenisation en sous-mots* |
| **partielle** | vocabulaire en partie différent | « comment découper un mot en morceaux plus petits » → *Tokenisation en sous-mots* |
| **sémantique** | **quasiment aucun terme en commun** | « pourquoi un score automatique juge mal une bonne reformulation » → *Évaluation de la traduction* |

J'ai écrit les requêtes du troisième niveau en vérifiant, **terme racinisé par
terme racinisé**, qu'elles ne partagent rien avec le document attendu. C'est le
seul moyen d'isoler ce que la recherche lexicale ne peut structurellement pas
faire.

---

## Résultat n°1 — La moyenne globale ne dit rien

| moteur | hit@1 | hit@3 | hit@5 | MRR | MAP |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 0,722 | 0,778 | 0,778 | 0,750 | 0,750 |
| BM25 | 0,778 | 0,778 | 0,778 | 0,778 | 0,778 |

À première vue : « environ 78 %, les deux se valent ». Le détail raconte une
tout autre histoire.

## Résultat n°2 — Le détail, où tout se joue

| moteur | lexicale | partielle | **sémantique** |
|---|---:|---:|---:|
| TF-IDF | 1,000 | 1,000 | **0,333** |
| BM25 | 1,000 | 1,000 | **0,333** |

**Parfait quand la requête partage des mots. Deux tiers d'échecs quand elle n'en
partage aucun.**

Et ce n'est pas un défaut d'implémentation : c'est **structurel**. TF-IDF et
BM25 ne savent comparer que des chaînes de caractères. Si l'utilisateur écrit
« un idiome voisin compense-t-il l'absence de corpus annoté » et que le document
dit « transfert cross-lingue », il n'y a rien à apparier.

Or c'est précisément le cas d'usage réel : **un utilisateur ne connaît pas le
vocabulaire du document qu'il cherche.** S'il le connaissait, il n'aurait pas
besoin de chercher.

C'est ce constat qui motive tout le reste du projet.

---

## Résultat n°3 — Ce que mon projet de L3 aurait dû mesurer

| configuration | hit@5 | MRR | \|V\| |
|---|---:|---:|---:|
| brut (ni mots vides ni racines) | 0,833 | 0,833 | 337 |
| sans mots vides | **0,778** | 0,778 | 297 |
| racinisation seule | 0,944 | 0,917 | 317 |
| **sans mots vides + racinisation** | **0,944** | **0,944** | 277 |

En L3, mon rendu comparait quatre fonctions `index1` à `index4` **sans une seule
mesure**. Voilà les chiffres que j'aurais dû produire.

Deux choses que je n'avais pas comprises :

1. **La racinisation est ce qui aide vraiment.** Elle fait se rencontrer
   « tokenisation » (document) et « tokeniser » (requête). Sans elle, ce sont
   deux termes sans rapport.
2. **Le retrait des mots vides est en grande partie redondant avec l'IDF.** Un
   mot présent partout a un IDF de zéro : TF-IDF l'annule tout seul. Le retirer
   à la main réduit le vocabulaire, mais peut même **dégrader** le score (ligne 2)
   quand la liste retire un mot qui portait de l'information.

---

## Ce que j'ai appris en écrivant BM25

Deux idées que TF-IDF n'a pas, et que je vérifie chacune par un test :

**La saturation** (`k1`). Le terme `tf / (tf + k1·…)` tend vers 1. Passer de 1 à
2 occurrences change beaucoup ; de 20 à 21, presque rien. Mesuré : un document
contenant 20 fois le terme ne score que ~2× plus qu'un document le contenant une
fois — pas 20×.

**La normalisation par la longueur** (`b`). Un long document a mécaniquement
plus d'occurrences de tout. `b = 0` : aucune correction. `b = 1` : correction
complète. La valeur usuelle 0,75 est un compromis empirique — la longueur est en
partie un artefact, et en partie une vraie information (un long document couvre
réellement plus de choses).

**Un détail que je trouve élégant** : l'IDF de BM25 s'écrit
`log((N − df + 0,5)/(df + 0,5) + 1)`. Sans le `+ 1`, un terme présent dans plus
de la moitié des documents recevrait un poids **négatif** — le contenir ferait
*baisser* le score. Vérifié par `test_idf_bm25_toujours_positif`.

---

## Un choix que j'assume : une racinisation écrite à la main

Je n'importe pas Snowball. Ma racinisation coupe une quinzaine de suffixes, et
je montre ses erreurs :

```
tokenisation  -> tokenis     tokeniser -> tokenis    (convergence : le but)
gradient      -> gradi       gradients -> gradi      (convergence)
apprentissage -> apprentissag  apprendre -> apprendr (ÉCHEC : alternance de radical)
souris        -> souri                              (SUR-racinisation)
```

Je garde ces défauts **visibles** plutôt que d'importer une boîte noire : ils
expliquent une partie des échecs que je mesure, et c'est exactement ce que je
veux pouvoir diagnostiquer. Le cas `apprentissage / apprendre` est
particulièrement instructif — **aucune** règle de troncature ne peut le
résoudre, il faudrait une lemmatisation avec dictionnaire.

---

## Ce qui reste ouvert

Deux tiers des requêtes sémantiques échouent, et la raison est toujours la même :
aucun terme partagé entre la requête et le document.

Aucun réglage de BM25 ne peut répondre à cela. Ce n'est pas la pondération qu'il
faut changer, c'est la **représentation**. Je ne sais pas encore par quoi la
remplacer.
