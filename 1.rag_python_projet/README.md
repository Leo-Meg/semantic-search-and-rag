# Version 1, recherche lexicale. Index inversé, TF-IDF, BM25

> **Où j'en suis.** Je pars du premier programme sérieux que j'ai écrit en TAL,
> un moteur de recherche, en L3, et je lui donne ce qui lui manquait, une
> évaluation. Je mesure ensuite où la recherche lexicale échoue, et je m'arrête
> sur ce constat.

---

## Ce que contient cette version

| Fichier | Ce que j'y fais |
|---|---|
| `src/corpus.py` | 20 documents sur le TAL, 18 requêtes annotées et classées par difficulté, normalisation Unicode, mots vides justifiés, racinisation française écrite à la main. |
| `src/recherche.py` | `IndexInverse` avec deux formules d'IDF, `RechercheTFIDF` en TF logarithmique et cosinus, `RechercheBM25` avec saturation et normalisation de longueur, et les métriques Hit Rate, MRR et MAP. |
| `tests/test_recherche.py` | 24 tests, dont la vérification que BM25 sature bien la fréquence. |

## Origine universitaire

- Projet de L3, un moteur de recherche. Index inversé sur un corpus d'articles,
  quatre variantes de prétraitement, pondération TF-IDF.
- TP de M2, *RAG, semantic search* (Florent Storme). TF-IDF et k-NN, BM25 et
  plongements de phrases sur Quora Question Pairs, évalués en Hit Rate et MAP.

Trois ans séparent ces deux travaux. Ce qui a changé entre les deux n'est pas la
métrique d'évaluation, c'est la représentation.

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

Aucune dépendance, pas même NumPy.

---

## Le protocole, des requêtes classées par difficulté

C'est la décision de conception qui compte le plus ici. Mes 18 requêtes sont
réparties en trois niveaux.

| niveau | définition | exemple |
|---|---|---|
| lexicale | partage plusieurs termes avec le document | « comment fonctionne la tokenisation en sous-mots », vers *Tokenisation en sous-mots* |
| partielle | vocabulaire en partie différent | « comment découper un mot en morceaux plus petits », vers *Tokenisation en sous-mots* |
| sémantique | quasiment aucun terme en commun | « pourquoi un score automatique juge mal une bonne reformulation », vers *Évaluation de la traduction* |

J'ai écrit les requêtes du troisième niveau en vérifiant, terme racinisé par terme
racinisé, qu'elles ne partagent rien avec le document attendu. C'est le seul moyen
d'isoler ce que la recherche lexicale ne peut pas faire.

---

## Résultat n°1, la moyenne globale ne dit rien

| moteur | hit@1 | hit@3 | hit@5 | MRR | MAP |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 0,722 | 0,778 | 0,778 | 0,750 | 0,750 |
| BM25 | 0,778 | 0,778 | 0,778 | 0,778 | 0,778 |

À première vue, environ 78 % pour les deux moteurs, qui semblent donc se valoir.
Le détail donne une lecture différente.

## Résultat n°2, le détail

| moteur | lexicale | partielle | **sémantique** |
|---|---:|---:|---:|
| TF-IDF | 1,000 | 1,000 | 0,333 |
| BM25 | 1,000 | 1,000 | 0,333 |

Les deux moteurs sont parfaits quand la requête partage des mots avec le
document, et échouent deux fois sur trois quand elle n'en partage aucun.

Ce n'est pas un défaut d'implémentation, c'est la limite de la méthode. TF-IDF et
BM25 ne savent comparer que des chaînes de caractères. Si l'utilisateur écrit
« un idiome voisin compense-t-il l'absence de corpus annoté » et que le document
dit « transfert cross-lingue », il n'y a rien à apparier.

C'est pourtant le cas d'usage réel. Un utilisateur ne connaît pas le vocabulaire
du document qu'il cherche, sinon il n'aurait pas besoin de chercher.

---

## Résultat n°3, ce que mon projet de L3 aurait dû mesurer

| configuration | hit@5 | MRR | \|V\| |
|---|---:|---:|---:|
| brut, ni mots vides ni racines | 0,833 | 0,833 | 337 |
| sans mots vides | 0,778 | 0,778 | 297 |
| racinisation seule | 0,944 | 0,917 | 317 |
| sans mots vides et racinisation | 0,944 | 0,944 | 277 |

En L3, mon rendu comparait quatre fonctions `index1` à `index4` sans une seule
mesure. Voilà les chiffres que j'aurais dû produire.

Deux choses que je n'avais pas comprises.

1. **La racinisation est ce qui aide le plus.** Elle fait se rencontrer
   « tokenisation » dans le document et « tokeniser » dans la requête. Sans elle,
   ce sont deux termes sans rapport.
2. **Le retrait des mots vides recoupe en grande partie l'IDF.** Un mot présent
   partout a un IDF de zéro, et TF-IDF l'annule donc tout seul. Le retirer à la
   main réduit le vocabulaire, et peut dégrader le score, comme à la ligne 2,
   quand la liste retire un mot qui portait de l'information.

---

## Ce que j'ai appris en écrivant BM25

Deux idées que TF-IDF n'a pas, et que je vérifie chacune par un test.

**La saturation**, réglée par `k1`. Le terme `tf / (tf + k1·…)` tend vers 1.
Passer de 1 à 2 occurrences change beaucoup, passer de 20 à 21 ne change presque
rien. Mesuré, un document contenant 20 fois le terme score environ 2 fois plus
qu'un document le contenant une fois, et non 20 fois plus.

**La normalisation par la longueur**, réglée par `b`. Un long document a
mécaniquement plus d'occurrences de tout. À `b = 0` il n'y a aucune correction, à
`b = 1` la correction est complète. La valeur usuelle de 0,75 est un compromis
empirique, parce que la longueur est en partie un artefact et en partie une vraie
information, un long document couvrant réellement plus de choses.

**Un détail que je trouve bien vu.** L'IDF de BM25 s'écrit
`log((N − df + 0,5)/(df + 0,5) + 1)`. Sans le `+ 1`, un terme présent dans plus de
la moitié des documents recevrait un poids négatif, et le contenir ferait baisser
le score. Vérifié par `test_idf_bm25_toujours_positif`.

---

## Un choix que j'assume, une racinisation écrite à la main

Je n'importe pas Snowball. Ma racinisation coupe une quinzaine de suffixes, et je
montre ses erreurs.

```
tokenisation  -> tokenis     tokeniser -> tokenis    (convergence, le but)
gradient      -> gradi       gradients -> gradi      (convergence)
apprentissage -> apprentissag  apprendre -> apprendr (ECHEC, alternance de radical)
souris        -> souri                              (SUR-racinisation)
```

Je garde ces défauts visibles plutôt que d'importer une boîte noire. Ils
expliquent une partie des échecs que je mesure, et c'est ce que je veux pouvoir
diagnostiquer. Le cas `apprentissage` contre `apprendre` est instructif, aucune
règle de troncature ne peut le résoudre, il faudrait une lemmatisation avec
dictionnaire.

---

## Ce qui reste ouvert

Deux tiers des requêtes sémantiques échouent, et la raison est chaque fois la
même, aucun terme partagé entre la requête et le document.

Aucun réglage de BM25 ne répond à cela. C'est la représentation qu'il faudrait
changer, et je ne sais pas encore par quoi la remplacer.
