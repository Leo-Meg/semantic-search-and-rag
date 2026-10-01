# Recherche sémantique — de l'index inversé au RAG

Reprendre mon tout premier programme de TAL, un moteur de recherche, et lui donner enfin l'évaluation qui lui manquait.

**Léo Mégret** — Master Linguistique Informatique, Université Paris Cité

> **État du dépôt : version 1.** C'est la première étape d'un travail que je
> mène par étapes, chacune dans son propre dossier. Seule la version 1 existe à
> ce jour. Je publie au fur et à mesure plutôt qu'une fois tout terminé, parce
> que l'intérêt de ce travail est justement l'enchaînement des questions.

---

## Pourquoi ce dépôt

En L3, mon premier programme sérieux de TAL était un moteur de recherche : un
index inversé, quelques variantes de prétraitement, une pondération TF-IDF. En
M2, mon TP final était un RAG complet, avec base vectorielle et plongements de
phrases.

Trois ans séparent ces deux travaux, et c'est le même problème : trouver
l'information pertinente dans une collection. Ce qui a changé entre les deux,
ce n'est pas la métrique d'évaluation, c'est la représentation.

Je reconstruis la chaîne depuis le début, en mesurant à chaque étape ce que l'on
gagne et ce que l'on perd.

---

## Ce qui existe aujourd'hui

### Version 1 — Recherche lexicale : index inversé, TF-IDF, BM25

| Fichier | Ce que j'y fais |
|---|---|
| `src/corpus.py` | 20 documents sur le TAL, 18 requêtes annotées et classées par difficulté, normalisation Unicode, mots vides justifiés, racinisation française écrite à la main. |
| `src/recherche.py` | `IndexInverse` avec deux formules d'IDF, `RechercheTFIDF` en TF logarithmique et cosinus, `RechercheBM25` avec saturation et normalisation de longueur, et les métriques Hit Rate, MRR et MAP. |
| `tests/test_recherche.py` | 24 tests, dont la vérification que BM25 sature bien la fréquence. |

Origine universitaire : projet de L3, un moteur de recherche sur un corpus
d'articles, et le TP de M2 *RAG : semantic search* (Florent Storme), évalué en
Hit Rate et MAP sur Quora Question Pairs.

---

## Lancer le code

```bash
cd 1.rag_python_projet
python -m src.corpus
python -m src.recherche
python -m tests.test_recherche
```

Aucune dépendance, pas même NumPy.

---

## Ce que je retiens de cette étape

**La recherche lexicale est parfaite tant que la requête partage des mots avec le
document, et s'effondre dès que ce n'est plus le cas.** Sur mes requêtes
sémantiques, celles qui ne partagent aucun terme avec la bonne réponse, je mesure
0,333.

**Classer les requêtes par difficulté change tout.** Tant que je mesurais une
moyenne globale, je ne voyais rien. C'est en séparant les requêtes lexicales des
requêtes sémantiques que l'échec devient lisible, et localisable.

**La racinisation a ses propres limites, et elles sont instructives.** Le cas
`apprentissage` contre `apprendre` ne peut être résolu par aucune règle de
troncature : il faudrait une lemmatisation avec dictionnaire.

---

## Ce qui reste ouvert

Deux tiers des requêtes sémantiques échouent, et la raison est toujours la même :
aucun terme partagé entre la requête et le document.

Aucun réglage de BM25 ne peut répondre à cela. Ce n'est pas la pondération qu'il
faut changer, c'est la représentation. Je ne sais pas encore par quoi la
remplacer.

---

## Comment je travaille

Quatre règles que je me suis données en commençant, et que je compte tenir sur
tout le dépôt.

**Rien à télécharger.** Le corpus est dans le code. Mes notebooks de master
commençaient tous par un `wget` vers un serveur universitaire ou un montage de
Google Drive ; deux ans plus tard, la moitié ne s'exécutent plus.

**Rien n'est affirmé sans mesure.** Chaque chiffre de ce fichier correspond à une
commande qu'on peut relancer.

**Les erreurs de mes rendus sont citées, pas effacées.** Quand un résultat que
j'avais rendu en cours était faux ou incomplet, je le dis et je donne le résultat
correct.

**Les résultats négatifs restent.** Quand une expérience montre l'inverse de ce
que j'attendais, je change la conclusion, pas l'expérience.

**Le code est commenté en français.** C'est un dépôt à lire autant qu'à exécuter.

---

*Version anglaise : [README.md](README.md).*
