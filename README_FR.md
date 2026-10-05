# Recherche sémantique, de l'index inversé au RAG

Reprendre mon premier programme de TAL, un moteur de recherche, et lui donner l'évaluation qui lui manquait.

**Léo Mégret**, Master Linguistique Informatique, Université Paris Cité

> **État du dépôt, version 2.** Je mène ce travail par étapes, chacune dans son
> propre dossier. Je publie au fur et à mesure plutôt qu'une fois tout terminé.

---

## Pourquoi ce dépôt

En L3, mon premier programme sérieux de TAL était un moteur de recherche, avec un
index inversé, quelques variantes de prétraitement et une pondération TF-IDF. En
M2, mon TP final était un RAG complet, avec base vectorielle et plongements de
phrases.

Trois ans séparent ces deux travaux et le problème est le même, trouver
l'information pertinente dans une collection. Ce qui a changé entre les deux,
c'est la représentation.

Je reconstruis la chaîne depuis le début, en mesurant à chaque étape ce que l'on
gagne et ce que l'on perd.

---

## Les versions publiées

| | Dossier | Contenu | Tests |
|---|---|---|---:|
| **1** | `1.rag_python_projet` | Recherche lexicale, index inversé, TF-IDF, BM25 | 24 |
| **2** | `2.rag_python_projet` | Analyse sémantique latente, la première recherche par le sens | 18 |

Soit **42 tests** au total. Chaque dossier contient tout le contenu du
précédent, plus une étape.

---

## Lancer la dernière version

```bash
cd 2.rag_python_projet
python -m src.semantique
python -m tests.test_semantique
```

---

## Ce qui reste ouvert

LSA rattrape une partie des échecs, mais sa projection de requête reste
lexicale, et deux requêtes sur six n'ont aucun terme connu.

Pour aller plus loin il faudrait une représentation apprise sur un corpus bien
plus grand que le mien. Je ne sais pas encore si cela suffit.

---

## Comment je travaille

Quatre règles que je me suis données en commençant, et que je compte tenir sur
tout le dépôt.

**Rien à télécharger.** Le corpus est écrit dans le code. Mes notebooks de master
commençaient tous par un `wget` vers un serveur universitaire ou un montage de
Google Drive. Deux ans plus tard, la moitié ne s'exécutent plus.

**Rien n'est affirmé sans mesure.** Chaque chiffre de ce fichier correspond à une
commande qu'on peut relancer.

**Les erreurs de mes rendus sont citées, pas effacées.** Quand un résultat que
j'avais rendu en cours était faux ou incomplet, je le dis et je donne le résultat
correct.

**Les résultats négatifs restent.** Quand une expérience montre l'inverse de ce
que j'attendais, j'écris ce que j'ai trouvé.

**Le code est commenté en français.**

---

---

*Version anglaise, [README.md](README.md).*
