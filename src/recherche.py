"""
Recherche documentaire : index inversé, TF-IDF, BM25 — et leur évaluation.

D'où ça vient
-------------
* **Projet de L3** (« moteur de recherche ») : construire un index inversé sur
  un corpus d'articles, avec plusieurs variantes de prétraitement, puis une
  pondération TF-IDF. C'est le tout premier programme sérieux que j'ai écrit en
  TAL.
* **TP de M2** — *RAG : semantic search* (Florent Storme) : comparer TF-IDF +
  k-NN, BM25 et un modèle de plongements de phrases sur Quora Question Pairs,
  avec Hit Rate et MAP.

Les deux travaux portent sur la même question à trois ans d'écart, et c'est
pour ça que je les réunis : entre les deux, ce qui a changé n'est pas la
métrique d'évaluation, c'est la représentation.

Ce que je réimplémente ici
---------------------------
Tout, sauf les plongements neuronaux (qui arrivent à la version 2, en NumPy
également). L'index inversé, l'IDF, la normalisation cosinus, BM25 avec ses deux
paramètres, et les métriques Hit Rate / MAP / MRR.
"""

from __future__ import annotations

import math
from collections import defaultdict

from .corpus import DOCUMENTS, REQUETES, segmenter, texte_complet


# --------------------------------------------------------------------------- #
# 1. L'index inversé
# --------------------------------------------------------------------------- #


class IndexInverse:
    """La structure de données qui fonde toute la recherche documentaire.

    Un index inversé associe à chaque **terme** la liste des documents où il
    apparaît, avec sa fréquence. « Inversé » parce que l'index naturel serait
    l'inverse : document -> termes.

    Pourquoi c'est une bonne idée : pour répondre à une requête de 3 termes, on
    ne regarde que les documents contenant l'un de ces 3 termes, au lieu de
    parcourir toute la collection. Sur le web, c'est la différence entre
    possible et impossible.

    Mon projet de L3 s'arrêtait exactement là — je construisais l'index et je
    m'étonnais que les résultats soient mauvais. La raison, que je comprends
    maintenant : un index dit *où* sont les termes, pas *combien ils comptent*.
    C'est le rôle de la pondération.
    """

    def __init__(self, documents: list[dict], **options_segmentation):
        self.documents = documents
        self.options = options_segmentation
        self.index: dict[str, dict[str, int]] = defaultdict(dict)
        self.longueurs: dict[str, int] = {}
        self.frequences_document: dict[str, int] = defaultdict(int)

        for document in documents:
            termes = segmenter(texte_complet(document), **options_segmentation)
            self.longueurs[document["id"]] = len(termes)
            comptes: dict[str, int] = defaultdict(int)
            for terme in termes:
                comptes[terme] += 1
            for terme, compte in comptes.items():
                self.index[terme][document["id"]] = compte
                self.frequences_document[terme] += 1

        self.nb_documents = len(documents)
        self.longueur_moyenne = (sum(self.longueurs.values())
                                 / max(self.nb_documents, 1))

    def documents_contenant(self, terme: str) -> dict[str, int]:
        return self.index.get(terme, {})

    def candidats(self, termes: list[str]) -> set[str]:
        """Documents contenant **au moins un** terme de la requête.

        C'est le filtrage préliminaire qui rend la recherche rapide. Les vrais
        moteurs raffinent (WAND, block-max) mais le principe est celui-là.
        """
        resultat: set[str] = set()
        for terme in termes:
            resultat |= set(self.index.get(terme, {}))
        return resultat

    def idf(self, terme: str, lissage: str = "standard") -> float:
        """Inverse Document Frequency — combien ce terme est-il discriminant ?

        Intuition : un terme présent dans tous les documents ne permet de
        distinguer aucun document. Un terme présent dans un seul est
        extrêmement informatif. L'IDF formalise ça par un logarithme.

        Deux variantes, et la différence compte :

        * ``standard`` : ``log(N / df)`` — vaut **0** pour un terme présent
          partout, ce qui l'élimine proprement ;
        * ``bm25`` : ``log((N − df + 0,5) / (df + 0,5) + 1)`` — issue du modèle
          probabiliste de pertinence, elle décroît plus vite et reste toujours
          positive grâce au ``+ 1``. Sans ce ``+ 1``, un terme présent dans plus
          de la moitié des documents recevrait un poids **négatif** : le
          contenir ferait *baisser* le score, ce qui est absurde.
        """
        df = self.frequences_document.get(terme, 0)
        if df == 0:
            return 0.0
        if lissage == "bm25":
            return math.log((self.nb_documents - df + 0.5) / (df + 0.5) + 1.0)
        return math.log(self.nb_documents / df)

    def termes_les_plus_discriminants(self, n: int = 10) -> list[tuple[str, float]]:
        scores = [(t, self.idf(t)) for t in self.index]
        return sorted(scores, key=lambda x: -x[1])[:n]

    def termes_les_moins_discriminants(self, n: int = 10) -> list[tuple[str, float]]:
        scores = [(t, self.idf(t)) for t in self.index]
        return sorted(scores, key=lambda x: x[1])[:n]


# --------------------------------------------------------------------------- #
# 2. TF-IDF
# --------------------------------------------------------------------------- #


class RechercheTFIDF:
    """Recherche vectorielle TF-IDF avec similarité cosinus.

    Chaque document devient un vecteur creux dont les coordonnées sont les
    poids TF-IDF de ses termes. La requête aussi. On classe par cosinus.

    Le choix de la pondération TF
    ------------------------------
    J'utilise le TF **logarithmique** ``1 + log(tf)`` plutôt que le compte brut.
    Raison : un document qui contient 20 fois « attention » n'est pas 20 fois
    plus pertinent qu'un document qui la contient une fois. La croissance
    logarithmique traduit exactement ce rendement décroissant.

    C'est le schéma ``ltc`` de la notation SMART, et c'est ce que fait
    `sklearn.TfidfVectorizer(sublinear_tf=True)` — une option que je cochais en
    M2 sans savoir ce qu'elle changeait.
    """

    def __init__(self, index: IndexInverse, tf_logarithmique: bool = True):
        self.index = index
        self.tf_logarithmique = tf_logarithmique
        self.vecteurs: dict[str, dict[str, float]] = {}
        self.normes: dict[str, float] = {}

        for document in index.documents:
            identifiant = document["id"]
            vecteur = {}
            for terme, poids_index in index.index.items():
                tf = poids_index.get(identifiant, 0)
                if tf == 0:
                    continue
                vecteur[terme] = self._poids(tf) * index.idf(terme)
            self.vecteurs[identifiant] = vecteur
            self.normes[identifiant] = math.sqrt(
                sum(v * v for v in vecteur.values())) or 1.0

    def _poids(self, tf: int) -> float:
        return 1.0 + math.log(tf) if self.tf_logarithmique else float(tf)

    def rechercher(self, question: str, k: int = 5) -> list[tuple[str, float]]:
        termes = segmenter(question, **self.index.options)
        comptes: dict[str, int] = defaultdict(int)
        for terme in termes:
            comptes[terme] += 1

        vecteur_requete = {
            terme: self._poids(compte) * self.index.idf(terme)
            for terme, compte in comptes.items()
        }
        norme_requete = math.sqrt(
            sum(v * v for v in vecteur_requete.values())) or 1.0

        scores = []
        for identifiant in self.index.candidats(termes):
            vecteur = self.vecteurs[identifiant]
            produit = sum(poids * vecteur.get(terme, 0.0)
                          for terme, poids in vecteur_requete.items())
            if produit > 0:
                scores.append(
                    (identifiant, produit / (norme_requete * self.normes[identifiant]))
                )

        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores[:k]


# --------------------------------------------------------------------------- #
# 3. BM25
# --------------------------------------------------------------------------- #


class RechercheBM25:
    """BM25 Okapi (Robertson & Walker, 1994).

        score(D, Q) = Σ  IDF(q) ·  tf · (k1 + 1)
                         ─────────────────────────────────────
                         tf + k1 · (1 − b + b · |D| / |D|moyen)

    Deux idées que TF-IDF n'a pas :

    * **saturation** (paramètre ``k1``). Le terme ``tf/(tf + k1·…)`` tend vers 1
      quand ``tf`` grandit : passer de 1 à 2 occurrences change beaucoup, passer
      de 20 à 21 ne change presque rien. Le TF logarithmique de TF-IDF fait la
      même chose de façon plus grossière ; BM25 le fait avec une **asymptote**,
      donc un plafond explicite.

    * **normalisation par la longueur** (paramètre ``b``). Un long document a
      mécaniquement plus d'occurrences de tout. ``b = 1`` normalise
      complètement, ``b = 0`` pas du tout. La valeur usuelle ``b = 0,75`` est un
      compromis empirique — la longueur est en partie un artefact et en partie
      une vraie information (un long document *couvre* réellement plus de choses).

    BM25 a plus de 30 ans et reste la référence à battre pour toute nouvelle
    méthode de recherche. Dans mon TP de M2, il obtenait le meilleur Hit Rate
    des trois pipelines (0,982) — devant TF-IDF — mais le pire MAP (0,50).
    Je reviens sur cette apparente contradiction dans le module d'évaluation.
    """

    def __init__(self, index: IndexInverse, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b

    def rechercher(self, question: str, k: int = 5) -> list[tuple[str, float]]:
        termes = segmenter(question, **self.index.options)
        scores: dict[str, float] = defaultdict(float)

        for terme in set(termes):
            idf = self.index.idf(terme, lissage="bm25")
            if idf <= 0:
                continue
            for identifiant, tf in self.index.documents_contenant(terme).items():
                longueur = self.index.longueurs[identifiant]
                denominateur = tf + self.k1 * (
                    1 - self.b + self.b * longueur / self.index.longueur_moyenne
                )
                scores[identifiant] += idf * tf * (self.k1 + 1) / denominateur

        classement = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return classement[:k]


# --------------------------------------------------------------------------- #
# 4. Évaluation
# --------------------------------------------------------------------------- #


def hit_rate(resultats: list[list[str]], pertinents: list[str], k: int) -> float:
    """Proportion de requêtes dont le document pertinent est dans le top-k.

    C'est la métrique la plus lisible : « est-ce que la bonne réponse est dans
    ce que je montre à l'utilisateur ? ». Elle ignore complètement le **rang**.
    """
    return sum(pertinent in classement[:k]
               for classement, pertinent in zip(resultats, pertinents)) / max(len(pertinents), 1)


def mean_reciprocal_rank(resultats: list[list[str]], pertinents: list[str]) -> float:
    """Moyenne de ``1 / rang du premier document pertinent``.

    Rang 1 -> 1,0 ; rang 2 -> 0,5 ; rang 10 -> 0,1 ; absent -> 0.

    Contrairement au Hit Rate, le MRR est **sensible au rang**. C'est la
    métrique à regarder quand l'utilisateur ne consulte que le premier
    résultat — le cas d'un système RAG qui n'injecte qu'un seul passage.
    """
    total = 0.0
    for classement, pertinent in zip(resultats, pertinents):
        if pertinent in classement:
            total += 1.0 / (classement.index(pertinent) + 1)
    return total / max(len(pertinents), 1)


def mean_average_precision(resultats: list[list[str]],
                           pertinents: list[str]) -> float:
    """MAP. Avec un seul document pertinent par requête, elle coïncide avec le MRR.

    Je la garde parce que c'était la métrique demandée dans le TP de M2, et
    parce que sa généralisation à plusieurs documents pertinents est immédiate :
    on moyenne la précision à chaque position où l'on trouve un pertinent.
    """
    total = 0.0
    for classement, pertinent in zip(resultats, pertinents):
        trouves = 0
        somme_precisions = 0.0
        for position, identifiant in enumerate(classement, start=1):
            if identifiant == pertinent:
                trouves += 1
                somme_precisions += trouves / position
        if trouves:
            total += somme_precisions / trouves
    return total / max(len(pertinents), 1)


def evaluer(moteur, requetes: list[dict] | None = None,
            k: int = 5) -> dict[str, float]:
    """Évalue un moteur sur le jeu de requêtes, globalement et par difficulté."""
    requetes = requetes or REQUETES
    resultats = [[i for i, _ in moteur.rechercher(r["question"], k=k)]
                 for r in requetes]
    pertinents = [r["pertinent"] for r in requetes]

    mesures = {
        "hit@1": hit_rate(resultats, pertinents, 1),
        "hit@3": hit_rate(resultats, pertinents, 3),
        f"hit@{k}": hit_rate(resultats, pertinents, k),
        "mrr": mean_reciprocal_rank(resultats, pertinents),
        "map": mean_average_precision(resultats, pertinents),
    }

    for difficulte in ("lexicale", "partielle", "semantique"):
        indices = [i for i, r in enumerate(requetes)
                   if r["difficulte"] == difficulte]
        if indices:
            mesures[f"hit@{k}_{difficulte}"] = hit_rate(
                [resultats[i] for i in indices],
                [pertinents[i] for i in indices], k,
            )
    return mesures


if __name__ == "__main__":
    print("=== Recherche documentaire : index inversé, TF-IDF, BM25 ===\n")

    index = IndexInverse(DOCUMENTS)
    print(f"  {index.nb_documents} documents | {len(index.index)} termes distincts")
    print(f"  longueur moyenne : {index.longueur_moyenne:.1f} termes\n")

    print("  Termes les PLUS discriminants (IDF élevé) :")
    print("    " + ", ".join(f"{t} ({v:.2f})"
                             for t, v in index.termes_les_plus_discriminants(6)))
    print("  Termes les MOINS discriminants (IDF faible) :")
    print("    " + ", ".join(f"{t} ({v:.2f})"
                             for t, v in index.termes_les_moins_discriminants(6)))
    print("\n  Un terme présent dans tous les documents a un IDF de 0 : il")
    print("  n'aide à distinguer aucun document, et TF-IDF l'annule proprement.\n")

    tfidf = RechercheTFIDF(index)
    bm25 = RechercheBM25(index)

    # -- Exemple détaillé --------------------------------------------------- #
    print("--- Exemple : une même requête, deux moteurs ---\n")
    question = "comment découper un mot en morceaux plus petits"
    titres = {d["id"]: d["titre"] for d in DOCUMENTS}
    print(f"  « {question} »\n")
    for nom, moteur in (("TF-IDF", tfidf), ("BM25", bm25)):
        print(f"    {nom} :")
        for rang, (identifiant, score) in enumerate(moteur.rechercher(question, k=3), 1):
            print(f"      {rang}. {identifiant} {titres[identifiant]:<34} "
                  f"score {score:.4f}")
        print()

    # -- Évaluation globale -------------------------------------------------- #
    print("--- Évaluation sur les 18 requêtes ---\n")
    entete = (f"  {'moteur':>16} | {'hit@1':>6} | {'hit@3':>6} | {'hit@5':>6} | "
              f"{'MRR':>6} | {'MAP':>6}")
    print(entete)
    print("  " + "-" * (len(entete) - 2))
    for nom, moteur in (("TF-IDF", tfidf), ("BM25", bm25)):
        m = evaluer(moteur)
        print(f"  {nom:>16} | {m['hit@1']:>6.3f} | {m['hit@3']:>6.3f} | "
              f"{m['hit@5']:>6.3f} | {m['mrr']:>6.3f} | {m['map']:>6.3f}")

    # -- Là où tout se joue -------------------------------------------------- #
    print("\n--- Le détail par difficulté : là où tout se joue ---\n")
    print(f"  {'moteur':>16} | {'lexicale':>9} | {'partielle':>10} | {'sémantique':>11}")
    print("  " + "-" * 54)
    for nom, moteur in (("TF-IDF", tfidf), ("BM25", bm25)):
        m = evaluer(moteur)
        print(f"  {nom:>16} | {m['hit@5_lexicale']:>9.3f} | "
              f"{m['hit@5_partielle']:>10.3f} | {m['hit@5_semantique']:>11.3f}")

    print(
        "\n  C'est CE tableau qui compte, pas la moyenne globale. Les moteurs\n"
        "  lexicaux réussissent quand la requête partage des mots avec le\n"
        "  document, et échouent quand elle n'en partage aucun — ce qui est\n"
        "  précisément le cas d'usage d'un utilisateur réel, qui ne connaît pas\n"
        "  le vocabulaire du document qu'il cherche.\n"
    )

    # -- L'effet des paramètres de BM25 -------------------------------------- #
    print("--- Les deux paramètres de BM25 ---\n")
    print(f"  {'k1':>5} | {'b':>5} | {'hit@5':>6} | {'MRR':>6} | commentaire")
    print("  " + "-" * 62)
    for k1, b, commentaire in [
        (1.5, 0.75, "valeurs usuelles"),
        (0.1, 0.75, "saturation quasi immédiate"),
        (10.0, 0.75, "quasiment pas de saturation"),
        (1.5, 0.0, "aucune normalisation de longueur"),
        (1.5, 1.0, "normalisation complète"),
    ]:
        m = evaluer(RechercheBM25(index, k1=k1, b=b))
        print(f"  {k1:>5.1f} | {b:>5.2f} | {m['hit@5']:>6.3f} | {m['mrr']:>6.3f} | "
              f"{commentaire}")

    # -- L'effet du prétraitement -------------------------------------------- #
    print("\n--- L'effet du prétraitement (mon sujet de projet en L3) ---\n")
    print(f"  {'configuration':>34} | {'hit@5':>6} | {'MRR':>6} | {'|V|':>5}")
    print("  " + "-" * 60)
    for description, options in [
        ("brut (ni mots vides ni racines)", dict(retirer_mots_vides=False,
                                                 racinisation=False)),
        ("sans mots vides", dict(retirer_mots_vides=True, racinisation=False)),
        ("racinisation seule", dict(retirer_mots_vides=False, racinisation=True)),
        ("sans mots vides + racinisation", dict(retirer_mots_vides=True,
                                                racinisation=True)),
    ]:
        index_variante = IndexInverse(DOCUMENTS, **options)
        m = evaluer(RechercheBM25(index_variante))
        print(f"  {description:>34} | {m['hit@5']:>6.3f} | {m['mrr']:>6.3f} | "
              f"{len(index_variante.index):>5}")

    print(
        "\n  En L3 je faisais varier ces options sans savoir les évaluer : mon\n"
        "  rendu comparait quatre fonctions `index1` à `index4` sans une seule\n"
        "  mesure. Voilà les chiffres que j'aurais dû produire.\n"
        "\n"
        "  La racinisation aide parce qu'elle fait se rencontrer les variantes\n"
        "  d'un même lemme. Le retrait des mots vides change surtout la taille\n"
        "  du vocabulaire — l'IDF les aurait de toute façon presque annulés.\n"
        "  C'est un point que je n'avais pas compris : avec TF-IDF, la liste de\n"
        "  mots vides est en grande partie REDONDANTE avec l'IDF."
    )
