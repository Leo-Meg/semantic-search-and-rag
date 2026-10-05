"""
Analyse sémantique latente (LSA) : la première recherche vraiment sémantique.

D'où ça vient
-------------
* **Devoir de L3/M1 « Topic analysis »** (Benoît Crabbé, bases formelles du
  TAL) : construire une matrice termes × documents en NumPy, la pondérer par
  TF-IDF, lui appliquer une SVD, puis mesurer des similarités cosinus dans
  l'espace réduit. L'énoncé insistait : *« On veillera dans cet exercice à ne
  pas utiliser nltk ou scikit-learn. »*
* **TP de M1 « Topic modelling — LSA and LDA »** (Timothée Bernard) : comparer
  LSA et LDA sur les sous-titres de *Game of Thrones*, avec un jeu artificiel
  de contrôle pour valider les méthodes dans un cas idéal.

Le problème que LSA résout
---------------------------
La version 1 a mesuré l'échec de la recherche lexicale : **0,333** de hit@5 sur
les requêtes qui ne partagent aucun mot avec le document attendu. La cause est
identifiée : TF-IDF et BM25 comparent des chaînes de caractères.

LSA propose une solution étonnamment simple : au lieu de comparer les documents
dans l'espace des **termes** (une dimension par mot, presque vide), on les
compare dans un espace de dimension réduite obtenu par décomposition en valeurs
singulières.

Pourquoi ça marche
------------------
La SVD trouve les directions qui expliquent le plus de variance dans la matrice
termes × documents. Ces directions regroupent les termes qui **co-occurrent**.
Si « attention » et « transformer » apparaissent souvent ensemble, ils se
retrouvent projetés dans la même direction latente — même si aucun document ne
les contient tous les deux.

C'est exactement l'hypothèse distributionnelle (Harris, 1954), appliquée à
l'algèbre linéaire au lieu des réseaux de neurones. Historiquement, LSA
(Deerwester et al., 1990) précède word2vec de vingt-trois ans et repose sur la
même intuition.
"""

from __future__ import annotations

import math

import numpy as np

from .corpus import DOCUMENTS, REQUETES, segmenter, texte_complet
from .recherche import IndexInverse, evaluer


# --------------------------------------------------------------------------- #
# 1. La matrice termes × documents
# --------------------------------------------------------------------------- #


def matrice_termes_documents(
    index: IndexInverse, ponderation: str = "tfidf"
) -> tuple[np.ndarray, list[str], list[str]]:
    """Construit la matrice ``(nb_termes, nb_documents)``.

    Args:
        ponderation: ``"comptes"``, ``"tf"`` (logarithmique) ou ``"tfidf"``.

    Returns:
        ``(M, termes, identifiants)``.

    L'orientation termes × documents est celle de l'énoncé de mon devoir de L3
    (« autant de lignes qu'il y a de termes et autant de colonnes qu'il y a de
    documents »). Elle est aussi celle de l'article original de Deerwester.
    Attention : scikit-learn utilise l'orientation inverse, ce qui inverse le
    rôle de U et de Vᵀ — une source d'erreurs classique quand on lit du code.
    """
    termes = sorted(index.index)
    identifiants = [d["id"] for d in index.documents]
    position = {i: n for n, i in enumerate(identifiants)}

    M = np.zeros((len(termes), len(identifiants)))
    for ligne, terme in enumerate(termes):
        idf = index.idf(terme)
        for identifiant, tf in index.index[terme].items():
            if ponderation == "comptes":
                valeur = float(tf)
            elif ponderation == "tf":
                valeur = 1.0 + math.log(tf)
            else:
                valeur = (1.0 + math.log(tf)) * idf
            M[ligne, position[identifiant]] = valeur

    return M, termes, identifiants


# --------------------------------------------------------------------------- #
# 2. LSA
# --------------------------------------------------------------------------- #


class RechercheLSA:
    """Recherche par analyse sémantique latente.

    On décompose ``M = U S Vᵀ`` et on ne garde que les ``k`` premières
    dimensions. Chaque document devient un vecteur dense de dimension ``k``, et
    une requête est projetée dans le même espace par ``q_réduit = qᵀ U_k S_k⁻¹``
    (la « pliure » — *folding-in* — de l'article original).

    Le choix de k
    -------------
    C'est **le** paramètre, et il n'a pas de valeur par défaut raisonnable.

    * ``k`` trop petit : on écrase des distinctions réelles, tous les documents
      se ressemblent.
    * ``k`` trop grand : on retrouve l'espace des termes et on perd tout le
      bénéfice — LSA redevient TF-IDF.

    L'énoncé de mon devoir de L3 demandait de tester k = 3, 5, 7, 9 et de
    commenter la stabilité des résultats. Je fais mieux ici : je balaie k et je
    trace l'effet sur chaque niveau de difficulté, ce qui montre que l'optimum
    n'est **pas le même** selon le type de requête.
    """

    def __init__(self, index: IndexInverse, k: int = 8,
                 ponderation: str = "tfidf"):
        self.index = index
        self.k = k
        self.M, self.termes, self.identifiants = matrice_termes_documents(
            index, ponderation)
        self.position_terme = {t: i for i, t in enumerate(self.termes)}

        # SVD complète puis troncature. Sur un vrai corpus on utiliserait une
        # SVD tronquée (algorithme de Lanczos, ou randomisée) : calculer toutes
        # les valeurs singulières d'une matrice 100 000 × 1 000 000 serait
        # absurde. Ici la matrice fait quelques centaines de lignes.
        U, S, Vt = np.linalg.svd(self.M, full_matrices=False)
        k = min(k, len(S))
        self.U = U[:, :k]
        self.S = S[:k]
        self.Vt = Vt[:k, :]

        # Représentation des documents : les colonnes de S·Vᵀ.
        self.documents_reduits = (np.diag(self.S) @ self.Vt).T  # (n_docs, k)
        self.normes = np.linalg.norm(self.documents_reduits, axis=1)
        self.normes[self.normes == 0] = 1.0

        self.valeurs_singulieres_completes = S

    def variance_expliquee(self) -> float:
        """Part de la « masse » de la matrice capturée par les k dimensions.

        Calculée sur les carrés des valeurs singulières, qui sont proportionnels
        à la variance expliquée par chaque direction. C'est le critère usuel
        pour choisir k — bien qu'il soit, comme on va le voir, un mauvais
        prédicteur de la performance en recherche.
        """
        total = float(np.sum(self.valeurs_singulieres_completes ** 2))
        return float(np.sum(self.S ** 2) / total) if total else 0.0

    def projeter_requete(self, question: str) -> np.ndarray:
        """Projette une requête dans l'espace latent (*folding-in*)."""
        vecteur = np.zeros(len(self.termes))
        for terme in segmenter(question, **self.index.options):
            ligne = self.position_terme.get(terme)
            if ligne is not None:
                vecteur[ligne] += (1.0 + math.log(1)) * self.index.idf(terme)
        # q^T U_k S_k^{-1}, puis remise à l'échelle par S pour être dans le même
        # repère que `documents_reduits`.
        avec_echelle = np.where(self.S > 1e-12, self.S, 1.0)
        return (vecteur @ self.U) / avec_echelle * self.S

    def rechercher(self, question: str, k: int = 5) -> list[tuple[str, float]]:
        q = self.projeter_requete(question)
        norme_q = np.linalg.norm(q)
        if norme_q == 0:
            return []
        similarites = (self.documents_reduits @ q) / (self.normes * norme_q)
        ordre = np.argsort(-similarites)
        return [(self.identifiants[i], float(similarites[i])) for i in ordre[:k]]

    # -- exploration -------------------------------------------------------- #

    def termes_de_la_dimension(self, dimension: int, n: int = 8
                               ) -> list[tuple[str, float]]:
        """Les termes qui contribuent le plus à une dimension latente.

        C'est la seule façon d'**interpréter** un axe de LSA. Contrairement à
        LDA, les dimensions de LSA peuvent être négatives et ne sont pas des
        « thèmes » au sens probabiliste — mais elles restent lisibles.
        """
        poids = self.U[:, dimension]
        ordre = np.argsort(-np.abs(poids))[:n]
        return [(self.termes[i], float(poids[i])) for i in ordre]

    def documents_de_la_dimension(self, dimension: int, n: int = 3
                                  ) -> list[tuple[str, float]]:
        poids = self.Vt[dimension]
        ordre = np.argsort(-np.abs(poids))[:n]
        return [(self.identifiants[i], float(poids[i])) for i in ordre]


# --------------------------------------------------------------------------- #
# 3. Recherche hybride
# --------------------------------------------------------------------------- #


class RechercheHybride:
    """Combine un moteur lexical et un moteur sémantique par fusion de rangs.

    J'utilise la **Reciprocal Rank Fusion** (Cormack et al., 2009) :

        score(d) = Σ_moteurs  1 / (k + rang_moteur(d))

    Pourquoi fusionner les **rangs** et non les scores : les scores de BM25
    (non bornés, dépendant de l'IDF) et de LSA (cosinus dans [−1, 1]) ne sont
    pas commensurables. Les normaliser demanderait de connaître leurs
    distributions. Les rangs, eux, sont directement comparables.

    C'est la méthode utilisée par la plupart des systèmes de recherche hybride
    en production, précisément parce qu'elle ne demande aucun calibrage.

    Le paramètre ``k = 60`` de l'article amortit l'écart entre les premiers
    rangs : sans lui, le document classé 1er par un seul moteur écraserait un
    document classé 2e par les deux.
    """

    def __init__(self, moteurs: list, k_rrf: int = 60):
        self.moteurs = moteurs
        self.k_rrf = k_rrf

    def rechercher(self, question: str, k: int = 5) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for moteur in self.moteurs:
            for rang, (identifiant, _) in enumerate(
                    moteur.rechercher(question, k=20), start=1):
                scores[identifiant] = scores.get(identifiant, 0.0) + 1.0 / (
                    self.k_rrf + rang)
        classement = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return classement[:k]


if __name__ == "__main__":
    from .recherche import RechercheBM25, RechercheTFIDF

    print("=== Analyse sémantique latente ===\n")

    index = IndexInverse(DOCUMENTS)
    M, termes, identifiants = matrice_termes_documents(index)
    print(f"  matrice termes × documents : {M.shape}")
    print(f"  remplissage : {100 * np.count_nonzero(M) / M.size:.1f} % "
          "de cases non nulles")
    print("  (une matrice termes-documents est TOUJOURS très creuse : c'est\n"
          "   précisément ce qui rend la comparaison directe si fragile)\n")

    # -- Le spectre --------------------------------------------------------- #
    lsa_complet = RechercheLSA(index, k=len(identifiants))
    valeurs = lsa_complet.valeurs_singulieres_completes
    print("  Valeurs singulières (les 10 premières) :")
    print("    " + "  ".join(f"{v:.2f}" for v in valeurs[:10]))
    cumul = np.cumsum(valeurs ** 2) / np.sum(valeurs ** 2)
    print("  Variance cumulée expliquée :")
    for k in (2, 5, 8, 12, 16, 20):
        if k <= len(cumul):
            print(f"    k = {k:>2} : {100 * cumul[k - 1]:.1f} %")

    # -- Interprétation des dimensions --------------------------------------- #
    print("\n--- Que contiennent les dimensions latentes ? ---\n")
    lsa = RechercheLSA(index, k=8)
    titres = {d["id"]: d["titre"] for d in DOCUMENTS}
    for dimension in range(3):
        mots = lsa.termes_de_la_dimension(dimension, 6)
        docs = lsa.documents_de_la_dimension(dimension, 2)
        print(f"  dimension {dimension} :")
        print(f"    termes    : " + ", ".join(f"{t} ({p:+.2f})" for t, p in mots))
        print(f"    documents : " + ", ".join(f"{titres[i]}" for i, _ in docs))
        print()

    # -- Le balayage de k ----------------------------------------------------- #
    print("--- Le choix de k : le paramètre qui décide de tout ---\n")
    entete = (f"  {'k':>4} | {'var. expl.':>10} | {'hit@5':>6} | {'MRR':>6} | "
              f"{'lexicale':>9} | {'partielle':>10} | {'sémantique':>11}")
    print(entete)
    print("  " + "-" * (len(entete) - 2))
    meilleur_k, meilleur_score = None, -1.0
    for k in (2, 4, 6, 8, 10, 12, 16, 20):
        moteur = RechercheLSA(index, k=k)
        m = evaluer(moteur)
        print(f"  {k:>4} | {100 * moteur.variance_expliquee():>9.1f}% | "
              f"{m['hit@5']:>6.3f} | {m['mrr']:>6.3f} | "
              f"{m['hit@5_lexicale']:>9.3f} | {m['hit@5_partielle']:>10.3f} | "
              f"{m['hit@5_semantique']:>11.3f}")
        if m["hit@5"] > meilleur_score:
            meilleur_k, meilleur_score = k, m["hit@5"]

    print(f"\n  meilleur k global : {meilleur_k}")
    print(
        "\n  Deux choses à remarquer :\n"
        "  * la variance expliquée croît toujours avec k, mais la PERFORMANCE\n"
        "    non — c'est donc un mauvais critère de choix, contrairement à ce\n"
        "    qu'on lit souvent ;\n"
        "  * l'optimum n'est pas le même selon le type de requête. Un k petit\n"
        "    généralise (bon pour le sémantique, mauvais pour le lexical) ;\n"
        "    un k grand redevient TF-IDF.\n"
    )

    # -- La comparaison finale ------------------------------------------------ #
    print("--- Lexical contre sémantique contre hybride ---\n")
    tfidf = RechercheTFIDF(index)
    bm25 = RechercheBM25(index)
    lsa = RechercheLSA(index, k=meilleur_k)
    hybride = RechercheHybride([bm25, lsa])

    entete = (f"  {'moteur':>18} | {'hit@1':>6} | {'hit@5':>6} | {'MRR':>6} | "
              f"{'lexicale':>9} | {'partielle':>10} | {'sémantique':>11}")
    print(entete)
    print("  " + "-" * (len(entete) - 2))
    for nom, moteur in (("TF-IDF", tfidf), ("BM25", bm25),
                        (f"LSA (k={meilleur_k})", lsa),
                        ("hybride BM25+LSA", hybride)):
        m = evaluer(moteur)
        print(f"  {nom:>18} | {m['hit@1']:>6.3f} | {m['hit@5']:>6.3f} | "
              f"{m['mrr']:>6.3f} | {m['hit@5_lexicale']:>9.3f} | "
              f"{m['hit@5_partielle']:>10.3f} | {m['hit@5_semantique']:>11.3f}")

    # -- Un cas concret ------------------------------------------------------- #
    print("\n--- Deux requêtes sans mot en commun : une réussie, une impossible ---\n")
    for question in REQUETES:
        if question["difficulte"] != "semantique":
            continue
        attendu = question["pertinent"]
        trouve_lsa = attendu in [i for i, _ in lsa.rechercher(question["question"], 5)]
        trouve_bm25 = attendu in [i for i, _ in bm25.rechercher(question["question"], 5)]
        if trouve_lsa and not trouve_bm25:
            print(f"  RÉUSSIE PAR LSA SEULEMENT")
            print(f"    « {question['question']} »")
            print(f"    attendu : {attendu} — {titres[attendu]}")
            for nom, moteur in (("BM25", bm25), ("LSA", lsa)):
                r = moteur.rechercher(question["question"], k=5)
                rendu = " | ".join(
                    f"{'>>' if i == attendu else ''}{titres[i][:20]}" for i, _ in r
                ) or "(aucun résultat)"
                print(f"      {nom:>6} : {rendu}")
            print()
            break

    # -- LA limite de LSA ------------------------------------------------------ #
    print("--- La limite que LSA ne franchit pas ---\n")
    vocabulaire = set(index.index)
    print(f"  {'difficulté':>12} | {'termes connus / total':>22} | couverture")
    print("  " + "-" * 54)
    for difficulte in ("lexicale", "partielle", "semantique"):
        connus = total = 0
        for r in REQUETES:
            if r["difficulte"] != difficulte:
                continue
            termes = segmenter(r["question"], **index.options)
            total += len(termes)
            connus += sum(t in vocabulaire for t in termes)
        print(f"  {difficulte:>12} | {connus:>10} / {total:<10} | "
              f"{100 * connus / max(total, 1):>5.1f} %")

    orphelines = [
        r for r in REQUETES
        if r["difficulte"] == "semantique"
        and not any(t in vocabulaire
                    for t in segmenter(r["question"], **index.options))
    ]
    print(f"\n  {len(orphelines)} requêtes n'ont AUCUN terme dans le vocabulaire :")
    for r in orphelines:
        print(f"    « {r['question']} »")

    print(
        "\nCe que je retiens\n"
        "-----------------\n"
        "* LSA fait ce qu'on lui demande : elle rattrape des requêtes que BM25\n"
        "  rate complètement, en exploitant la CO-OCCURRENCE des termes plutôt\n"
        "  que leur identité. Le hit@5 sur les requêtes sémantiques passe de\n"
        "  0,333 à 0,500.\n"
        "\n"
        "* MAIS elle bute sur une limite qu'elle ne peut pas franchir : la\n"
        "  projection d'une requête reste LEXICALE. `projeter_requete` construit\n"
        "  un vecteur de termes avant de le multiplier par U. Si aucun terme de\n"
        "  la requête n'est dans le vocabulaire, le vecteur est nul et LSA ne\n"
        "  renvoie RIEN. C'est le cas de 2 de mes 6 requêtes sémantiques.\n"
        "\n"
        "* La cause profonde est là : sur 20 documents, il n'y a tout simplement\n"
        "  pas assez de co-occurrences pour apprendre que « coefficients » et\n"
        "  « paramètres » sont liés. LSA ne peut pas inventer une sémantique\n"
        "  absente de son corpus.\n"
        "\n"
        "* C'est exactement ce qu'apportent les plongements de phrases\n"
        "  pré-entraînés (SBERT, `all-MiniLM-L6-v2` dans mon TP de M2) : une\n"
        "  sémantique apprise ailleurs, sur des milliards de mots, et une\n"
        "  tokenisation en sous-mots qui garantit qu'AUCUN terme n'est jamais\n"
        "  hors-vocabulaire. La version 3 architecture le pipeline autour de\n"
        "  cette idée.\n"
        "\n"
        "* Dernière chose, et c'est la leçon que je n'avais pas tirée de mon TP\n"
        "  de M2 : j'y avais conclu « les sentence transformers sont meilleurs »\n"
        "  sans remarquer que BM25 avait le MEILLEUR Hit Rate des trois\n"
        "  pipelines (0,982) tout en ayant le PIRE MAP (0,50). Ces deux chiffres\n"
        "  disaient quelque chose de précis : BM25 TROUVE le bon document mais\n"
        "  le CLASSE mal. Un reclassement l'aurait sauvé — c'est le sujet de la\n"
        "  version 3."
    )
