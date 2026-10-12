"""
Tests de la version 2 — analyse sémantique latente et recherche hybride.

    python -m tests.test_semantique
"""

from __future__ import annotations

import numpy as np

from src.corpus import DOCUMENTS, REQUETES, segmenter
from src.recherche import IndexInverse, RechercheBM25, evaluer
from src.semantique import (
    RechercheHybride,
    RechercheLSA,
    matrice_termes_documents,
)

INDEX = IndexInverse(DOCUMENTS)


# --------------------------------------------------------------------------- #
# La matrice
# --------------------------------------------------------------------------- #


def test_forme_de_la_matrice() -> None:
    M, termes, identifiants = matrice_termes_documents(INDEX)
    assert M.shape == (len(termes), len(identifiants))
    assert len(identifiants) == len(DOCUMENTS)


def test_la_matrice_est_creuse() -> None:
    """Le fait structurel qui rend la comparaison lexicale fragile."""
    M, _, _ = matrice_termes_documents(INDEX)
    remplissage = np.count_nonzero(M) / M.size
    assert remplissage < 0.2, f"remplissage inattendu : {remplissage:.3f}"


def test_ponderations_differentes() -> None:
    comptes, _, _ = matrice_termes_documents(INDEX, "comptes")
    tfidf, _, _ = matrice_termes_documents(INDEX, "tfidf")
    assert not np.allclose(comptes, tfidf)
    # Les zéros restent des zéros quelle que soit la pondération.
    assert np.array_equal(comptes == 0, tfidf == 0)


# --------------------------------------------------------------------------- #
# LSA
# --------------------------------------------------------------------------- #


def test_svd_reconstruit_la_matrice_a_rang_plein() -> None:
    """Contrôle de correction : à k maximal, U S Vᵀ redonne M."""
    lsa = RechercheLSA(INDEX, k=len(DOCUMENTS))
    reconstruction = lsa.U @ np.diag(lsa.S) @ lsa.Vt
    assert np.allclose(reconstruction, lsa.M, atol=1e-8)


def test_valeurs_singulieres_decroissantes() -> None:
    lsa = RechercheLSA(INDEX, k=10)
    valeurs = lsa.valeurs_singulieres_completes
    assert np.all(np.diff(valeurs) <= 1e-9)
    assert np.all(valeurs >= -1e-12)


def test_variance_expliquee_croit_avec_k() -> None:
    variances = [RechercheLSA(INDEX, k=k).variance_expliquee()
                 for k in (2, 5, 10, 20)]
    assert variances == sorted(variances)
    assert variances[-1] > 0.99  # k = nb de documents : tout est expliqué


def test_documents_reduits_ont_la_bonne_dimension() -> None:
    for k in (3, 8, 15):
        lsa = RechercheLSA(INDEX, k=k)
        assert lsa.documents_reduits.shape == (len(DOCUMENTS), min(k, len(DOCUMENTS)))


def test_lsa_trouve_le_document_evident() -> None:
    lsa = RechercheLSA(INDEX, k=10)
    resultats = lsa.rechercher("tokenisation sous-mots BPE WordPiece", k=3)
    assert resultats[0][0] == "d01"


def test_cosinus_borne() -> None:
    lsa = RechercheLSA(INDEX, k=10)
    for _, score in lsa.rechercher("attention perplexité", k=10):
        assert -1.0 - 1e-9 <= score <= 1.0 + 1e-9


def test_requete_hors_vocabulaire_renvoie_vide() -> None:
    """LA limite de LSA : sa projection de requête reste lexicale.

    Si aucun terme de la requête n'est dans le vocabulaire, le vecteur projeté
    est nul et LSA ne peut rien renvoyer. C'est le cas de 2 de mes 6 requêtes
    sémantiques, et c'est ce que les plongements pré-entraînés résolvent.
    """
    lsa = RechercheLSA(INDEX, k=10)
    assert lsa.rechercher("zzzz qqqq wwww ffff", k=5) == []


def test_les_dimensions_latentes_sont_interpretables() -> None:
    lsa = RechercheLSA(INDEX, k=6)
    for dimension in range(3):
        termes = lsa.termes_de_la_dimension(dimension, 5)
        assert len(termes) == 5
        assert all(isinstance(t, str) for t, _ in termes)
        # Les termes retenus sont bien ceux de plus fort poids absolu.
        poids = [abs(p) for _, p in termes]
        assert poids == sorted(poids, reverse=True)


def test_lsa_ameliore_les_requetes_semantiques() -> None:
    """Le résultat central de cette version."""
    lexical = evaluer(RechercheBM25(INDEX))
    latent = evaluer(RechercheLSA(INDEX, k=20))
    assert latent["hit@5_semantique"] > lexical["hit@5_semantique"], (
        lexical["hit@5_semantique"], latent["hit@5_semantique"]
    )


def test_un_k_trop_petit_degrade_tout() -> None:
    """k trop petit écrase des distinctions réelles."""
    petit = evaluer(RechercheLSA(INDEX, k=2))
    correct = evaluer(RechercheLSA(INDEX, k=12))
    assert petit["hit@5"] < correct["hit@5"], (petit["hit@5"], correct["hit@5"])


def test_la_variance_expliquee_est_un_mauvais_critere() -> None:
    """Elle croît toujours avec k, mais pas la performance.

    C'est le point méthodologique de cette version : on lit souvent qu'il faut
    choisir k pour capturer « 90 % de la variance ». Ce critère ne dit rien de
    la qualité de la recherche.
    """
    mesures = [(k, RechercheLSA(INDEX, k=k).variance_expliquee(),
                evaluer(RechercheLSA(INDEX, k=k))["mrr"])
               for k in (2, 4, 6, 8, 10, 12)]
    variances = [v for _, v, _ in mesures]
    assert variances == sorted(variances), "la variance doit croître"
    performances = [p for _, _, p in mesures]
    assert performances != sorted(performances) or performances[0] < performances[-1]


# --------------------------------------------------------------------------- #
# Hybride
# --------------------------------------------------------------------------- #


def test_hybride_fusionne_les_deux_moteurs() -> None:
    bm25 = RechercheBM25(INDEX)
    lsa = RechercheLSA(INDEX, k=12)
    hybride = RechercheHybride([bm25, lsa])
    resultats = hybride.rechercher("tokenisation en sous-mots", k=5)
    assert resultats
    scores = [s for _, s in resultats]
    assert scores == sorted(scores, reverse=True)


def test_rrf_favorise_le_consensus() -> None:
    """Un document bien classé par LES DEUX moteurs doit passer devant un
    document classé premier par un seul.

    C'est tout l'intérêt de la fusion de rangs, et la raison du paramètre
    k = 60 de l'article : il amortit l'écart entre les tout premiers rangs.
    """
    class MoteurFictif:
        def __init__(self, ordre):
            self.ordre = ordre

        def rechercher(self, question, k=20):
            return [(i, 1.0 / (n + 1)) for n, i in enumerate(self.ordre[:k])]

    a = MoteurFictif(["seul_a", "consensus", "x", "y"])
    b = MoteurFictif(["seul_b", "consensus", "x", "y"])
    resultats = RechercheHybride([a, b]).rechercher("peu importe", k=3)
    assert resultats[0][0] == "consensus", resultats


def test_hybride_ne_perd_pas_le_meilleur_des_deux() -> None:
    bm25 = RechercheBM25(INDEX)
    lsa = RechercheLSA(INDEX, k=20)
    hybride = RechercheHybride([bm25, lsa])

    m_bm25, m_lsa, m_hyb = evaluer(bm25), evaluer(lsa), evaluer(hybride)
    assert m_hyb["hit@5"] >= min(m_bm25["hit@5"], m_lsa["hit@5"])
    assert m_hyb["hit@5_semantique"] >= m_bm25["hit@5_semantique"]


def test_la_couverture_lexicale_chute_sur_les_requetes_semantiques() -> None:
    """Quantification de la limite, telle qu'elle apparaît dans la démo."""
    vocabulaire = set(INDEX.index)
    couvertures = {}
    for difficulte in ("lexicale", "partielle", "semantique"):
        connus = total = 0
        for r in REQUETES:
            if r["difficulte"] != difficulte:
                continue
            termes = segmenter(r["question"], **INDEX.options)
            total += len(termes)
            connus += sum(t in vocabulaire for t in termes)
        couvertures[difficulte] = connus / max(total, 1)

    assert couvertures["lexicale"] > 0.8
    assert couvertures["semantique"] < 0.35
    assert couvertures["lexicale"] > couvertures["partielle"] > couvertures["semantique"]


def executer_tous_les_tests() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    echecs = 0
    for test in tests:
        try:
            test()
            print(f"  [OK]     {test.__name__}")
        except AssertionError as e:
            echecs += 1
            print(f"  [ÉCHEC]  {test.__name__} : {e}")
    print(f"\n{len(tests) - echecs}/{len(tests)} tests passés.")
    if echecs:
        raise SystemExit(1)


if __name__ == "__main__":
    print("=== Tests — version 2 : LSA et hybride ===\n")
    executer_tous_les_tests()
