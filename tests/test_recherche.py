"""
Tests de la version 1 — index inversé, TF-IDF, BM25, métriques.

    python -m tests.test_recherche
"""

from __future__ import annotations

import math

from src.corpus import (
    DOCUMENTS,
    REQUETES,
    normaliser,
    raciniser,
    segmenter,
    texte_complet,
)
from src.recherche import (
    IndexInverse,
    RechercheBM25,
    RechercheTFIDF,
    evaluer,
    hit_rate,
    mean_average_precision,
    mean_reciprocal_rank,
)

INDEX = IndexInverse(DOCUMENTS)


# --------------------------------------------------------------------------- #
# Prétraitement
# --------------------------------------------------------------------------- #


def test_normalisation_retire_les_diacritiques() -> None:
    assert normaliser("Modèle Évalué à Paris") == "modele evalue a paris"


def test_segmentation_retire_mots_vides_et_ponctuation() -> None:
    termes = segmenter("Le modèle, est-il bon ?", racinisation=False)
    assert "le" not in termes and "est" not in termes
    assert "modele" in termes


def test_racinisation_fait_converger_les_variantes() -> None:
    racines = {raciniser(normaliser(m))
               for m in ("tokenisation", "tokeniser", "tokenise")}
    assert len(racines) == 1, racines
    assert raciniser("gradient") == raciniser("gradients")


def test_racinisation_protege_les_mots_courts() -> None:
    for mot in ("mais", "pas", "vie", "or"):
        assert raciniser(mot) == mot


# --------------------------------------------------------------------------- #
# Index inversé
# --------------------------------------------------------------------------- #


def test_index_couvre_tous_les_documents() -> None:
    assert INDEX.nb_documents == len(DOCUMENTS)
    assert set(INDEX.longueurs) == {d["id"] for d in DOCUMENTS}
    assert all(v > 0 for v in INDEX.longueurs.values())


def test_index_coherent_avec_les_documents() -> None:
    """Chaque entrée de l'index doit correspondre à un vrai comptage."""
    document = DOCUMENTS[0]
    termes = segmenter(texte_complet(document))
    for terme in set(termes):
        assert INDEX.index[terme][document["id"]] == termes.count(terme)


def test_idf_nul_pour_un_terme_omnipresent() -> None:
    """Un terme présent partout ne distingue rien : TF-IDF doit l'annuler."""
    documents = [{"id": f"x{i}", "titre": "commun", "texte": "commun unique%d" % i}
                 for i in range(5)]
    index = IndexInverse(documents, retirer_mots_vides=False, racinisation=False)
    assert math.isclose(index.idf("commun"), 0.0)


def test_idf_decroit_avec_la_frequence_documentaire() -> None:
    frequences = sorted(INDEX.frequences_document.items(), key=lambda x: x[1])
    rare, frequent = frequences[0][0], frequences[-1][0]
    assert INDEX.idf(rare) > INDEX.idf(frequent)


def test_idf_bm25_toujours_positif() -> None:
    """Sans le « + 1 », un terme présent dans plus de la moitié des documents
    recevrait un poids NÉGATIF — le contenir ferait baisser le score."""
    for terme in INDEX.index:
        assert INDEX.idf(terme, lissage="bm25") > 0.0, terme


def test_candidats_filtre_bien() -> None:
    termes = segmenter("tokenisation sous-mots")
    candidats = INDEX.candidats(termes)
    assert "d01" in candidats
    assert len(candidats) < INDEX.nb_documents


# --------------------------------------------------------------------------- #
# Moteurs
# --------------------------------------------------------------------------- #


def test_tfidf_trouve_le_document_evident() -> None:
    moteur = RechercheTFIDF(INDEX)
    resultats = moteur.rechercher("tokenisation en sous-mots BPE WordPiece", k=3)
    assert resultats[0][0] == "d01"


def test_bm25_trouve_le_document_evident() -> None:
    moteur = RechercheBM25(INDEX)
    resultats = moteur.rechercher("index inversé pondération TF-IDF", k=3)
    assert resultats[0][0] == "d18"


def test_scores_tries_par_ordre_decroissant() -> None:
    for moteur in (RechercheTFIDF(INDEX), RechercheBM25(INDEX)):
        scores = [s for _, s in moteur.rechercher("attention transformer", k=5)]
        assert scores == sorted(scores, reverse=True)


def test_cosinus_borne_entre_zero_et_un() -> None:
    moteur = RechercheTFIDF(INDEX)
    for _, score in moteur.rechercher("attention perplexité gradient", k=10):
        assert 0.0 <= score <= 1.0 + 1e-9


def test_requete_sans_terme_connu_ne_plante_pas() -> None:
    for moteur in (RechercheTFIDF(INDEX), RechercheBM25(INDEX)):
        assert moteur.rechercher("zzzz qqqq wwww", k=5) == []


def test_bm25_sature_la_frequence() -> None:
    """La propriété qui distingue BM25 de TF-IDF : le score plafonne.

    Je compare un document où le terme apparaît 1 fois à un document où il
    apparaît 20 fois. Le score ne doit PAS être 20 fois plus grand.
    """
    documents = [
        {"id": "peu", "titre": "sujet", "texte": "attention " + "remplissage " * 19},
        {"id": "beaucoup", "titre": "sujet", "texte": "attention " * 20},
    ]
    index = IndexInverse(documents, retirer_mots_vides=False, racinisation=False)
    scores = dict(RechercheBM25(index).rechercher("attention", k=2))
    rapport = scores["beaucoup"] / scores["peu"]
    assert 1.0 < rapport < 3.0, f"saturation attendue, rapport obtenu : {rapport:.2f}"


def test_bm25_normalise_par_la_longueur() -> None:
    """Avec b = 1, un document long est pénalisé ; avec b = 0, non."""
    documents = [
        {"id": "court", "titre": "x", "texte": "attention"},
        {"id": "long", "titre": "x", "texte": "attention " + "bruit " * 50},
    ]
    index = IndexInverse(documents, retirer_mots_vides=False, racinisation=False)

    sans = dict(RechercheBM25(index, b=0.0).rechercher("attention", k=2))
    avec = dict(RechercheBM25(index, b=1.0).rechercher("attention", k=2))

    assert math.isclose(sans["court"], sans["long"], rel_tol=1e-9)
    assert avec["court"] > avec["long"]


# --------------------------------------------------------------------------- #
# Métriques
# --------------------------------------------------------------------------- #


def test_hit_rate() -> None:
    resultats = [["a", "b", "c"], ["x", "y", "z"]]
    pertinents = ["b", "w"]
    assert hit_rate(resultats, pertinents, 1) == 0.0
    assert hit_rate(resultats, pertinents, 2) == 0.5
    assert hit_rate(resultats, pertinents, 3) == 0.5


def test_mrr_sensible_au_rang() -> None:
    assert mean_reciprocal_rank([["a", "b"]], ["a"]) == 1.0
    assert mean_reciprocal_rank([["b", "a"]], ["a"]) == 0.5
    assert mean_reciprocal_rank([["b", "c"]], ["a"]) == 0.0


def test_map_coincide_avec_mrr_a_un_seul_pertinent() -> None:
    resultats = [["a", "b", "c"], ["x", "a", "z"], ["q", "w", "e"]]
    pertinents = ["a", "a", "a"]
    assert math.isclose(mean_average_precision(resultats, pertinents),
                        mean_reciprocal_rank(resultats, pertinents))


def test_hit_rate_insensible_au_rang_contrairement_au_mrr() -> None:
    """La distinction que le TP de M2 faisait apparaître sans l'expliquer.

    Deux systèmes peuvent avoir le même Hit Rate et des MRR très différents.
    """
    premier = [["a", "b", "c", "d", "e"]]
    dernier = [["b", "c", "d", "e", "a"]]
    assert hit_rate(premier, ["a"], 5) == hit_rate(dernier, ["a"], 5) == 1.0
    assert mean_reciprocal_rank(premier, ["a"]) > mean_reciprocal_rank(dernier, ["a"])


# --------------------------------------------------------------------------- #
# Le résultat central de la version
# --------------------------------------------------------------------------- #


def test_les_moteurs_lexicaux_echouent_sur_les_requetes_semantiques() -> None:
    """Le constat qui motive toute la suite du projet.

    Les requêtes de niveau « sémantique » ne partagent presque aucun terme avec
    le document attendu. TF-IDF et BM25 y échouent, par construction : ils ne
    savent comparer que des chaînes de caractères.
    """
    for moteur in (RechercheTFIDF(INDEX), RechercheBM25(INDEX)):
        mesures = evaluer(moteur)
        assert mesures["hit@5_lexicale"] >= 0.9, mesures
        assert mesures["hit@5_semantique"] <= 0.6, mesures
        assert mesures["hit@5_lexicale"] > mesures["hit@5_semantique"] + 0.3


def test_le_pretraitement_change_les_resultats() -> None:
    """Le sujet de mon projet de L3, enfin mesuré."""
    brut = evaluer(RechercheBM25(IndexInverse(
        DOCUMENTS, retirer_mots_vides=False, racinisation=False)))
    complet = evaluer(RechercheBM25(IndexInverse(
        DOCUMENTS, retirer_mots_vides=True, racinisation=True)))
    assert complet["hit@5"] > brut["hit@5"], (brut["hit@5"], complet["hit@5"])


def test_toutes_les_requetes_ont_un_document_valide() -> None:
    identifiants = {d["id"] for d in DOCUMENTS}
    for requete in REQUETES:
        assert requete["pertinent"] in identifiants
        assert requete["difficulte"] in {"lexicale", "partielle", "semantique"}


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
    print("=== Tests — version 1 : recherche lexicale ===\n")
    executer_tous_les_tests()
