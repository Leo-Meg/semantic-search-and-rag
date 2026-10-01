"""
Corpus de documents et jeu d'évaluation embarqués.

Pourquoi je construis mon propre jeu d'évaluation
--------------------------------------------------
Le TP de RAG de M2 (Florent Storme, « RAG - semantic search ») utilisait le jeu
Quora Question Pairs : des paires de questions étiquetées « doublon » ou non.
L'idée est excellente — l'annotation humaine existe déjà, il suffit de la
détourner : un bon moteur de recherche doit retrouver la question dupliquée
parmi toutes les autres.

Je reprends exactement ce protocole, avec un corpus que j'écris moi-même pour
que le dépôt reste exécutable sans téléchargement. J'y ajoute quelque chose que
le TP n'avait pas : des paires **délibérément piégeuses**, où la reformulation
ne partage presque aucun mot avec l'originale. C'est là que se joue toute la
différence entre recherche lexicale et recherche sémantique — et c'est
exactement ce que je veux pouvoir mesurer.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------- #
# La base documentaire
# --------------------------------------------------------------------------- #

# Chaque document est un court paragraphe sur le TAL. Le corpus est petit mais
# thématiquement dense : plusieurs documents parlent de sujets voisins, ce qui
# rend la discrimination non triviale.
DOCUMENTS: list[dict] = [
    {"id": "d01", "titre": "Tokenisation en sous-mots",
     "texte": "La tokenisation en sous-mots découpe les mots en unités plus petites "
              "apprises sur un corpus. BPE fusionne itérativement les paires de "
              "caractères les plus fréquentes. WordPiece utilise un critère de "
              "vraisemblance plutôt qu'une fréquence brute."},
    {"id": "d02", "titre": "Mécanisme d'attention",
     "texte": "L'attention calcule une moyenne pondérée des valeurs, où les poids "
              "viennent de la similarité entre requêtes et clés. Elle permet à "
              "chaque position d'aller chercher directement l'information dont "
              "elle a besoin, quelle que soit la distance."},
    {"id": "d03", "titre": "Réseaux récurrents",
     "texte": "Un réseau récurrent maintient un état caché mis à jour à chaque pas "
              "de temps. Le LSTM ajoute des portes qui contrôlent ce qui est "
              "retenu et ce qui est oublié, ce qui limite la disparition du "
              "gradient sur les longues séquences."},
    {"id": "d04", "titre": "Plongements lexicaux",
     "texte": "Un plongement lexical représente chaque mot par un vecteur dense "
              "appris. Les mots apparaissant dans des contextes similaires "
              "obtiennent des vecteurs proches, conformément à l'hypothèse "
              "distributionnelle."},
    {"id": "d05", "titre": "Perplexité",
     "texte": "La perplexité mesure l'incertitude d'un modèle de langue. Elle vaut "
              "l'exponentielle de l'entropie croisée et s'interprète comme le "
              "nombre moyen de mots entre lesquels le modèle hésite."},
    {"id": "d06", "titre": "Évaluation de la traduction",
     "texte": "BLEU compare les n-grammes d'une traduction candidate à ceux d'une "
              "référence. La métrique est aveugle à l'ordre des groupes de mots "
              "et pénalise les paraphrases correctes."},
    {"id": "d07", "titre": "Étiquetage morphosyntaxique",
     "texte": "L'étiquetage morphosyntaxique attribue à chaque mot sa catégorie "
              "grammaticale. Les corpus Universal Dependencies fournissent des "
              "annotations harmonisées pour une centaine de langues."},
    {"id": "d08", "titre": "Reconnaissance d'entités nommées",
     "texte": "La reconnaissance d'entités nommées repère les personnes, les lieux "
              "et les organisations dans un texte. La convention BIO marque le "
              "début et l'intérieur de chaque entité."},
    {"id": "d09", "titre": "Analyse syntaxique",
     "texte": "Un analyseur syntaxique construit la structure d'une phrase à partir "
              "d'une grammaire. L'algorithme d'Earley traite n'importe quelle "
              "grammaire hors-contexte en temps cubique."},
    {"id": "d10", "titre": "Logique du premier ordre",
     "texte": "La logique du premier ordre ajoute aux connecteurs propositionnels "
              "des quantificateurs et des prédicats. Vérifier la valeur de vérité "
              "d'une formule close dans un modèle fini s'appelle le model checking."},
    {"id": "d11", "titre": "Descente de gradient",
     "texte": "La descente de gradient met à jour les paramètres dans la direction "
              "opposée au gradient de la fonction de coût. Le taux d'apprentissage "
              "contrôle la taille du pas."},
    {"id": "d12", "titre": "Rétropropagation",
     "texte": "La rétropropagation applique la règle de dérivation des fonctions "
              "composées de la sortie vers l'entrée du réseau. Chaque couche "
              "reçoit le gradient de la couche suivante et calcule celui de ses "
              "paramètres."},
    {"id": "d13", "titre": "Modèles de langue n-grammes",
     "texte": "Un modèle n-gramme estime la probabilité d'un mot à partir des n-1 "
              "mots précédents. Le lissage redistribue de la masse de probabilité "
              "vers les séquences jamais observées."},
    {"id": "d14", "titre": "Recherche en faisceau",
     "texte": "La recherche en faisceau conserve les k meilleures hypothèses "
              "partielles à chaque étape de génération. Sans normalisation par la "
              "longueur, elle privilégie les sorties courtes."},
    {"id": "d15", "titre": "Transfert cross-lingue",
     "texte": "Un modèle multilingue entraîné sur plusieurs langues peut être "
              "appliqué à une langue pour laquelle il n'a pas vu de données "
              "annotées. Ce transfert fonctionne d'autant mieux que les langues "
              "sont typologiquement proches."},
    {"id": "d16", "titre": "Fine-tuning et LoRA",
     "texte": "Le fine-tuning adapte un modèle pré-entraîné à une tâche précise. "
              "LoRA n'entraîne que des matrices de rang faible ajoutées aux poids "
              "gelés, ce qui réduit énormément le nombre de paramètres à mettre à "
              "jour."},
    {"id": "d17", "titre": "Apprentissage en contexte",
     "texte": "L'apprentissage en contexte consiste à donner quelques exemples dans "
              "l'invite plutôt qu'à modifier les poids du modèle. La qualité "
              "dépend fortement du choix et de l'ordre des exemples."},
    {"id": "d18", "titre": "Recherche documentaire classique",
     "texte": "Un index inversé associe à chaque terme la liste des documents où il "
              "apparaît. La pondération TF-IDF favorise les termes fréquents dans "
              "un document et rares dans la collection."},
    {"id": "d19", "titre": "BM25",
     "texte": "BM25 est un modèle probabiliste de pertinence qui sature l'effet de "
              "la fréquence d'un terme et normalise par la longueur du document. "
              "Il reste une base très solide malgré son ancienneté."},
    {"id": "d20", "titre": "Analyse sémantique latente",
     "texte": "L'analyse sémantique latente applique une décomposition en valeurs "
              "singulières à la matrice termes-documents. Les dimensions retenues "
              "capturent des regroupements thématiques latents."},
]

# --------------------------------------------------------------------------- #
# Le jeu de requêtes
# --------------------------------------------------------------------------- #

# Chaque requête est associée au document qu'un humain jugerait pertinent.
# Je classe volontairement les requêtes en trois niveaux de difficulté, parce
# que la moyenne globale masque exactement ce qui m'intéresse.
REQUETES: list[dict] = [
    # --- Niveau 1 : reformulation lexicalement proche ---------------------- #
    {"question": "comment fonctionne la tokenisation en sous-mots",
     "pertinent": "d01", "difficulte": "lexicale"},
    {"question": "qu'est-ce que le mécanisme d'attention",
     "pertinent": "d02", "difficulte": "lexicale"},
    {"question": "à quoi sert la perplexité d'un modèle de langue",
     "pertinent": "d05", "difficulte": "lexicale"},
    {"question": "qu'est-ce qu'un index inversé et le TF-IDF",
     "pertinent": "d18", "difficulte": "lexicale"},
    {"question": "comment marche la recherche en faisceau",
     "pertinent": "d14", "difficulte": "lexicale"},
    {"question": "expliquer la rétropropagation dans un réseau",
     "pertinent": "d12", "difficulte": "lexicale"},

    # --- Niveau 2 : vocabulaire partiellement différent -------------------- #
    {"question": "comment découper un mot en morceaux plus petits",
     "pertinent": "d01", "difficulte": "partielle"},
    {"question": "quelle méthode évite d'oublier le début d'une longue phrase",
     "pertinent": "d03", "difficulte": "partielle"},
    {"question": "représenter un mot par un vecteur de nombres",
     "pertinent": "d04", "difficulte": "partielle"},
    {"question": "repérer les noms de personnes et de villes dans un texte",
     "pertinent": "d08", "difficulte": "partielle"},
    {"question": "adapter un gros modèle sans tout réentraîner",
     "pertinent": "d16", "difficulte": "partielle"},
    {"question": "réduire les dimensions d'une matrice termes documents",
     "pertinent": "d20", "difficulte": "partielle"},

    # --- Niveau 3 : (quasi) aucun mot en commun, seulement le sens --------- #
    # J'ai écrit ces requêtes en vérifiant, terme racinisé par terme racinisé,
    # qu'elles ne partagent (presque) rien avec le document attendu. C'est le
    # seul moyen d'isoler ce que la recherche lexicale ne peut PAS faire.
    {"question": "pourquoi un score automatique juge mal une bonne reformulation",
     "pertinent": "d06", "difficulte": "semantique"},
    {"question": "montrer quelques cas résolus au modèle sans rien réentraîner",
     "pertinent": "d17", "difficulte": "semantique"},
    {"question": "un idiome voisin compense-t-il l'absence de corpus annoté",
     "pertinent": "d15", "difficulte": "semantique"},
    {"question": "savoir si tel terme est un nom un verbe ou un adjectif",
     "pertinent": "d07", "difficulte": "semantique"},
    {"question": "tester automatiquement la vérité d'un énoncé sur un petit univers",
     "pertinent": "d10", "difficulte": "semantique"},
    {"question": "ajuster progressivement les coefficients pour réduire l'erreur",
     "pertinent": "d11", "difficulte": "semantique"},
]

# Mots vides du français. Liste volontairement courte : je préfère une liste que
# je peux justifier mot à mot plutôt qu'une liste importée dont je ne connais
# ni le contenu ni la provenance. C'est une leçon de mon TP de L3, où
# `stopwords.words('french')` de NLTK retirait des mots dont j'avais besoin.
MOTS_VIDES = {
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "l", "à", "au", "aux",
    "et", "ou", "que", "qui", "quoi", "dont", "où", "ce", "cet", "cette", "ces",
    "en", "dans", "sur", "sous", "par", "pour", "avec", "sans", "est", "sont",
    "a", "ont", "il", "elle", "ils", "elles", "on", "je", "tu", "nous", "vous",
    "son", "sa", "ses", "leur", "leurs", "plus", "moins", "très", "aussi",
    "n", "ne", "pas", "s", "c", "y", "se", "qu", "j", "m", "t",
}


# --------------------------------------------------------------------------- #
# Prétraitement
# --------------------------------------------------------------------------- #


def normaliser(texte: str) -> str:
    """Minuscules et suppression des diacritiques.

    Retirer les accents est un choix, pas une évidence. Avantage : « modele » et
    « modèle » deviennent le même terme, ce qui aide sur des requêtes tapées
    vite. Inconvénient : on perd la distinction « cote / côte / côté ».

    Pour de la recherche documentaire, l'avantage l'emporte largement — les
    utilisateurs tapent rarement les accents. Pour de l'analyse linguistique,
    ce serait une faute. Je le note parce que c'est typiquement le genre de
    décision qu'on prend sans y penser et qui change les résultats.
    """
    texte = texte.lower()
    decompose = unicodedata.normalize("NFD", texte)
    return "".join(c for c in decompose if unicodedata.category(c) != "Mn")


def segmenter(texte: str, retirer_mots_vides: bool = True,
              racinisation: bool = True) -> list[str]:
    """Texte -> liste de termes indexables."""
    texte = normaliser(texte)
    mots = re.findall(r"[a-z0-9]+", texte)
    if retirer_mots_vides:
        mots = [m for m in mots if m not in MOTS_VIDES and len(m) > 1]
    if racinisation:
        mots = [raciniser(m) for m in mots]
    return mots


def raciniser(mot: str) -> str:
    """Racinisation française minimale, écrite à la main.

    Ce n'est pas un vrai racinisateur (ni Snowball, ni Porter). Je coupe
    quelques suffixes flexionnels très fréquents, du plus long au plus court.

    Pourquoi le faire quand même : sans racinisation, « tokenisation » dans le
    document et « tokeniser » dans la requête sont deux termes sans rapport pour
    un modèle lexical. La racinisation est le seul moyen, avant les plongements,
    de faire se rencontrer les variantes d'un même lemme.

    Pourquoi je l'écris à la main : pour pouvoir montrer ses erreurs. Il coupera
    « mais » en « mai », et c'est très bien qu'on puisse le constater.
    """
    if len(mot) <= 4:
        return mot
    for suffixe in ("issement", "issant", "ations", "ation", "ements", "ement",
                    "atrice", "ateur", "ances", "ance", "ences", "ence",
                    "ismes", "isme", "istes", "iste", "ités", "ité",
                    "eront", "aient", "erais", "erait", "ions", "iez",
                    "ants", "ant", "ents", "ent", "ées", "ée",
                    "er", "ir", "es", "e", "s"):
        if mot.endswith(suffixe) and len(mot) - len(suffixe) >= 4:
            return mot[:-len(suffixe)]
    return mot


def texte_complet(document: dict) -> str:
    """Titre + corps. Le titre compte double, volontairement.

    Un titre est une description condensée et de haute précision : le répéter
    lui donne plus de poids dans les comptes de termes. C'est une forme
    rudimentaire de **recherche par champs** (field-weighted search), que les
    vrais moteurs implémentent avec des coefficients explicites par champ.
    """
    return f"{document['titre']} {document['titre']} {document['texte']}"


def statistiques_corpus() -> dict[str, float]:
    tailles = [len(segmenter(texte_complet(d))) for d in DOCUMENTS]
    vocabulaire = {t for d in DOCUMENTS for t in segmenter(texte_complet(d))}
    return {
        "nb_documents": float(len(DOCUMENTS)),
        "nb_requetes": float(len(REQUETES)),
        "vocabulaire": float(len(vocabulaire)),
        "longueur_moyenne": sum(tailles) / len(tailles),
        "longueur_min": float(min(tailles)),
        "longueur_max": float(max(tailles)),
    }


if __name__ == "__main__":
    print("=== Corpus et jeu d'évaluation ===\n")

    stats = statistiques_corpus()
    for cle, valeur in stats.items():
        print(f"  {cle:>18} : {valeur:.1f}")

    print("\n  Répartition des requêtes par difficulté :")
    from collections import Counter
    for difficulte, n in Counter(r["difficulte"] for r in REQUETES).items():
        print(f"    {difficulte:>12} : {n}")

    print("\n=== Effet du prétraitement ===\n")
    exemple = "Comment découper un mot en morceaux plus petits ?"
    print(f"  requête brute      : {exemple}")
    print(f"  normalisée         : {normaliser(exemple)}")
    print(f"  segmentée (brut)   : {segmenter(exemple, False, False)}")
    print(f"  sans mots vides    : {segmenter(exemple, True, False)}")
    print(f"  + racinisée        : {segmenter(exemple, True, True)}")

    print("\n=== Ce que fait (et rate) ma racinisation ===\n")
    for mot in ("tokenisation", "tokeniser", "tokenise", "apprentissage",
                "apprendre", "gradient", "gradients", "mais", "souris"):
        print(f"  {mot:>15} -> {raciniser(normaliser(mot))}")

    print(
        "\n  Ce qui marche : « tokenisation », « tokeniser » et « tokenise »\n"
        "  convergent vers un radical commun, de même que « gradient » et\n"
        "  « gradients ». C'est tout l'intérêt — sans ça, une requête au verbe\n"
        "  ne trouverait jamais un document au substantif.\n"
        "\n"
        "  Ce qui ne marche pas : « apprentissage » et « apprendre » ne\n"
        "  convergent pas. Aucune règle de troncature ne le pourrait : il y a\n"
        "  alternance de radical. Il faudrait une lemmatisation avec\n"
        "  dictionnaire, ce que fait spaCy et que je n'implémente pas ici.\n"
        "\n"
        "  Et « souris » devient « souri » : sur-racinisation pure et simple,\n"
        "  ma règle croit voir un pluriel. Le seuil de longueur minimale\n"
        "  (4 caractères) protège « mais », mais pas tout.\n"
        "\n"
        "  Je garde ces défauts VISIBLES plutôt que d'importer une boîte noire :\n"
        "  ils expliqueront une partie des échecs que je vais mesurer, et c'est\n"
        "  exactement ce que je veux pouvoir diagnostiquer.\n"
    )

    print("=== Exemples de requêtes, par difficulté ===\n")
    par_document = {d["id"]: d for d in DOCUMENTS}
    for difficulte in ("lexicale", "partielle", "semantique"):
        requete = next(r for r in REQUETES if r["difficulte"] == difficulte)
        document = par_document[requete["pertinent"]]
        termes_requete = set(segmenter(requete["question"]))
        termes_document = set(segmenter(texte_complet(document)))
        commun = termes_requete & termes_document
        print(f"  [{difficulte}]")
        print(f"    requête  : « {requete['question']} »")
        print(f"    attendu  : {document['id']} — {document['titre']}")
        print(f"    termes en commun : {len(commun)}/{len(termes_requete)} "
              f"{sorted(commun) if commun else '(aucun)'}\n")
