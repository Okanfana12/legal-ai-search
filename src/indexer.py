# indexer.py
# Responsabilite unique : indexation des PDFs dans ChromaDB
# Pipeline complet : PDF -> extraction texte -> chunks -> embeddings -> ChromaDB
#
# Pourquoi ce fichier existe :
# Un avocat ne peut pas interroger 2000 PDFs manuellement.
# Ce pipeline lit tous les documents UNE SEULE FOIS,
# les transforme en vecteurs et les stocke dans ChromaDB.
# Ensuite retriever.py peut repondre a n'importe quelle question
# en quelques millisecondes.

import os       # interaction avec le systeme de fichiers
import sys      # manipulation du chemin Python (sys.path)
import logging  # journalisation professionnelle (remplace print)
from pathlib import Path          # manipulation des chemins de fichiers
from dataclasses import dataclass # structure de donnees typee

# PyPDFLoader : lit un PDF page par page et extrait le texte brut
# Alternative possible : pdfplumber (meilleur sur PDFs complexes)
# On choisit PyPDFLoader car il gere bien les PDFs juridiques simples
from langchain_community.document_loaders import PyPDFLoader

# RecursiveCharacterTextSplitter : decoupe le texte en chunks
# "Recursive" signifie qu'il essaie plusieurs separateurs dans l'ordre :
# d'abord "\n\n" (paragraphes), puis "\n" (lignes), puis "." (phrases)
# puis " " (mots) — jusqu'a obtenir des chunks de la bonne taille
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chroma : interface LangChain vers ChromaDB
# ChromaDB est une base de donnees specialisee dans le stockage
# et la recherche de vecteurs (embeddings)
from langchain_chroma import Chroma

# OpenAIEmbeddings : appelle l'API OpenAI pour transformer
# du texte en vecteurs numeriques (liste de 1536 nombres)
# Ces vecteurs capturent le SENS du texte, pas juste les mots
from langchain_openai import OpenAIEmbeddings

# Ajout du dossier src/ au chemin Python
# Necessaire pour que "from config import ..." fonctionne
# quand on lance depuis la racine du projet
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    OPENAI_API_KEY,          # cle API pour authentifier les appels OpenAI
    OPENAI_EMBEDDING_MODEL,  # modele qui transforme texte en vecteurs
    CHUNK_SIZE,              # taille maximale d'un chunk en caracteres
    CHUNK_OVERLAP,           # chevauchement entre chunks consecutifs
    DATA_DIR,                # dossier contenant les PDFs source
    CHROMA_DIR               # dossier ou ChromaDB stocke ses donnees
)

# ── Configuration du logger ──
# format : "2024-04-30 18:00:00 | INFO | message"
# level INFO : affiche INFO, WARNING, ERROR mais pas DEBUG
# DEBUG serait trop verbeux en production
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
# __name__ = nom du module courant ("indexer")
# Permet d'identifier quel fichier a genere quel log
logger = logging.getLogger(__name__)

# ── Constantes locales ──
# Separees de config.py car specifiques aux tests de ce fichier
# Pas dans config.py car elles ne concernent pas l'app entiere
TEST_QUERY = "clause de resiliation contrat"  # question de test apres indexation
TEST_K = 3  # nombre de resultats a retourner lors du test


# ── Exceptions personnalisees ──
class DocumentLoadError(Exception):
    """
    Leve quand un PDF ne peut pas etre charge.

    Pourquoi une exception personnalisee ?
    Exception generique "Exception" ne dit pas QUOI a plante.
    DocumentLoadError dit immediatement : probleme de chargement PDF.
    Facilite le debogage et la gestion d'erreurs dans app.py.
    """
    pass


class VectorStoreError(Exception):
    """
    Leve quand la creation de la base vectorielle echoue.

    Causes possibles :
    - Cle API OpenAI invalide ou epuisee
    - Probleme reseau vers l'API OpenAI
    - Dossier ChromaDB non accessible en ecriture
    """
    pass


# ── Structure de donnees ──
@dataclass
class IndexingResult:
    """
    Encapsule les statistiques du pipeline d'indexation.

    Pourquoi une dataclass ?
    Retourner un dictionnaire {"total_pdfs": 5, ...} n'est pas type.
    Une dataclass force les types et est plus lisible.
    Permet aussi d'ajouter des methodes si besoin plus tard.

    Attributes:
        total_pdfs: nombre de fichiers PDF lus
        total_pages: nombre total de pages extraites
        total_chunks: nombre de chunks stockes dans ChromaDB
        chroma_dir: chemin vers la base ChromaDB creee
    """
    total_pdfs: int
    total_pages: int
    total_chunks: int
    chroma_dir: str


# ── Fonctions ──

def load_pdfs(data_dir: str) -> list:
    """
    Charge tous les PDFs du dossier data/ et ses sous-dossiers.

    Pourquoi rglob("*.pdf") ?
    rglob = recursive glob = cherche dans TOUS les sous-dossiers.
    Permet d'avoir data/contrats/, data/jurisprudences/, data/notes/
    sans avoir a specifier chaque dossier manuellement.

    Args:
        data_dir: chemin vers le dossier racine des documents

    Returns:
        liste de documents LangChain (un element par page de PDF)

    Raises:
        DocumentLoadError: si aucun PDF n'est trouve dans data_dir
    """
    pdf_files = list(Path(data_dir).rglob("*.pdf"))

    # Verification defensive : mieux vaut echouer tot avec un message clair
    # que continuer et avoir une erreur cryptique plus loin
    if not pdf_files:
        raise DocumentLoadError(f"Aucun PDF trouve dans {data_dir}")

    logger.info(f"{len(pdf_files)} PDFs trouves")

    documents = []
    for pdf_path in pdf_files:
        # Fonction privee (prefixe _) car usage interne uniquement
        docs = _load_single_pdf(pdf_path)
        documents.extend(docs)  # extend ajoute tous les elements de la liste

    logger.info(f"Total : {len(documents)} pages chargees")
    return documents


def _load_single_pdf(pdf_path: Path) -> list:
    """
    Charge un seul PDF et enrichit ses metadonnees.

    Pourquoi cette fonction separee ?
    Principe de responsabilite unique : load_pdfs orchestre,
    _load_single_pdf execute. Plus facile a tester et a debugger.

    Pourquoi le prefixe _ ?
    Convention Python : _ indique une fonction "privee"
    destinee a usage interne uniquement dans ce module.
    Ne pas appeler depuis un autre fichier.

    Pourquoi ajouter source et folder en metadonnees ?
    Sans metadonnees, on sait qu'un chunk parle de resiliation
    mais on ne sait pas dans QUEL fichier ni a quelle PAGE.
    Les metadonnees permettent de citer les sources dans la reponse :
    "Selon contrat_01.pdf page 3..."

    Args:
        pdf_path: chemin Path vers le fichier PDF

    Returns:
        liste de pages du document avec metadonnees enrichies
        liste vide si erreur (on continue avec les autres PDFs)
    """
    try:
        logger.info(f"Lecture : {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        # Enrichissement des metadonnees pour chaque page
        for doc in docs:
            doc.metadata["source"] = pdf_path.name        # ex: contrat_01.pdf
            doc.metadata["folder"] = pdf_path.parent.name # ex: contrats

        logger.info(f"   {len(docs)} pages extraites")
        return docs

    except Exception as e:
        # On log l'erreur mais on ne stoppe pas le pipeline
        # Si un PDF est corrompu, on continue avec les autres
        logger.error(f"   Erreur sur {pdf_path.name} : {e}")
        return []  # liste vide = ce PDF est ignore, les autres continuent


def split_documents(documents: list) -> list:
    """
    Decoupe les documents en chunks avec chevauchement.

    Pourquoi le chevauchement (overlap) ?
    Sans overlap :
      Chunk 1 : "...La clause de resiliation prevoit un preavis de"
      Chunk 2 : "30 jours calendaires a compter de..."
      -> L'information est coupee en deux chunks
      -> Le LLM ne comprend pas la phrase complete

    Avec overlap de 200 caracteres :
      Chunk 1 : "...La clause de resiliation prevoit un preavis de"
      Chunk 2 : "un preavis de 30 jours calendaires a compter de..."
      -> L'information apparait dans les deux chunks
      -> Le LLM comprend la phrase complete

    Pourquoi RecursiveCharacterTextSplitter ?
    Il est "intelligent" : il essaie de couper aux endroits naturels
    (paragraphes, phrases) avant de couper en plein milieu d'un mot.

    Args:
        documents: liste de documents LangChain (pages de PDFs)

    Returns:
        liste de chunks — plus nombreux que les pages originales
    """
    logger.info(f"Decoupage — chunk_size={CHUNK_SIZE} overlap={CHUNK_OVERLAP}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,       # max 1000 caracteres par chunk
        chunk_overlap=CHUNK_OVERLAP, # 200 caracteres de chevauchement
        separators=[
            "\n\n",  # 1er essai : couper entre paragraphes
            "\n",    # 2eme essai : couper entre lignes
            ".",     # 3eme essai : couper entre phrases
            " ",     # 4eme essai : couper entre mots
            ""       # dernier recours : couper n'importe ou
        ]
    )

    chunks = splitter.split_documents(documents)
    logger.info(f"{len(chunks)} chunks crees")
    return chunks


def create_vector_store(chunks: list) -> Chroma:
    """
    Vectorise les chunks et les stocke dans ChromaDB.

    Pourquoi persist_directory ?
    Sans persist_directory, ChromaDB stocke les vecteurs en memoire RAM.
    Si le programme redemarre -> tout est perdu -> il faut re-vectoriser.
    Avec persist_directory, ChromaDB sauvegarde sur le disque.
    Redemarrage du programme -> ChromaDB recharge depuis le disque.
    Evite de re-payer les appels API OpenAI a chaque lancement.

    Pourquoi OpenAIEmbeddings sans passer la cle manuellement ?
    LangChain lit automatiquement OPENAI_API_KEY depuis les variables
    d'environnement. Passer la cle manuellement via openai_api_key=
    peut causer des conflits avec le client async dans les nouvelles versions.
    La bonne pratique est de laisser LangChain la lire lui-meme.

    Args:
        chunks: liste de chunks LangChain prets a vectoriser

    Returns:
        base vectorielle Chroma persistee sur le disque

    Raises:
        VectorStoreError: si la vectorisation ou le stockage echoue
    """
    try:
        logger.info(f"Vectorisation avec {OPENAI_EMBEDDING_MODEL}...")
        logger.info(f"   Cela peut prendre quelques minutes...")

        # LangChain lit OPENAI_API_KEY automatiquement depuis .env
        embeddings = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL
        )

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )

        logger.info(f"{len(chunks)} chunks stockes dans {CHROMA_DIR}/")
        return vector_store

    except Exception as e:
        raise VectorStoreError(f"Echec vectorisation : {e}")



def test_search(vector_store: Chroma) -> None:
    """
    Valide le pipeline avec une recherche test.

    Pourquoi cette fonction ?
    Apres l'indexation, on ne sait pas si ChromaDB fonctionne
    correctement sans faire une vraie recherche.
    Ce test rapide confirme que :
    1. Les embeddings sont bien stockes
    2. La recherche par similarite retourne des resultats
    3. Les metadonnees (source, page) sont bien presentes

    C'est un "smoke test" — test minimal qui valide le flux principal.

    Args:
        vector_store: base vectorielle ChromaDB fraichement creee
    """
    logger.info(f"Test recherche : '{TEST_QUERY}'")

    # similarity_search : transforme la query en vecteur
    # puis cherche les k chunks les plus proches dans ChromaDB
    results = vector_store.similarity_search(TEST_QUERY, k=TEST_K)

    for i, doc in enumerate(results):
        logger.info(
            f"  [{i+1}] {doc.metadata.get('source', 'inconnu')} "
            f"p.{doc.metadata.get('page', '?')} | "
            f"{doc.page_content[:100]}..."  # affiche les 100 premiers caracteres
        )


def main() -> IndexingResult:
    """
    Point d'entree du script — orchestre le pipeline complet.

    Pourquoi retourner IndexingResult ?
    main() pourrait etre appellee depuis un autre script
    (ex: un orchestrateur qui veut savoir combien de chunks ont ete crees).
    Retourner un objet structure est plus propre que de tout logger
    et plus flexible qu'une variable globale.

    Returns:
        IndexingResult contenant les statistiques de l'indexation
    """
    logger.info("=" * 50)
    logger.info("LEGAL AI SEARCH — Pipeline d'indexation")
    logger.info("=" * 50)

    # Etape 1 — Charger les PDFs depuis data/
    documents = load_pdfs(DATA_DIR)

    # Etape 2 — Decouper en chunks avec overlap
    chunks = split_documents(documents)

    # Etape 3 — Vectoriser et stocker dans ChromaDB
    vector_store = create_vector_store(chunks)

    # Etape 4 — Valider avec une recherche test
    test_search(vector_store)

    # Etape 5 — Compiler et retourner les statistiques
    result = IndexingResult(
        total_pdfs=len(list(Path(DATA_DIR).rglob("*.pdf"))),
        total_pages=len(documents),
        total_chunks=len(chunks),
        chroma_dir=CHROMA_DIR
    )

    logger.info("=" * 50)
    logger.info(f"Indexation terminee : {result.total_chunks} chunks prets")
    logger.info("=" * 50)

    return result


if __name__ == "__main__":
    # Ce bloc ne s'execute QUE si on lance directement ce fichier :
    # python src/indexer.py  -> s'execute
    # from indexer import load_pdfs -> ne s'execute PAS
    # Permet d'importer les fonctions sans lancer le pipeline entier
    main()