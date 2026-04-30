# retriever.py
# Responsabilite unique : recherche + scoring + generation de reponse
# Pipeline : question -> embedding -> ChromaDB -> scoring -> Claude -> reponse

import os
import sys
import logging
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    OPENAI_EMBEDDING_MODEL,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    TOP_K_RESULTS,
    CONFIDENCE_THRESHOLD,
    CHROMA_DIR
)

# ── Logger ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Constantes ──
PROMPT_TEMPLATE = """Tu es un assistant juridique expert.
Reponds a la question en te basant UNIQUEMENT sur les documents fournis.
Si la reponse n'est pas dans les documents, dis-le clairement.

Documents pertinents :
{context}

Question : {question}

Reponse detaillee avec sources :"""


# ── Structures de donnees ──
@dataclass
class SearchResult:
    """Resultat d'une recherche dans ChromaDB."""
    chunks: list[Document]
    scores: list[float]
    query: str


@dataclass
class RAGResponse:
    """Reponse complete du pipeline RAG."""
    question: str
    answer: str
    sources: list[str]
    confidence: float
    is_reliable: bool


# ── Exceptions ──
class RetrieverError(Exception):
    """Erreur lors de la recherche ou generation."""
    pass


# ── Fonctions ──
def load_vector_store() -> Chroma:
    """
    Charge la base vectorielle ChromaDB depuis le disque.

    Pourquoi charger depuis le disque ?
    L'indexation est faite une seule fois par indexer.py.
    retriever.py recharge la base existante sans re-vectoriser.
    Economie de temps et de credits API.

    Returns:
        base vectorielle ChromaDB chargee

    Raises:
        RetrieverError: si ChromaDB n'existe pas
    """
    if not os.path.exists(CHROMA_DIR):
        raise RetrieverError(
            f"ChromaDB introuvable dans {CHROMA_DIR}. "
            f"Lance d'abord indexer.py"
        )

    logger.info(f"Chargement ChromaDB depuis {CHROMA_DIR}...")

    embeddings = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL
    )

    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    logger.info("ChromaDB charge")
    return vector_store


def search_documents(
    vector_store: Chroma,
    query: str,
    k: int = TOP_K_RESULTS
) -> SearchResult:
    """
    Recherche les chunks les plus pertinents pour une question.

    Pourquoi similarity_search_with_relevance_scores ?
    Retourne les chunks ET leur score de similarite (0 a 1).
    Score 1.0 = parfaitement similaire a la question.
    Score 0.0 = aucun rapport avec la question.
    Le score permet d'evaluer la fiabilite de la reponse.

    Args:
        vector_store: base vectorielle ChromaDB
        query: question de l'utilisateur
        k: nombre de chunks a recuperer

    Returns:
        SearchResult avec chunks et scores
    """
    logger.info(f"Recherche : '{query}'")

    results = vector_store.similarity_search_with_relevance_scores(
        query, k=k
    )

    chunks = [doc for doc, score in results]
    scores = [score for doc, score in results]

    logger.info(f"{len(chunks)} chunks trouves")
    for i, (chunk, score) in enumerate(zip(chunks, scores)):
        logger.info(
            f"  [{i+1}] score={score:.3f} | "
            f"{chunk.metadata.get('source')} "
            f"p.{chunk.metadata.get('page')}"
        )

    return SearchResult(chunks=chunks, scores=scores, query=query)


def compute_confidence(scores: list[float]) -> float:
    """
    Calcule le score de confiance moyen des chunks recuperes.

    Pourquoi la moyenne des scores ?
    Un seul chunk tres pertinent ne suffit pas.
    On veut que PLUSIEURS chunks soient pertinents
    pour avoir confiance en la reponse.

    Args:
        scores: liste de scores de similarite (0 a 1)

    Returns:
        score de confiance moyen entre 0 et 1
    """
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def build_context(chunks: list[Document]) -> str:
    """
    Construit le contexte a injecter dans le prompt Claude.

    Pourquoi formater le contexte ?
    Claude doit savoir quelle information vient de quel document.
    Le format "Source: fichier.pdf | Page: X" permet a Claude
    de citer correctement ses sources dans la reponse.

    Args:
        chunks: liste de chunks LangChain

    Returns:
        contexte formate pret pour le prompt
    """
    context_parts = []

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "inconnu")
        page = chunk.metadata.get("page", "?")
        content = chunk.page_content.strip()

        context_parts.append(
            f"[Document {i+1}] Source: {source} | Page: {page}\n{content}"
        )

    return "\n\n---\n\n".join(context_parts)


def generate_response(
    query: str,
    context: str
) -> str:
    """
    Genere une reponse avec Claude 3 Opus.

    Pourquoi Claude 3 Opus ?
    Meilleur modele Anthropic pour les taches complexes.
    Excellent pour les documents juridiques.
    Respecte les sources et ne hallucine pas si bien guide.

    Args:
        query: question de l'utilisateur
        context: chunks pertinents formates

    Returns:
        reponse generee par Claude
    """
    logger.info(f"Generation avec {ANTHROPIC_MODEL}...")

    llm = ChatAnthropic(
        model=ANTHROPIC_MODEL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        max_tokens=1024,
        temperature=0
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": query
    })

    return response.content


def extract_sources(chunks: list[Document]) -> list[str]:
    """
    Extrait les sources uniques des chunks recuperes.

    Args:
        chunks: liste de chunks LangChain

    Returns:
        liste de sources formatees "fichier.pdf (page X)"
    """
    sources = []
    seen = set()

    for chunk in chunks:
        source = chunk.metadata.get("source", "inconnu")
        page = chunk.metadata.get("page", "?")
        ref = f"{source} (page {page})"

        if ref not in seen:
            sources.append(ref)
            seen.add(ref)

    return sources


def answer_question(
    vector_store: Chroma,
    question: str
) -> RAGResponse:
    """
    Pipeline RAG complet : question -> reponse sourcee.

    Orchestre toutes les etapes :
    1. Recherche des chunks pertinents
    2. Calcul du score de confiance
    3. Verification du seuil de confiance
    4. Generation de la reponse par Claude
    5. Extraction des sources

    Args:
        vector_store: base vectorielle ChromaDB
        question: question de l'utilisateur

    Returns:
        RAGResponse avec reponse, sources et score de confiance
    """
    # Etape 1 — Recherche
    search_result = search_documents(vector_store, question)

    # Etape 2 — Score de confiance
    confidence = compute_confidence(search_result.scores)
    is_reliable = confidence >= CONFIDENCE_THRESHOLD

    logger.info(f"Score de confiance : {confidence:.3f} "
                f"(seuil : {CONFIDENCE_THRESHOLD})")

    # Etape 3 — Verification fiabilite
    if not is_reliable:
        logger.warning("Score insuffisant — reponse incertaine")
        return RAGResponse(
            question=question,
            answer=(
                "Je ne trouve pas d'information suffisamment fiable "
                "dans les documents pour repondre a cette question. "
                "Veuillez consulter directement les documents ou "
                "reformuler votre question."
            ),
            sources=[],
            confidence=confidence,
            is_reliable=False
        )

    # Etape 4 — Construction du contexte
    context = build_context(search_result.chunks)

    # Etape 5 — Generation par Claude
    answer = generate_response(question, context)

    # Etape 6 — Extraction des sources
    sources = extract_sources(search_result.chunks)

    return RAGResponse(
        question=question,
        answer=answer,
        sources=sources,
        confidence=confidence,
        is_reliable=True
    )


def main() -> None:
    """Point d'entree — teste le pipeline RAG complet."""

    logger.info("=" * 50)
    logger.info("LEGAL AI SEARCH — Pipeline de recherche")
    logger.info("=" * 50)

    # Chargement ChromaDB
    vector_store = load_vector_store()

    # Questions de test
    questions = [
        "Quelles sont les conditions de rupture d'un contrat de travail ?",
        "Quelles sont les obligations de l'employeur ?",
        "Comment fonctionne le preavis de licenciement ?",
    ]

    for question in questions:
        logger.info(f"\n{'='*50}")
        logger.info(f"Question : {question}")

        response = answer_question(vector_store, question)

        logger.info(f"Fiable : {response.is_reliable}")
        logger.info(f"Confiance : {response.confidence:.3f}")
        logger.info(f"Reponse : {response.answer[:200]}...")
        logger.info(f"Sources : {response.sources}")


if __name__ == "__main__":
    main()