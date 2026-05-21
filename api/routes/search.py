# routes/search.py — Route principale : pipeline RAG via REST
#
# Concepts illustrés :
#   - POST avec request body (Pydantic)
#   - Dependency Injection (Depends)
#   - Gestion d'état partagé (lifespan / state)
#   - Codes HTTP : 200, 400, 503

import sys
import os
import logging

from fastapi import APIRouter, HTTPException, Depends, Request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from api.models import SearchRequest, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Recherche RAG"])


# ── Dependency : accès au vector store ─────────────────────────────────────
#
# Dependency Injection (Depends) :
# FastAPI permet de déclarer des dépendances comme fonctions.
# Elles sont résolues automatiquement avant d'appeler le handler.
# Avantages :
#   - Partage de ressources (connexion DB, vector store…)
#   - Testabilité (on peut surcharger la dépendance dans les tests)
#   - Séparation des responsabilités

def get_vector_store(request: Request):
    """
    Dépendance : récupère le vector store initialisé au démarrage.
    Levée de 503 si la base vectorielle n'est pas disponible.
    """
    vector_store = request.app.state.vector_store

    if vector_store is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "ChromaDB non disponible. "
                "Vérifiez que indexer.py a été exécuté d'abord."
            )
        )

    return vector_store


# ── POST /search/ ───────────────────────────────────────────────────────────
#
# Corps de requête (Request Body) :
# Quand la fonction attend un modèle Pydantic (ici SearchRequest),
# FastAPI lit automatiquement le JSON du body et le valide.
# Si la validation échoue → 422 Unprocessable Entity automatique.

@router.post(
    "/",
    response_model=SearchResponse,
    summary="Interroger les documents juridiques",
    description=(
        "Envoie une question au pipeline RAG. "
        "Le système recherche les passages pertinents dans ChromaDB "
        "puis génère une réponse avec Claude."
    ),
    responses={
        200: {"description": "Réponse générée avec succès"},
        400: {"description": "Requête invalide"},
        503: {"description": "ChromaDB non disponible"},
    }
)
def search(
    body: SearchRequest,
    vector_store=Depends(get_vector_store)   # injection de dépendance
) -> SearchResponse:
    """
    POST /search/
    Corps attendu : { "question": "...", "top_k": 5 }
    """
    from src.retriever import answer_question

    logger.info(f"Recherche : '{body.question}' (top_k={body.top_k})")

    try:
        rag_response = answer_question(vector_store, body.question)
    except Exception as exc:
        logger.error(f"Erreur pipeline RAG : {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne du pipeline : {str(exc)}"
        )

    return SearchResponse(
        question=rag_response.question,
        answer=rag_response.answer,
        sources=rag_response.sources,
        confidence=rag_response.confidence,
        is_reliable=rag_response.is_reliable
    )
