# routes/health.py — Route de santé de l'API
#
# Concept : APIRouter
# Au lieu de tout mettre dans main.py, on découpe les routes en modules.
# FastAPI agrège ensuite tous les routers dans l'application principale.
# Avantage : code organisé, testable indépendamment.

import os
from fastapi import APIRouter
from api.models import HealthResponse

# APIRouter = mini-application FastAPI qui regroupe des routes liées.
# prefix  : tous les chemins de ce router commencent par /health
# tags    : catégorie affichée dans Swagger UI
router = APIRouter(prefix="/health", tags=["Monitoring"])


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Vérifier l'état de l'API",
    description="Retourne le statut de l'API et si ChromaDB est accessible."
)
def health_check() -> HealthResponse:
    """
    GET /health/
    Endpoint de santé — utilisé par les load balancers et systèmes de monitoring.
    """
    chroma_ready = os.path.exists("chroma_db")

    return HealthResponse(
        status="ok" if chroma_ready else "degraded",
        chroma_ready=chroma_ready,
        message=(
            "API opérationnelle, ChromaDB prête."
            if chroma_ready
            else "ChromaDB absente. Lancez d'abord indexer.py."
        )
    )
