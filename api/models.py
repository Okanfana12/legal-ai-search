# models.py — Schémas Pydantic pour la validation des données
#
# Pydantic = bibliothèque de validation de données basée sur les type hints Python.
# FastAPI l'utilise pour :
#   - Valider automatiquement les corps de requêtes (request body)
#   - Sérialiser les réponses JSON
#   - Générer la documentation OpenAPI (Swagger)
#
# Deux catégories de modèles ici :
#   - *Request* : ce que le client envoie au serveur
#   - *Response* : ce que le serveur renvoie au client

from pydantic import BaseModel, Field


# ── Requêtes (Request) ──────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """Corps de la requête POST /search."""

    question: str = Field(
        ...,                              # ... = champ obligatoire
        min_length=5,
        max_length=500,
        description="Question juridique à poser au pipeline RAG",
        examples=["Quelles sont les conditions de rupture d'un contrat ?"]
    )
    top_k: int = Field(
        default=5,
        ge=1,                             # ge = greater or equal
        le=20,
        description="Nombre de documents à récupérer (1–20)"
    )


# ── Réponses (Response) ─────────────────────────────────────────────────────

class SearchResponse(BaseModel):
    """Réponse renvoyée par POST /search."""

    question: str
    answer: str
    sources: list[str]
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score de confiance entre 0 et 1"
    )
    is_reliable: bool


class DocumentInfo(BaseModel):
    """Métadonnées d'un document indexé."""

    filename: str
    path: str
    size_kb: float


class DocumentListResponse(BaseModel):
    """Liste des documents disponibles dans ChromaDB."""

    count: int
    documents: list[DocumentInfo]


class HealthResponse(BaseModel):
    """Réponse du endpoint /health."""

    status: str
    chroma_ready: bool
    message: str


class ErrorResponse(BaseModel):
    """Format standard des erreurs de l'API."""

    error: str
    detail: str
