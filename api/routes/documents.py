# routes/documents.py — Routes pour les documents indexés
#
# Concepts illustrés :
#   - Path parameters  : /documents/{filename}
#   - Query parameters : /documents?extension=pdf
#   - HTTPException    : retourner une erreur HTTP structurée
#   - Réponse 404      : ressource introuvable

import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from api.models import DocumentInfo, DocumentListResponse

router = APIRouter(prefix="/documents", tags=["Documents"])

DATA_DIR = "data"


# ── Utilitaire interne ──────────────────────────────────────────────────────

def _scan_documents(extension: str | None = None) -> list[DocumentInfo]:
    """Parcourt DATA_DIR et retourne les métadonnées des fichiers."""
    docs = []

    if not os.path.exists(DATA_DIR):
        return docs

    for root, _, files in os.walk(DATA_DIR):
        for filename in sorted(files):
            if extension and not filename.endswith(f".{extension}"):
                continue
            filepath = os.path.join(root, filename)
            size_kb = round(os.path.getsize(filepath) / 1024, 2)
            docs.append(DocumentInfo(
                filename=filename,
                path=filepath,
                size_kb=size_kb
            ))

    return docs


# ── GET /documents/ ─────────────────────────────────────────────────────────
#
# Query parameters : paramètres optionnels dans l'URL après le "?"
# Exemple : GET /documents/?extension=pdf
#
# FastAPI les détecte automatiquement si le paramètre de la fonction
# n'est PAS dans le path ("/documents/{truc}").

@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="Lister les documents disponibles"
)
def list_documents(
    extension: str | None = Query(
        default=None,
        description="Filtrer par extension (ex: pdf, txt)",
        examples=["pdf"]
    )
) -> DocumentListResponse:
    """
    GET /documents/?extension=pdf
    Retourne la liste des documents dans le dossier data/.
    """
    docs = _scan_documents(extension)

    return DocumentListResponse(count=len(docs), documents=docs)


# ── GET /documents/{filename} ───────────────────────────────────────────────
#
# Path parameter : {filename} est une variable extraite de l'URL.
# FastAPI la passe directement comme argument de la fonction.
# Exemple : GET /documents/contrat.pdf

@router.get(
    "/{filename}",
    response_model=DocumentInfo,
    summary="Informations sur un document",
    responses={
        404: {"description": "Document introuvable"}
    }
)
def get_document(filename: str) -> DocumentInfo:
    """
    GET /documents/{filename}
    Retourne les métadonnées d'un document spécifique.
    Lève une 404 si le fichier n'existe pas.
    """
    for root, _, files in os.walk(DATA_DIR):
        if filename in files:
            filepath = os.path.join(root, filename)
            size_kb = round(os.path.getsize(filepath) / 1024, 2)
            return DocumentInfo(
                filename=filename,
                path=filepath,
                size_kb=size_kb
            )

    # HTTPException = façon standard de retourner une erreur HTTP dans FastAPI.
    # status_code : code HTTP (404, 400, 422, 500…)
    # detail      : message lisible par le client
    raise HTTPException(
        status_code=404,
        detail=f"Document '{filename}' introuvable dans {DATA_DIR}/"
    )
