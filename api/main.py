# main.py — Point d'entrée de l'API FastAPI
#
# ═══════════════════════════════════════════════════════════
# GUIDE APPRENTISSAGE — REST API avec FastAPI
# ═══════════════════════════════════════════════════════════
#
# Lancer l'API :
#   uvicorn api.main:app --reload --port 8000
#
# Documentation interactive :
#   http://localhost:8000/docs      ← Swagger UI
#   http://localhost:8000/redoc     ← ReDoc
#   http://localhost:8000/openapi.json  ← schéma brut
#
# ═══════════════════════════════════════════════════════════

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.routes import health, documents, search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ── Lifespan — Démarrage / Arrêt ───────────────────────────────────────────
#
# Le lifespan remplace les anciens @app.on_event("startup").
# C'est un context manager asynchrone :
#   - Tout avant `yield` s'exécute au démarrage de l'app
#   - Tout après `yield` s'exécute à l'arrêt
#
# Utilisation typique : ouvrir/fermer des connexions coûteuses
# (base de données, vector store, modèles ML…)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── DÉMARRAGE ──
    logger.info("Démarrage de l'API Legal AI Search…")

    # On charge ChromaDB une seule fois et on le stocke dans app.state.
    # app.state = objet partagé entre toutes les requêtes (thread-safe en lecture).
    app.state.vector_store = None

    try:
        from src.retriever import load_vector_store
        app.state.vector_store = load_vector_store()
        logger.info("ChromaDB chargée avec succès.")
    except Exception as exc:
        logger.warning(f"ChromaDB non disponible au démarrage : {exc}")
        logger.warning("L'API démarrera en mode dégradé (recherche désactivée).")

    yield  # ← L'application est opérationnelle ici

    # ── ARRÊT ──
    logger.info("Arrêt de l'API. Nettoyage des ressources…")
    app.state.vector_store = None


# ── Création de l'application ───────────────────────────────────────────────
#
# FastAPI(
#   title       : nom affiché dans Swagger
#   description : description Markdown affichée dans Swagger
#   version     : version de l'API (semver)
#   lifespan    : hook démarrage/arrêt
# )

app = FastAPI(
    title="Legal AI Search API",
    description=(
        "## API REST pour l'interrogation de documents juridiques\n\n"
        "Pipeline RAG : **ChromaDB** + **OpenAI Embeddings** + **Claude Anthropic**\n\n"
        "### Endpoints disponibles\n"
        "- `POST /search/` — Poser une question aux documents\n"
        "- `GET  /documents/` — Lister les documents indexés\n"
        "- `GET  /health/` — Vérifier l'état de l'API\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# ── Middleware CORS ─────────────────────────────────────────────────────────
#
# CORS (Cross-Origin Resource Sharing) :
# Nécessaire quand un front-end (React, Vue…) hébergé sur un domaine différent
# envoie des requêtes vers cette API.
# En dev, on autorise tout (*). En prod, restreindre aux domaines connus.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # En prod : ["https://mon-app.com"]
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Gestionnaire d'erreurs global ───────────────────────────────────────────
#
# @app.exception_handler capture toutes les exceptions d'un type donné.
# Permet de retourner un format JSON cohérent pour toutes les erreurs.

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Erreur non gérée sur {request.url} : {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Erreur interne du serveur",
            "detail": str(exc)
        }
    )


# ── Inclusion des routers ───────────────────────────────────────────────────
#
# include_router agrège les routes définies dans chaque module.
# L'ordre des inclusions n'a pas d'importance pour le routage.

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(search.router)


# ── Route racine ────────────────────────────────────────────────────────────

@app.get("/", tags=["Accueil"], summary="Bienvenue")
def root():
    """
    GET /
    Retourne un message de bienvenue et les liens de documentation.
    """
    return {
        "message": "Legal AI Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "health":    "GET  /health/",
            "documents": "GET  /documents/",
            "search":    "POST /search/",
        }
    }
