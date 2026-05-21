# tests/test_api.py — Tests unitaires de l'API FastAPI
#
# ═══════════════════════════════════════════════════════════
# TESTER UNE API FASTAPI
# ═══════════════════════════════════════════════════════════
#
# FastAPI fournit TestClient (basé sur httpx) pour simuler
# des requêtes HTTP sans lancer de vrai serveur.
#
# Lancer les tests :
#   pytest tests/test_api.py -v
#
# ═══════════════════════════════════════════════════════════

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Fixture : client de test ────────────────────────────────────────────────
#
# Une fixture pytest est une fonction qui prépare un contexte de test.
# @pytest.fixture indique à pytest de l'injecter dans les tests qui la demandent.

@pytest.fixture
def client():
    """
    Crée un client de test avec ChromaDB simulée (mock).
    Le mock évite d'avoir une vraie base ChromaDB pour les tests.
    """
    with patch("src.retriever.load_vector_store") as mock_load:
        mock_load.return_value = MagicMock()  # faux vector store

        from api.main import app
        with TestClient(app) as test_client:
            yield test_client


# ── Tests GET / ─────────────────────────────────────────────────────────────

class TestRoot:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_version(self, client):
        data = client.get("/").json()
        assert data["version"] == "1.0.0"

    def test_root_contains_endpoints(self, client):
        data = client.get("/").json()
        assert "search" in data["endpoints"]
        assert "documents" in data["endpoints"]
        assert "health" in data["endpoints"]


# ── Tests GET /health/ ───────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health/")
        assert response.status_code == 200

    def test_health_has_status_field(self, client):
        data = client.get("/health/").json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded")

    def test_health_has_chroma_ready_field(self, client):
        data = client.get("/health/").json()
        assert "chroma_ready" in data
        assert isinstance(data["chroma_ready"], bool)


# ── Tests GET /documents/ ────────────────────────────────────────────────────

class TestDocuments:
    def test_list_documents_returns_200(self, client):
        response = client.get("/documents/")
        assert response.status_code == 200

    def test_list_documents_has_count(self, client):
        data = client.get("/documents/").json()
        assert "count" in data
        assert isinstance(data["count"], int)

    def test_list_documents_filter_by_extension(self, client):
        response = client.get("/documents/?extension=pdf")
        assert response.status_code == 200
        data = response.json()
        # Tous les fichiers retournés doivent être des PDF
        for doc in data["documents"]:
            assert doc["filename"].endswith(".pdf")

    def test_get_document_not_found(self, client):
        response = client.get("/documents/inexistant.pdf")
        assert response.status_code == 404
        assert "introuvable" in response.json()["detail"]


# ── Tests POST /search/ ──────────────────────────────────────────────────────

class TestSearch:
    def test_search_question_trop_courte(self, client):
        """Une question < 5 caractères → 422 Unprocessable Entity."""
        response = client.post("/search/", json={"question": "Non"})
        assert response.status_code == 422

    def test_search_sans_body(self, client):
        """POST sans body → 422."""
        response = client.post("/search/")
        assert response.status_code == 422

    def test_search_top_k_invalide(self, client):
        """top_k=0 (< 1) → 422."""
        response = client.post(
            "/search/",
            json={"question": "Question valide ?", "top_k": 0}
        )
        assert response.status_code == 422

    def test_search_valide_avec_mock(self, client):
        """
        Test d'un appel valide avec le pipeline RAG entièrement mocké.
        On vérifie que l'API retourne le bon format sans appeler les vraies APIs.
        """
        fake_response = MagicMock()
        fake_response.question = "Quelles sont les clauses de résiliation ?"
        fake_response.answer = "Réponse simulée pour le test."
        fake_response.sources = ["contrat.pdf (page 2)"]
        fake_response.confidence = 0.87
        fake_response.is_reliable = True

        with patch("src.retriever.answer_question", return_value=fake_response):
            response = client.post(
                "/search/",
                json={"question": "Quelles sont les clauses de résiliation ?"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["is_reliable"] is True
        assert data["confidence"] == 0.87
        assert len(data["sources"]) == 1


# ── Exemple de test de validation Pydantic ───────────────────────────────────

class TestValidation:
    def test_question_max_length(self, client):
        """Question de 501 caractères → 422."""
        trop_long = "a" * 501
        response = client.post("/search/", json={"question": trop_long})
        assert response.status_code == 422

    def test_top_k_max(self, client):
        """top_k=21 (> 20) → 422."""
        response = client.post(
            "/search/",
            json={"question": "Question valide ?", "top_k": 21}
        )
        assert response.status_code == 422
