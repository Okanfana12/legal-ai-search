# config.py
# Point central de configuration du projet
# Toutes les variables sont lues depuis .env
# La cle API ne doit JAMAIS etre ecrite en dur ici

import os
from dotenv import load_dotenv

# Charge les variables du fichier .env en memoire
load_dotenv()

# ── OpenAI ──
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

# ── RAG ──
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.75))

# ── Chemins ──
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

# ── LangSmith ──
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "legal-ai-search")

# ── Verification au demarrage ──
if __name__ == "__main__":
    print(f"Cle OpenAI chargee : {str(OPENAI_API_KEY)[:10]}...")
    print(f"Modele : {OPENAI_MODEL}")
    print(f"Chunk size : {CHUNK_SIZE}")
    print(f"Seuil confiance : {CONFIDENCE_THRESHOLD}")