import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.75))
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "legal-ai-search")
if __name__ == "__main__":
    print(f"Cle OpenAI : {str(OPENAI_API_KEY)[:10]}...")
    print(f"Modele : {OPENAI_MODEL}")