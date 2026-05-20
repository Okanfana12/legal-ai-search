# legal-ai-search

# ⚖️ Legal AI — RAG Pipeline for Legal Document Analysis

A production-ready RAG (Retrieval-Augmented Generation) pipeline designed for legal document analysis, built with LangChain, FAISS, and LLM integration (OpenAI / Claude).

---

## 🎯 Objectif

Permettre l'interrogation intelligente de documents juridiques (contrats, jurisprudences, textes réglementaires) via un système RAG sécurisé, évaluable et industrialisable.

---

## 🏗️ Architecture

```
documents/          ← Sources juridiques (PDF, DOCX, TXT)
ingestion/          ← Parsers, chunking, nettoyage
embeddings/         ← Génération et stockage des embeddings
vector_store/       ← FAISS / Chroma (local & offline)
rag_pipeline/       ← Chain LangChain / LCEL
evaluation/         ← RAGAS, métriques, benchmarking
api/                ← FastAPI endpoint (optionnel)
```

---

## 🛠️ Stack technique

| Composant         | Technologie                        |
|-------------------|------------------------------------|
| Ingestion         | LangChain Loaders (PDF, DOCX, CSV) |
| Chunking          | RecursiveCharacterTextSplitter     |
| Embeddings        | OpenAI / Claude opus 3       |
| Vector Store      | FAISS (offline) / Pinecone (cloud) |
| LLM               | OpenAI GPT-4 / claude    |
| RAG Chain         | LangChain LCEL                     |
| Évaluation        | RAGAS                              |
| Monitoring        | LangSmith                          |
| Environnement     | Python 3.11+, UV, dotenv           |

---

## ⚙️ Installation

```bash
# Cloner le repo
git clone https://github.com/username/legal-ai.git
cd legal-ai

# Créer l'environnement virtuel
uv venv
source .venv/bin/activate

# Installer les dépendances
uv pip install -r requirements.txt

# Configurer les variables d'environnement
cp .env.example .env
# Remplir OPENAI_API_KEY, LANGCHAIN_API_KEY, etc.
```

---

## 🚀 Utilisation

```python
# Ingestion des documents
python ingestion/ingest.py --source documents/

# Lancer le pipeline RAG
python rag_pipeline/run.py --query "Quelles sont les clauses de résiliation ?"

# Évaluation RAGAS
python evaluation/evaluate.py
```

---

## 📊 Évaluation & Benchmarking

Le pipeline est évalué via **RAGAS** sur 4 métriques clés :

- **Faithfulness** — la réponse est-elle fidèle aux documents sources ?
- **Answer Relevancy** — la réponse répond-elle à la question ?
- **Context Precision** — les chunks récupérés sont-ils pertinents ?
- **Context Recall** — le contexte nécessaire est-il bien récupéré ?

```bash
python evaluation/evaluate.py --dataset data/eval_dataset.json
```

---

## 🔒 Souveraineté des données

Ce pipeline est conçu pour fonctionner **100% en local** :

- Embeddings via HuggingFace (offline)
- LLM via **Mistral + Ollama** (aucune donnée envoyée en dehors)
- Vector store FAISS local

```bash
# Lancer Mistral en local
ollama pull mistral
python rag_pipeline/run_local.py
```

---

## 📁 Données d'exemple

```
documents/
├── contrat_type.pdf
├── jurisprudence_2024.docx
└── rgpd_texte_officiel.txt
```

---

## 🧪 Tests

```bash
pytest tests/
```

---

## 📌 Roadmap

- [x] Pipeline RAG de base
- [x] Ingestion multi-sources (PDF, DOCX, CSV, SQL)
- [x] Évaluation RAGAS
- [x] Mode offline Mistral
- [x] Interface Streamlit
- [x] Agent LangGraph pour workflow juridique complexe
- [x] Déploiement Kubernetes

---

## 👤 Auteur

AI Engineer / Data Scientist — spécialisée RAG, LLM, évaluation de modèles  
Toulouse, France

---

## 📄 Licence

MIT
