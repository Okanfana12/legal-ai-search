# ⚖️ Legal AI — RAG Pipeline for Legal Document Analysis

> ⚠️ **Note importante — Version démo**  
> Ce projet a été développé à titre de démonstration avec les APIs **OpenAI** et **Claude (Anthropic)**.  
> Dans un contexte professionnel impliquant des données sensibles ou personnelles, je privilégierais une stack **100% locale et souveraine** (Mistral via Ollama, embeddings HuggingFace, FAISS local) afin de garantir la confidentialité des données et la conformité RGPD.

---

## 🎯 Objectif

Permettre l'interrogation intelligente de documents juridiques (contrats, jurisprudences, textes réglementaires) via un système RAG sécurisé, évaluable et industrialisable.

---

## 🏗️ Architecture

```
documents/          ← Sources juridiques (PDF, DOCX, TXT)
ingestion/          ← Parsers, chunking, nettoyage
embeddings/         ← Génération et stockage des embeddings
vector_store/       ← FAISS local
rag_pipeline/       ← Chain LangChain / LCEL
evaluation/         ← RAGAS, métriques, benchmarking
api/                ← FastAPI endpoint (optionnel)
```

---

## 🛠️ Stack technique — Version démo

| Composant         | Technologie (démo)                 | Alternative souveraine            |
|-------------------|------------------------------------|-----------------------------------|
| Ingestion         | LangChain Loaders (PDF, DOCX, CSV) | Identique                         |
| Chunking          | RecursiveCharacterTextSplitter     | Identique                         |
| Embeddings        | OpenAI text-embedding-3-small      | HuggingFace local (all-MiniLM)    |
| Vector Store      | FAISS local                        | Identique                         |
| LLM               | OpenAI GPT-4 / Claude Anthropic    | Mistral via Ollama (100% local)   |
| RAG Chain         | LangChain LCEL                     | Identique                         |
| Évaluation        | RAGAS                              | Identique                         |
| Monitoring        | LangSmith                          | Identique                         |

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
# Remplir OPENAI_API_KEY ou ANTHROPIC_API_KEY
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

## 🔒 Vers une version souveraine

Pour un déploiement en environnement sensible (données personnelles, secteur juridique, défense) :

```bash
# Lancer Mistral en local via Ollama
ollama pull mistral

# Utiliser les embeddings HuggingFace locaux
# Aucune donnée ne quitte l'infrastructure
python rag_pipeline/run_local.py
```

Cette configuration garantit :
- ✅ Aucune donnée envoyée à des APIs externes
- ✅ Conformité RGPD
- ✅ Déployable on-premise ou sur cloud privé (AWS, Kubernetes)

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
- [x] Ingestion multi-sources (PDF, DOCX, CSV)
- [x] Évaluation RAGAS
- [x] Versionning code Github Actions/Codespaces
- [x] Déploiement streamlit

---

## 👤 Auteur
Oumou Kanfana


AI Engineer / Data Scientist — spécialisée RAG, LLM, évaluation de modèles  
île de France, Toulouse, France

---

## 📄 Licence

MIT
