# ⚖️ Legal AI Search

Moteur de recherche juridique basé sur un pipeline RAG (Retrieval-Augmented Generation).  
Pose une question en langage naturel sur tes documents PDF — le système retrouve les passages pertinents et génère une réponse sourcée via Claude.

---

## Architecture

```
PDF → Extraction texte → Chunks → Embeddings OpenAI → ChromaDB
                                                           ↓
Question utilisateur → Embedding → Recherche similarité → Claude → Réponse + Sources
```

| Composant | Technologie |
|-----------|-------------|
| Interface | Streamlit |
| LLM | Claude Sonnet 4.6 (Anthropic) |
| Embeddings | text-embedding-ada-002 (OpenAI) |
| Base vectorielle | ChromaDB |
| Orchestration | LangChain |

---

## Structure du projet

```
legal-ai-search/
├── src/
│   ├── app.py          # Interface Streamlit
│   ├── retriever.py    # Pipeline RAG (recherche + génération)
│   ├── indexer.py      # Indexation des PDFs dans ChromaDB
│   ├── config.py       # Configuration centralisée
│   └── generate_data.py
├── data/
│   └── contrats/       # PDFs à indexer
├── chroma_db/          # Base vectorielle (générée localement)
├── notebooks/
│   └── exploration.ipynb
├── Dockerfile
├── .env.example
└── requirements.txt
```

---

## Installation locale

### 1. Cloner le repo

```bash
git clone https://github.com/Okanfana12/legal-ai-search.git
cd legal-ai-search
```

### 2. Configurer les variables d'environnement

```bash
cp .env.example .env
# Remplir OPENAI_API_KEY et ANTHROPIC_API_KEY dans .env
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Indexer les documents

```bash
python src/indexer.py
```

### 5. Lancer l'application

```bash
streamlit run src/app.py
```

L'app est accessible sur `http://localhost:8501`.

---

## Déploiement Docker (VPS)

### Build et lancement

```bash
docker build -t legal-ai-search .

docker run -d \
  --name legal-ai \
  -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/chroma_db:/app/chroma_db \
  legal-ai-search
```

### Vérifier les logs

```bash
docker logs legal-ai
```

L'app est accessible sur `http://<IP_VPS>:8501`.

---

## Variables d'environnement

| Variable | Description | Obligatoire |
|----------|-------------|-------------|
| `OPENAI_API_KEY` | Clé API OpenAI (embeddings) | Oui |
| `ANTHROPIC_API_KEY` | Clé API Anthropic (LLM) | Oui |
| `CHUNK_SIZE` | Taille des chunks en caractères (défaut : 1000) | Non |
| `CHUNK_OVERLAP` | Chevauchement entre chunks (défaut : 200) | Non |
| `TOP_K_RESULTS` | Nombre de chunks récupérés par recherche (défaut : 5) | Non |
| `CONFIDENCE_THRESHOLD` | Seuil de confiance minimum (défaut : 0.75) | Non |
