# ⚖️ Legal AI Search

> Moteur de recherche juridique intelligent basé sur un pipeline RAG (Retrieval-Augmented Generation).  
> Posez une question en langage naturel sur vos documents PDF — le système retrouve les passages les plus pertinents et génère une réponse détaillée et sourcée grâce à Claude (Anthropic).

> ⚠️ **Note importante — Version démo**  
> Ce projet utilise les APIs **OpenAI** (embeddings) et **Claude (Anthropic)** (génération).  
> Pour un déploiement en environnement sensible (données personnelles, RGPD), voir la section [Vers une version souveraine](#vers-une-version-souveraine).

---

## Table des matières

- [Aperçu](#aperçu)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Structure du projet](#structure-du-projet)
- [Prérequis](#prérequis)
- [Installation locale](#installation-locale)
- [Utilisation](#utilisation)
- [Déploiement Docker (VPS)](#déploiement-docker-vps)
- [Vers une version souveraine](#vers-une-version-souveraine)

---

## Aperçu

Legal AI Search permet à des juristes, avocats ou équipes compliance d'interroger un corpus de documents juridiques (contrats, jurisprudences, notes internes) sans avoir à les lire manuellement.

Le système :
1. **Indexe** les PDFs une seule fois et les stocke sous forme de vecteurs dans ChromaDB
2. **Recherche** les passages les plus pertinents à chaque question via similarité cosinus
3. **Génère** une réponse en langage naturel avec citation des sources via Claude Sonnet 4.6

---

## Fonctionnalités

- Recherche sémantique sur l'ensemble du corpus de documents juridiques
- Réponses sourcées avec référence au fichier PDF et à la page
- Score de confiance affiché pour chaque réponse
- Refus de répondre si le score de confiance est insuffisant (évite les hallucinations)
- Interface web intuitive via Streamlit
- Base vectorielle persistante — pas de re-vectorisation à chaque démarrage
- Déploiement containerisé via Docker

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PHASE D'INDEXATION                      │
│                   (exécutée une seule fois)                 │
│                                                             │
│  PDFs  ──►  PyPDFLoader  ──►  Text Splitter  ──►  Chunks   │
│                                                      │      │
│                                          OpenAI Embeddings  │
│                                                      │      │
│                                                  ChromaDB ◄─┘
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   PHASE DE RECHERCHE                        │
│                  (exécutée à chaque question)               │
│                                                             │
│  Question  ──►  OpenAI Embeddings  ──►  Vecteur requête     │
│                                               │             │
│                                    Similarité cosinus       │
│                                               │             │
│                                          ChromaDB           │
│                                               │             │
│                                    Top-K chunks + scores    │
│                                               │             │
│                             Score confiance ≥ seuil ?       │
│                                   │              │          │
│                                  OUI            NON         │
│                                   │              │          │
│                              Claude LLM    Réponse vide     │
│                                   │        + avertissement  │
│                              Réponse +                      │
│                               Sources                       │
└─────────────────────────────────────────────────────────────┘
```

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| Interface utilisateur | Streamlit | Application web |
| LLM | Claude Sonnet 4.6 (Anthropic) | Génération des réponses |
| Embeddings | text-embedding-ada-002 (OpenAI) | Vectorisation du texte |
| Base vectorielle | ChromaDB | Stockage et recherche de vecteurs |
| Orchestration | LangChain | Pipeline RAG |
| Lecture PDFs | PyPDFLoader | Extraction du texte |

---

## Structure du projet

```
legal-ai-search/
│
├── src/
│   ├── app.py              # Interface Streamlit (point d'entrée utilisateur)
│   ├── retriever.py        # Pipeline RAG : recherche + scoring + génération
│   ├── indexer.py          # Indexation PDFs → chunks → embeddings → ChromaDB
│   ├── config.py           # Configuration centralisée (modèles, seuils, chemins)
│   └── generate_data.py    # Génération de données de test
│
├── data/
│   └── contrats/           # Dossier contenant les PDFs à indexer
│       ├── contrat_01.pdf
│       └── ...
│
├── chroma_db/              # Base vectorielle persistée sur disque (généré automatiquement)
│
├── notebooks/
│   └── exploration.ipynb   # Exploration et tests du pipeline
│
├── Dockerfile              # Image Docker pour déploiement VPS
├── .env.example            # Template des variables d'environnement
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Stack technique

| Composant         | Technologie (démo)                 | Alternative souveraine            |
|-------------------|------------------------------------|-----------------------------------|
| Ingestion         | LangChain Loaders (PDF, DOCX, CSV) | Identique                         |
| Chunking          | RecursiveCharacterTextSplitter     | Identique                         |
| Embeddings        | OpenAI text-embedding-ada-002      | HuggingFace local (all-MiniLM)    |
| Vector Store      | ChromaDB                           | FAISS local                       |
| LLM               | Claude Sonnet 4.6 (Anthropic)      | Mistral via Ollama (100% local)   |
| RAG Chain         | LangChain LCEL                     | Identique                         |

---

## Prérequis

- Python 3.11+
- Une clé API **OpenAI** (pour les embeddings)
- Une clé API **Anthropic** (pour la génération de réponses)
- Docker (pour le déploiement VPS uniquement)

---

## Installation locale

### 1. Cloner le dépôt

```bash
git clone https://github.com/Okanfana12/legal-ai-search.git
cd legal-ai-search
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

```bash
cp .env.example .env
```

Ouvrir `.env` et renseigner les clés :

```env
PYTHONIOENCODING=utf-8
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Ajouter vos documents PDF

Placer vos fichiers PDF dans le dossier `data/contrats/` (ou tout sous-dossier de `data/`).

```bash
cp mes_contrats/*.pdf data/contrats/
```

### 6. Indexer les documents

Cette étape vectorise tous les PDFs et les stocke dans ChromaDB.  
**À exécuter une seule fois**, ou à chaque ajout de nouveaux documents.

```bash
python src/indexer.py
```

Sortie attendue :

```
2024-01-01 10:00:00 | INFO | 5 PDFs trouvés
2024-01-01 10:00:01 | INFO | 509 pages chargées
2024-01-01 10:00:02 | INFO | 1517 chunks créés
2024-01-01 10:00:15 | INFO | Vectorisation terminée
2024-01-01 10:00:15 | INFO | Indexation terminée : 1517 chunks prêts
```

### 7. Lancer l'application

```bash
streamlit run src/app.py
```

L'application est accessible sur **http://localhost:8501**.

---

## Utilisation

1. Ouvrir l'application dans le navigateur (`http://localhost:8501`)
2. Saisir une question juridique dans le champ de texte
3. Cliquer sur **Rechercher**
4. Lire la réponse générée avec :
   - Le **score de confiance** (pourcentage de pertinence des documents trouvés)
   - La **réponse détaillée** basée uniquement sur vos documents
   - Les **sources** (fichier PDF + numéro de page)

**Exemples de questions :**
- *Quelles sont les conditions de rupture d'un contrat de travail ?*
- *Quelles sont les obligations de l'employeur en matière de sécurité ?*
- *Comment fonctionne le préavis de licenciement ?*

> Si le score de confiance est insuffisant (< 75 % par défaut), le système affiche un avertissement plutôt qu'une réponse incertaine.

---

## Déploiement Docker (VPS)

### 1. Cloner le repo sur le VPS

```bash
git clone https://github.com/Okanfana12/legal-ai-search.git
cd legal-ai-search
```

### 2. Configurer l'environnement

```bash
cp .env.example .env
nano .env   # Renseigner OPENAI_API_KEY et ANTHROPIC_API_KEY
```

### 3. Indexer les documents (première fois uniquement)

```bash
pip install -r requirements.txt
python src/indexer.py
```

### 4. Builder l'image Docker

```bash
docker build -t legal-ai-search .
```

### 5. Lancer le conteneur

```bash
docker run -d \
  --name legal-ai \
  -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/chroma_db:/app/chroma_db \
  --restart unless-stopped \
  legal-ai-search
```

| Option | Description |
|--------|-------------|
| `-p 8501:8501` | Expose le port Streamlit |
| `--env-file .env` | Injecte les clés API |
| `-v $(pwd)/chroma_db:/app/chroma_db` | Monte la base vectorielle existante |
| `--restart unless-stopped` | Redémarre automatiquement après reboot |

### 6. Vérifier le déploiement

```bash
docker logs legal-ai          # Voir les logs
docker ps                     # Vérifier que le conteneur tourne
```

L'application est accessible sur **http://<IP_VPS>:8501**.

### Mettre à jour le code

```bash
git pull origin main
docker build -t legal-ai-search .
docker stop legal-ai && docker rm legal-ai
docker run -d --name legal-ai -p 8501:8501 --env-file .env \
  -v $(pwd)/chroma_db:/app/chroma_db --restart unless-stopped legal-ai-search
```

---

## Variables d'environnement

| Variable | Valeur par défaut | Description |
|----------|-------------------|-------------|
| `PYTHONIOENCODING` | `utf-8` | Encodage Unicode pour caractères accentués |
| `OPENAI_API_KEY` | — | **Obligatoire.** Clé API OpenAI pour les embeddings |
| `ANTHROPIC_API_KEY` | — | **Obligatoire.** Clé API Anthropic pour la génération |
| `CHUNK_SIZE` | `1000` | Taille maximale d'un chunk en caractères |
| `CHUNK_OVERLAP` | `200` | Chevauchement entre chunks consécutifs |
| `TOP_K_RESULTS` | `5` | Nombre de chunks récupérés par recherche |
| `CONFIDENCE_THRESHOLD` | `0.75` | Seuil de confiance minimum (0 à 1) |

---

## Pipeline RAG — détail technique

### Indexation (`indexer.py`)

1. **Chargement** — `PyPDFLoader` lit chaque PDF page par page et enrichit les métadonnées (`source`, `folder`, `page`)
2. **Découpage** — `RecursiveCharacterTextSplitter` découpe le texte en chunks de 1000 caractères avec 200 caractères de chevauchement pour éviter de couper les phrases en plein milieu
3. **Vectorisation** — `OpenAIEmbeddings` transforme chaque chunk en vecteur de 1536 dimensions via l'API OpenAI
4. **Stockage** — `ChromaDB` persiste les vecteurs sur le disque pour éviter de re-vectoriser à chaque démarrage

### Recherche (`retriever.py`)

1. **Embedding de la question** — la question est transformée en vecteur par le même modèle d'embedding
2. **Recherche par similarité** — ChromaDB calcule la similarité cosinus entre le vecteur de la question et tous les chunks stockés
3. **Score de confiance** — moyenne des scores de similarité des Top-K chunks retournés
4. **Vérification du seuil** — si la confiance est insuffisante, le système retourne un avertissement sans appeler Claude
5. **Construction du contexte** — les chunks pertinents sont formatés avec leur source et leur page
6. **Génération** — Claude reçoit le contexte + la question et génère une réponse basée uniquement sur les documents fournis

---

## Vers une version souveraine

Pour un déploiement en environnement sensible (données personnelles, secteur juridique, défense) :

```bash
# Lancer Mistral en local via Ollama
ollama pull mistral

# Utiliser les embeddings HuggingFace locaux
# Aucune donnée ne quitte l'infrastructure
python src/retriever_local.py
```

Cette configuration garantit :
- ✅ Aucune donnée envoyée à des APIs externes
- ✅ Conformité RGPD
- ✅ Déployable on-premise ou sur cloud privé (AWS, Kubernetes)

---

## Limitations connues

- **Qualité des PDFs** — les PDFs scannés sans OCR ne sont pas lisibles par PyPDFLoader
- **Langue** — le système fonctionne mieux avec des documents et questions en français
- **Taille du corpus** — au-delà de 100 000 chunks, les performances de ChromaDB peuvent se dégrader
- **Coût API** — chaque question génère un appel OpenAI (embedding) et un appel Anthropic (génération)
- **Mise à jour** — l'ajout de nouveaux documents nécessite de relancer `indexer.py`

---

## 📄 Licence

MIT
