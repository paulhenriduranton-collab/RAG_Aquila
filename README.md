# RAG Aquila

Système RAG (Retrieval-Augmented Generation) permettant à un LLM local de répondre à des questions sur tes propres documents PDF/DOCX/TXT, sans connexion internet et sans envoyer de données à l'extérieur.

Conçu pour des brochures universitaires (ENS DMA, Sorbonne) et testé sur 40 questions de 3 niveaux de difficulté.

## Architecture en bref

```text
─── INGESTION (une fois) ───────────────────────────────────────────
Documents PDF/TXT/DOCX
    → Extraction Markdown page par page (pymupdf4llm + ftfy)
    → Détection et suppression des pages de table des matières
    → Chevauchement inter-pages (300 chars) pour ne pas couper les listes
    → Découpage par titres Markdown (MarkdownHeaderTextSplitter)
    → Fusion des micro-chunks (< 500 chars)
    → Re-découpage par taille (RecursiveCharacterTextSplitter, 1000 chars)
    → Overlap systématique entre chunks adjacents (200 chars)
    → Préfixe contextuel par chunk : source + page + titres + mots-clés LLM (gemma2:2b)
    → Vectorisation (bge-m3 via Ollama) → ChromaDB

─── QUESTION/RÉPONSE (agentique) ───────────────────────────────────
Question utilisateur
    → Identification de la source et de la difficulté (LangGraph)
    → Décomposition en sous-requêtes si difficulté 3
    → HyDE : génération d'une réponse fictive pour améliorer la recherche
    → Recherche sémantique MMR (bge-m3, top 25 diversifiés)
    → Recherche lexicale BM25 normalisée (top 25)
    → Fusion RRF (Reciprocal Rank Fusion)
    → Déduplication Jaccard (seuil 80 %)
    → Re-ranking CrossEncoder (BAAI/bge-reranker-v2-m3, top 5, seuil 0.5)
    → Évaluation de la suffisance des chunks (LLM)
    → Si insuffisant : upgrade difficulté + reformulation + nouveau retrieval
    → Génération de la réponse (gemma4:12b, temperature=0)

─── ÉVALUATION (à la demande) ──────────────────────────────────────
run_agentic_all.py  → passe les 40 questions au pipeline agentique
evaluate_ragas.py   → score 5 métriques RAGAS via Ollama (Faithfulness,
                       AnswerRelevancy, ContextPrecision, ContextRecall,
                       AnswerCorrectness)
```

## Arborescence

```text
RAG_Aquila/
├── README.md
├── requirements.txt
├── colab_run.ipynb           ← notebook pour lancer le tout sur Google Colab
├── RELANCER_OPEN_WEBUI.md    ← aide-mémoire pour relancer Open WebUI en local
├── documents/                ← fichiers à indexer (.pdf, .txt, .docx)
│   ├── ENS.pdf
│   └── SORBONNE.pdf
├── prompts/
│   └── rag_prompt.txt        ← template du prompt envoyé au LLM
├── src/
│   ├── ingest.py             ← indexe les documents → ChromaDB
│   ├── ask.py                ← pipeline RAG hybride (retrieval + génération)
│   ├── agent.py              ← pipeline agentique LangGraph
│   ├── api_server.py         ← API OpenAI-compatible pour Open WebUI
│   ├── run_agentic_all.py    ← passe le dataset complet au pipeline agentique
│   ├── evaluate_ragas.py     ← évalue les résultats avec 5 métriques RAGAS
│   └── debug_question.py     ← script de debug pour analyser le retrieval d'une question
├── data/
│   ├── questions.json        ← 40 questions avec réponses de référence
│   ├── agentic_results.json  ← résultats du pipeline agentique (généré)
│   └── ragas_evaluation.csv  ← scores RAGAS par question (généré)
├── vector_db/                ← base vectorielle ChromaDB (versionnée pour Colab)
└── documentation/            ← documentation complète du projet
```

## Prérequis

- Python 3.10 à 3.13
- [Ollama](https://ollama.com/) installé et en cours d'exécution
- Modèles téléchargés :

```powershell
# Modèle d'embedding multilingue (indexation + recherche)
ollama pull bge-m3

# LLM léger pour la contextualisation des chunks à l'ingestion
ollama pull gemma2:2b

# LLM principal pour HyDE, grading, reformulation et génération finale
ollama pull gemma4:12b
```

## Installation

```powershell
# Créer et activer l'environnement virtuel
python -m venv venv
venv\Scripts\activate

# Installer les dépendances (re-ranker ~471 Mo téléchargé au 1er lancement)
pip install -r requirements.txt
```

## Utilisation

### 1. Ajouter les documents

Copier les fichiers dans `documents/` (`.pdf`, `.txt` ou `.docx`).

### 2. Indexer

```powershell
python src/ingest.py
```

Crée la base vectorielle dans `C:/vector_db_aquila` (local) ou `vector_db/` (Colab). À relancer à chaque ajout de document.

### 3. Poser une question

**Interface chat Open WebUI (recommandé) :**

```powershell
# Fenêtre 1 — serveur RAG agentique
cd src
uvicorn api_server:app --host 0.0.0.0 --port 8001

# Fenêtre 2 — interface Open WebUI
open-webui serve --port 3000
```

Puis connecter Open WebUI : Réglages → Connexions → URL `http://localhost:8001/v1`.

**Ligne de commande (avec logs de retrieval) :**

```powershell
python src/ask.py          # pipeline RAG classique
python src/agent.py        # pipeline agentique (plus lent, plus précis)
```

### 4. Évaluer le pipeline

```powershell
# Étape 1 : passe toutes les questions au pipeline agentique
python src/run_agentic_all.py

# Étape 2 : calcule les métriques RAGAS sur les résultats
python src/evaluate_ragas.py
```

### 5. Sur Google Colab (GPU recommandé)

Ouvre le notebook directement depuis GitHub :

**https://colab.research.google.com/github/paulhenriduranton-collab/RAG_Aquila/blob/main/colab_run.ipynb**

Avant de lancer : **Exécution → Modifier le type d'exécution → GPU (L4 ou A100)**.

Deux modes disponibles sur Colab :
- **Batch** (étape 4) : passe les 40 questions au pipeline agentique
- **Interface Gradio** (étape 4b) : interface web interactive avec surlignage des chunks dans les PDF sources

## Documentation complète

Voir [documentation/00_index.md](documentation/00_index.md).
