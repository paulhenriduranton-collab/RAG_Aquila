# RAG Aquila

Système RAG (Retrieval-Augmented Generation) permettant à un LLM local de répondre à des questions sur tes propres documents PDF/DOCX/TXT, sans connexion internet et sans envoyer de données à l'extérieur.

Conçu pour des brochures universitaires (ENS DMA, Sorbonne) et testé sur 50 questions de 3 niveaux de difficulté.

## Architecture en bref

```text
─── INGESTION (une fois) ───────────────────────────────────────────
Documents PDF/TXT/DOCX
    → Extraction Markdown page par page (pymupdf4llm + ftfy)
    → Détection et suppression des pages de table des matières
    → Concaténation des pages par document (marqueurs <!-- PAGE X -->)
    → Pré-découpe par titres Markdown (#, ##), fusion sections < 800 chars
    → Split agentique LLM (gemma4:12b insère ===SPLIT=== entre sous-sections)
    → Fallback récursif (RecursiveCharacterTextSplitter, 1500 chars, sans overlap)
    → Filtre micro-chunks (< 30 chars)
    → Préfixe contextuel déterministe : source + page + titres + mots-clés par fréquence (zéro LLM)
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
run_agentic_all.py       → passe les 50 questions au pipeline agentique
                            + capture les états intermédiaires
evaluate_ragas.py        → score 5 métriques RAGAS (bout-en-bout)
evaluate_components.py   → score 8 briques indépendamment
                            (Router, Retrieval, Re-ranking, Grading,
                             Rewriting, Generation, Déduplication, Fusion RRF)
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
│   ├── agent.py              ← pipeline agentique LangGraph (instrumenté pour l'éval)
│   ├── api_server.py         ← API OpenAI-compatible pour Open WebUI
│   ├── run_agentic_all.py    ← passe le dataset au pipeline + capture états intermédiaires
│   ├── evaluate_ragas.py     ← évaluation RAGAS globale (5 métriques, bout-en-bout)
│   ├── evaluate_components.py ← évaluation par composant (8 briques)
│   ├── eval_common.py        ← utilitaires partagés (ground truth sources, chunks → RAGAS)
│   └── debug_question.py     ← script de debug pour analyser le retrieval d'une question
├── data/
│   ├── questions.json            ← 50 questions avec réponses de référence
│   ├── agentic_results.json      ← résultats du pipeline agentique (généré)
│   ├── agentic_results_debug.json ← résultats + états intermédiaires (pour éval composant)
│   ├── ragas_evaluation.csv      ← scores RAGAS par question (généré)
│   └── component_evaluation.csv  ← scores par brique et par question (généré)
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

# LLM principal pour split agentique (ingestion), HyDE, grading, reformulation et génération
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
# Étape 1 : passe toutes les questions au pipeline agentique (capture les états intermédiaires)
python src/run_agentic_all.py

# Étape 2a : évaluation globale (5 métriques RAGAS bout-en-bout)
python src/evaluate_ragas.py

# Étape 2b : évaluation par composant (8 briques isolées)
python src/evaluate_components.py
```

### 5. Sur Google Colab (GPU recommandé)

Ouvre le notebook directement depuis GitHub :

**https://colab.research.google.com/github/paulhenriduranton-collab/RAG_Aquila/blob/main/colab_run.ipynb**

Avant de lancer : **Exécution → Modifier le type d'exécution → GPU (L4 ou A100)**.

Deux modes disponibles sur Colab :
- **Batch** (étape 4) : passe les 50 questions au pipeline agentique
- **Interface Gradio** (étape 4b) : interface web interactive avec surlignage des chunks dans les PDF sources

## Documentation complète

Voir [documentation/00_index.md](documentation/00_index.md).
