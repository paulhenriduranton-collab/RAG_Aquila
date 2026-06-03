# RAG Aquila — Stage découverte

Système RAG (Retrieval-Augmented Generation) permettant à un LLM local de répondre à des questions en s'appuyant uniquement sur des fichiers placés dans `documents/`. Conçu pour des polycopiés de mathématiques.

## Fonctionnement général

```text
Documents PDF/TXT/DOCX
    → Extraction Markdown (pymupdf4llm)
    → Découpage en chunks de 1000 caractères
    → Embeddings multilingues (bge-m3 via Ollama)
    → Base vectorielle ChromaDB

Question utilisateur
    → Recherche sémantique dense (bge-m3, top 20)
    → Recherche lexicale BM25 (top 20)
    → Fusion RRF (Reciprocal Rank Fusion)
    → Top 5 chunks envoyés à gemma2:2b
    → Réponse affichée
```

## Arborescence

```text
RAG_Aquila/
├── README.md
├── requirements.txt
├── documents/        ← fichiers à indexer (.pdf, .txt, .docx)
├── vector_db/        ← index ChromaDB (généré automatiquement)
├── prompts/
│   └── rag_prompt.txt
├── src/
│   ├── ingest.py     ← indexe les documents
│   ├── ask.py        ← pose une question en ligne de commande
│   └── app.py        ← interface web Streamlit
└── documentation/    ← documentation détaillée du projet
```

## Prérequis

- Python 3.10 à 3.13 (pas 3.14 — incompatibilité Pillow)
- [Ollama](https://ollama.com/) installé et en cours d'exécution
- Modèles téléchargés :

```powershell
ollama pull bge-m3
ollama pull gemma2:2b
```

## Installation

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

### 1. Ajouter les documents

Copier les fichiers dans `documents/` (`.pdf`, `.txt` ou `.docx`).

### 2. Indexer

```powershell
python src/ingest.py
```

Crée la base vectorielle dans `vector_db/`. À relancer à chaque ajout de document.

### 3. Poser une question

**Interface web (recommandé) :**

```powershell
python -m streamlit run src/app.py
```

**Ligne de commande :**

```powershell
python src/ask.py
```

## Documentation complète

Voir [documentation/00_index.md](documentation/00_index.md).
