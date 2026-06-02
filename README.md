# RAG simple — Stage découverte

Ce projet est une version volontairement simplifiée d'un RAG.

L'objectif est de permettre à un LLM local (Gemma via Ollama) de répondre à des questions en s'appuyant sur des fichiers placés dans le dossier `documents/`.

## Fonctionnement général

```text
Documents → Découpage du texte → Embeddings (nomic-embed-text) → Recherche des passages utiles → Réponse (gemma:latest)
```

## Arborescence

```text
Projet Aquila/
│
├── README.md
├── requirements.txt
│
├── documents/       ← mettre ici les fichiers à indexer (.pdf, .txt, .docx)
├── vector_db/       ← index ChromaDB généré automatiquement
├── prompts/
│   └── rag_prompt.txt
└── src/
    ├── ingest.py    ← indexe les documents
    └── ask.py       ← pose une question en ligne de commande
```

## Prérequis

- [Ollama](https://ollama.com/) installé et en cours d'exécution
- Les deux modèles suivants téléchargés :

```bash
ollama pull gemma:latest
ollama pull nomic-embed-text
```

## Installation

Créer et activer un environnement virtuel :

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Ajouter les documents

Mettre les fichiers dans le dossier `documents/` (`.pdf`, `.txt` ou `.docx`).

### 2. Indexer les documents

```bash
python src/ingest.py
```

Cela crée l'index vectoriel dans `vector_db/`. À relancer si vous ajoutez de nouveaux documents.

### 3. Poser une question

```bash
python src/ask.py
```

Le terminal affiche `Question :`, tapez votre question et appuyez sur Entrée. Ctrl+C pour quitter.

## Limites volontairement acceptées

Ce projet est pensé pour un stage découverte.

Il ne couvre pas :
- la gestion fine des droits utilisateurs ;
- l'évaluation avancée des réponses ;
- le monitoring ;
- le déploiement cloud ;
- la gestion complexe de centaines de milliers de documents.

L'objectif est de comprendre la logique du RAG, pas de construire une solution industrielle.
