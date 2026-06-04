# Documentation — Projet Aquila

## Table des matières

| Fichier | Contenu |
|---|---|
| [01_presentation.md](01_presentation.md) | C'est quoi ce projet ? Différence avec ChatGPT, cas d'usage |
| [02_les_outils.md](02_les_outils.md) | bge-m3, BM25, pymupdf4llm, gemma2:2b, ChromaDB, FastAPI, RAGAS |
| [03_fonctionnement_detaille.md](03_fonctionnement_detaille.md) | Le flux complet — ingestion, question/réponse, API, évaluation |
| [05_utilisation.md](05_utilisation.md) | Comment installer et lancer le projet |
| [06_problemes_courants.md](06_problemes_courants.md) | Erreurs fréquentes et leurs solutions |

## Par où commencer ?

- Tu découvres le projet → lis **01** puis **02**
- Tu veux comprendre comment ça marche → lis **03**
- Tu veux juste lancer le projet → lis **05**
- Tu as une erreur → lis **06**

## Résumé des choix techniques

| Composant | Choix | Raison |
|---|---|---|
| Extraction PDF | `pymupdf4llm` | Markdown structuré, espaces corrects autour des symboles mathématiques |
| Modèle d'embedding | `bge-m3` (Ollama) | Multilingue, fenêtre 8192 tokens, vocabulaire scientifique |
| Recherche | Hybride sémantique + BM25 | Sémantique pour les concepts, BM25 pour les termes exacts |
| Fusion | Score pondéré (sémantique + BM25 normalisé) | Simple, réglable via `BM25_WEIGHT` |
| Modèle de génération | `gemma2:2b` (Ollama) | Rapide, tourne en local |
| Chunks | 1000 caractères, overlap 200 | Assez grand pour contenir une définition mathématique complète |
| Interface utilisateur | Open WebUI (Docker) + API FastAPI | Interface web type ChatGPT branchée sur le pipeline RAG local |
| Évaluation retrieval | `evaluation/eval_retrieval.py` | Recall@K et MRR mesurés sur dataset synthétique |
| Évaluation génération | RAGAS | Faithfulness, Answer Relevancy, Context Precision |

## Structure du projet

```
Projet Aquila/
├── src/
│   ├── ingest.py          # Indexation des documents (à lancer une fois)
│   ├── ask.py             # Pipeline RAG — terminal
│   └── api.py             # Serveur FastAPI compatible OpenAI (pour Open WebUI)
├── evaluation/
│   ├── generate_dataset.py  # Génération du dataset synthétique
│   └── eval_retrieval.py    # Calcul des métriques Recall@K et MRR
├── documents/             # Tes fichiers PDF/DOCX/TXT
├── vector_db/             # Base vectorielle ChromaDB (générée automatiquement)
├── prompts/
│   └── rag_prompt.txt     # Instructions envoyées au LLM
├── docker-compose.yml     # Lance Open WebUI
└── requirements.txt       # Dépendances Python
```
