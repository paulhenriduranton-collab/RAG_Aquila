# Documentation — Projet RAG Aquila

## Table des matières

| Fichier | Contenu |
|---|---|
| [01_presentation.md](01_presentation.md) | C'est quoi ce projet ? Différence avec ChatGPT, cas d'usage |
| [02_les_outils.md](02_les_outils.md) | bge-m3, BM25, RRF, re-ranker, gemma2:2b, ChromaDB, Streamlit |
| [03_fonctionnement_detaille.md](03_fonctionnement_detaille.md) | Le flux complet — ingestion, retrieval hybride, re-ranking, évaluation |
| [05_utilisation.md](05_utilisation.md) | Comment installer et lancer le projet |
| [06_problemes_courants.md](06_problemes_courants.md) | Erreurs fréquentes et leurs solutions |
| [07_open_webui.md](07_open_webui.md) | Lancer le RAG agentique dans une interface de chat Open WebUI |

## Par où commencer ?

- Tu découvres le projet → lis **01** puis **02**
- Tu veux comprendre comment ça marche → lis **03**
- Tu veux juste lancer le projet → lis **05**
- Tu as une erreur → lis **06**

## Résumé des choix techniques

| Composant | Choix | Raison |
|---|---|---|
| Extraction PDF | `pymupdf4llm` | Markdown structuré, préserve titres et symboles mathématiques |
| Modèle d'embedding | `bge-m3` (Ollama) | Multilingue, fenêtre 8192 tokens, vocabulaire scientifique |
| Recherche sémantique | ChromaDB + similarité cosinus | Trouve les passages conceptuellement proches |
| Recherche lexicale | BM25 | Trouve les passages avec les mêmes mots exacts |
| Fusion | RRF (Reciprocal Rank Fusion) | Combine les deux classements indépendamment des scores bruts |
| Re-ranking | CrossEncoder `mmarco-mMiniLMv2` | Reclasse les 10 candidats RRF par pertinence réelle |
| Modèle de génération | `gemma2:2b` (Ollama) | Rapide, tourne entièrement en local |
| Chunks | Pipeline 3 étapes : titres → fusion micro-chunks → taille | Préserve la hiérarchie Markdown, évite les fragments courts, protège les tableaux |
| Interface utilisateur | Streamlit | Interface web simple, lancée en local |
| Évaluation | 5 métriques LLM-judge custom (pas de dépendance RAGAS) | Faithfulness, Answer Relevancy, Context Quality, Context Recall, Answer Correctness |

## Structure du projet

```
RAG_Aquila/
├── src/
│   ├── ingest.py       # Indexation des documents (à lancer une fois)
│   ├── ask.py          # Pipeline RAG complet (retrieval + génération)
│   ├── app.py          # Interface web Streamlit
│   └── evaluate.py     # Pipeline d'évaluation avec 5 métriques
├── data/
│   ├── questions.json  # Dataset de 40 questions avec réponses de référence
│   └── results.json    # Résultats de la dernière évaluation
├── documents/          # Tes fichiers PDF/DOCX/TXT à indexer
├── vector_db/          # Base vectorielle ChromaDB (générée automatiquement)
├── prompts/
│   └── rag_prompt.txt  # Template du prompt envoyé au LLM
└── requirements.txt    # Dépendances Python
```
