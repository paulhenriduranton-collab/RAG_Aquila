# Documentation — Projet RAG Aquila

## Table des matières

| Fichier | Contenu |
|---|---|
| [01_presentation.md](01_presentation.md) | C'est quoi ce projet ? Différence avec ChatGPT, cas d'usage, pipeline agentique |
| [02_les_outils.md](02_les_outils.md) | bge-m3, BM25, RRF, re-ranker, gemma4:12b, ChromaDB, LangGraph, RAGAS |
| [03_fonctionnement_detaille.md](03_fonctionnement_detaille.md) | Le flux complet — ingestion, HyDE, MMR, retrieval hybride, re-ranking, agentique, évaluation |
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
| Réparation d'encodage | `ftfy` | Corrige les accents cassés avant indexation BM25 |
| Chevauchement inter-pages | 300 chars de la page suivante recopiés | Évite de couper une liste/section à cheval sur 2 pages |
| Détection TdM | Ratio lignes avec `...` > 30 % | Élimine les pages de table des matières inutiles |
| Contextual retrieval | Préfixe LLM (source + page + titres + mots-clés) | L'embedding voit le contexte structurel même sur un chunk isolé |
| Modèle d'embedding | `bge-m3` (Ollama) | Multilingue, fenêtre 8192 tokens, vocabulaire scientifique |
| Recherche sémantique | ChromaDB + MMR | Chunks pertinents ET diversifiés (évite les doublons) |
| HyDE | Réponse fictive pour la recherche sémantique | Réduit l'écart embedding question/chunk |
| Recherche lexicale | BM25 normalisé (accents, écriture inclusive) | Trouve les passages avec les mots exacts |
| Fusion | RRF (Reciprocal Rank Fusion) | Combine les deux classements indépendamment des scores bruts |
| Re-ranking | CrossEncoder `BAAI/bge-reranker-v2-m3` | Reclasse les 10 candidats RRF par pertinence réelle |
| Modèle de génération | `gemma4:12b` (Ollama) | Meilleure qualité de raisonnement qu'un 2b, tourne en local |
| Chunks | Pipeline 5 étapes : TdM → titres → fusion → taille → contextualisation | Préserve la hiérarchie, évite les micro-chunks, ajoute le contexte structurel |
| Pipeline agentique | LangGraph (identification source → retrieval → grade → reformulation) | Cherche plus intelligemment en cas de réponse insuffisante |
| Interface utilisateur | Open WebUI + api_server.py (compatible OpenAI) | Interface chat locale sans Streamlit |
| Évaluation | RAGAS (5 métriques LLM-judge) via Ollama | Standard du secteur, aucun appel API externe |

## Structure du projet

```
RAG_Aquila/
├── src/
│   ├── ingest.py             # Indexation des documents (à lancer une fois)
│   ├── ask.py                # Pipeline RAG complet (retrieval hybride + génération)
│   ├── agent.py              # Pipeline agentique LangGraph
│   ├── api_server.py         # Serveur API OpenAI-compatible pour Open WebUI
│   ├── run_agentic_all.py    # Lance le pipeline agentique sur tout le dataset
│   └── evaluate_ragas.py     # Évaluation RAGAS avec 5 métriques
├── data/
│   ├── questions.json        # Dataset de 40 questions avec réponses de référence
│   ├── agentic_results.json  # Résultats du run agentique (généré par run_agentic_all.py)
│   └── ragas_evaluation.csv  # Scores RAGAS par question (généré par evaluate_ragas.py)
├── documents/                # Tes fichiers PDF/DOCX/TXT à indexer
├── vector_db/ → C:/vector_db_aquila  # Base vectorielle ChromaDB (hors OneDrive)
├── prompts/
│   └── rag_prompt.txt        # Template du prompt envoyé au LLM
├── colab_run.ipynb           # Notebook pour lancer sur Google Colab
└── requirements.txt          # Dépendances Python
```
