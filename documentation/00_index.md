# Documentation — Projet RAG Aquila

## Table des matières

| Fichier | Contenu |
|---|---|
| [01_presentation.md](01_presentation.md) | C'est quoi ce projet ? Différence avec ChatGPT, cas d'usage, pipeline agentique |
| [02_les_outils.md](02_les_outils.md) | bge-m3, BM25, RRF, re-ranker, gemma2:2b, gemma4:12b, ChromaDB, LangGraph, RAGAS |
| [03_fonctionnement_detaille.md](03_fonctionnement_detaille.md) | Le flux complet — ingestion, HyDE, MMR, retrieval hybride, re-ranking, agentique, évaluation |
| [04_colab.md](04_colab.md) | Lancer le projet sur Google Colab (GPU), interface Gradio, push automatique |
| [05_utilisation.md](05_utilisation.md) | Comment installer et lancer le projet en local |
| [06_problemes_courants.md](06_problemes_courants.md) | Erreurs fréquentes et leurs solutions |
| [07_open_webui.md](07_open_webui.md) | Lancer le RAG agentique dans une interface de chat Open WebUI |

## Par où commencer ?

- Tu découvres le projet → lis **01** puis **02**
- Tu veux comprendre comment ça marche → lis **03**
- Tu veux lancer sur Google Colab → lis **04**
- Tu veux juste lancer le projet en local → lis **05**
- Tu as une erreur → lis **06**

## Résumé des choix techniques

| Composant | Choix | Raison |
|---|---|---|
| Extraction PDF | `pymupdf4llm` | Markdown structuré, préserve titres et symboles mathématiques |
| Réparation d'encodage | `ftfy` | Corrige les accents cassés avant indexation BM25 |
| Chevauchement inter-pages | 300 chars de la page suivante recopiés | Évite de couper une liste/section à cheval sur 2 pages |
| Overlap intra-page | 200 chars entre chunks adjacents | Évite de perdre une info à la frontière entre deux chunks |
| Détection TdM | Ratio lignes avec `...` > 30 % | Élimine les pages de table des matières inutiles |
| Contextual retrieval | Préfixe (source + page + titres + mots-clés gemma2:2b) | L'embedding voit le contexte structurel même sur un chunk isolé |
| Modèle d'embedding | `bge-m3` (Ollama) | Multilingue, fenêtre 8192 tokens, vocabulaire scientifique |
| Recherche sémantique | ChromaDB + MMR | Chunks pertinents ET diversifiés (évite les doublons) |
| HyDE | Réponse fictive pour la recherche sémantique | Réduit l'écart embedding question/chunk |
| Recherche lexicale | BM25 normalisé (accents, écriture inclusive) | Trouve les passages avec les mots exacts |
| Fusion | RRF (Reciprocal Rank Fusion) | Combine les deux classements indépendamment des scores bruts |
| Déduplication | Jaccard sur tokens normalisés (seuil 80 %) | Supprime les quasi-doublons avant re-ranking |
| Re-ranking | CrossEncoder `BAAI/bge-reranker-v2-m3` (seuil 0.5) | Reclasse les candidats RRF par pertinence réelle |
| Modèle de contextualisation | `gemma2:2b` (Ollama) | Génère 3-6 mots-clés par chunk — léger, 4x plus rapide que gemma4:12b |
| Modèle de génération | `gemma4:12b` (Ollama) | HyDE, grading, reformulation et génération finale — meilleur raisonnement |
| Chunks | Pipeline 6 étapes : TdM → titres → fusion → taille → overlap → contextualisation | Préserve la hiérarchie, évite les micro-chunks, ajoute le contexte structurel |
| Pipeline agentique | LangGraph (identification source + difficulté → retrieval → grade → reformulation) | Cherche plus intelligemment en cas de réponse insuffisante |
| Interface utilisateur | Open WebUI + api_server.py (compatible OpenAI) | Interface chat locale sans Streamlit |
| Interface Colab | Gradio (colab_run.ipynb, étape 4b) | Interface web avec surlignage PDF des chunks sources |
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
│   ├── evaluate_ragas.py     # Évaluation RAGAS avec 5 métriques
│   └── debug_question.py     # Script de debug : retrieval verbose sur une question
├── data/
│   ├── questions.json        # Dataset de 40 questions avec réponses de référence
│   ├── agentic_results.json  # Résultats du run agentique (généré par run_agentic_all.py)
│   └── ragas_evaluation.csv  # Scores RAGAS par question (généré par evaluate_ragas.py)
├── documents/                # Fichiers PDF/DOCX/TXT à indexer
│   ├── ENS.pdf               # Brochure ENS DMA 2024-2025
│   └── SORBONNE.pdf          # Brochure Master Sorbonne 2025-2026
├── vector_db/                # Base vectorielle ChromaDB (versionnée dans le repo pour Colab)
├── prompts/
│   └── rag_prompt.txt        # Template du prompt envoyé au LLM
├── colab_run.ipynb           # Notebook pour lancer sur Google Colab (GPU)
├── requirements.txt          # Dépendances Python
└── documentation/            # Documentation complète (ce dossier)
```

### Double chemin pour la base vectorielle

| Contexte | Chemin | Raison |
|---|---|---|
| **Local (Windows)** | `C:/vector_db_aquila` (dans `ingest.py`) | Hors OneDrive — SQLite corrompu par la synchro cloud |
| **Colab / ask.py** | `vector_db/` (dans le repo) | Versionnée dans git pour que Colab la clone directement |
| **Colab (override notebook)** | `/content/RAG_Aquila/vector_db` | Le notebook écrase `VECTOR_DB_DIR` au runtime |

En local, `ask.py` lit toujours `vector_db/` (chemin relatif au repo). Pour l'ingestion locale, `ingest.py` écrit dans `C:/vector_db_aquila` pour éviter la corruption OneDrive — il faut ensuite copier le résultat dans `vector_db/` si on veut le versionner.
