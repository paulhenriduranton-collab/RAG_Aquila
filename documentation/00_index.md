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
| Extraction PDF | `pymupdf4llm` (page_chunks=True) | Markdown structuré, préserve titres et symboles mathématiques, un Document par page |
| Réparation d'encodage | `ftfy` | Corrige les accents cassés avant indexation BM25 |
| Concaténation des pages | Pages réassemblées en un texte continu par document (marqueurs `<!-- PAGE X -->`) | Permet à une section thématique de chevaucher plusieurs pages au lieu d'être bornée par la pagination du PDF |
| Détection TdM | Ratio lignes avec `...` > 30 % | Élimine les pages de table des matières inutiles |
| Pré-découpe par titres | Split déterministe sur `#`/`##`, fusion des sections < 800 caractères | Crée des blocs thématiques qui suivent la structure logique, pas la page |
| Split agentique | LLM `gemma4:12b` insère des marqueurs `===SPLIT===` entre sous-sections (validation : ≥60 % des mots du texte d'origine préservés, sinon fallback) | Découpe par cohérence sémantique plutôt que par une taille fixe |
| Fallback taille | `RecursiveCharacterTextSplitter` (max 1500 car., sans overlap, séparateurs qui protègent les tableaux) | Filet de sécurité pour les rares blocs encore trop gros après le split agentique |
| Filtre micro-chunks | Chunk < 30 caractères écarté | Élimine le bruit (numéros de page isolés, symboles seuls) |
| Contextual retrieval | Préfixe déterministe : source + page + chemin de titres + mots-clés par fréquence (zéro LLM) | L'embedding et BM25 voient le contexte structurel d'un chunk isolé, sans risque d'hallucination |
| Modèle d'embedding | `bge-m3` (Ollama) | Multilingue, fenêtre 8192 tokens, vocabulaire scientifique |
| Recherche sémantique | ChromaDB + MMR | Chunks pertinents ET diversifiés (évite les doublons) |
| HyDE | Réponse fictive pour la recherche sémantique | Réduit l'écart embedding question/chunk |
| Recherche lexicale | BM25 normalisé (accents, écriture inclusive) | Trouve les passages avec les mots exacts |
| Fusion | RRF (Reciprocal Rank Fusion) | Combine les deux classements indépendamment des scores bruts |
| Déduplication | Jaccard sur tokens normalisés (seuil 80 %) | Supprime les quasi-doublons avant re-ranking |
| Re-ranking | CrossEncoder `BAAI/bge-reranker-v2-m3` (seuil 0.5) | Reclasse les candidats RRF par pertinence réelle |
| Modèle de génération/découpe | `gemma4:12b` (Ollama) | Split agentique (ingestion), HyDE, grading, reformulation et génération finale |
| Pipeline agentique | LangGraph (identification source + difficulté → retrieval → grade → reformulation) | Cherche plus intelligemment en cas de réponse insuffisante |
| Interface utilisateur | Open WebUI + api_server.py (compatible OpenAI) | Interface chat locale sans Streamlit |
| Interface Colab | Gradio (colab_run.ipynb, étape 4b) | Interface web avec surlignage PDF des chunks sources |
| Évaluation globale | RAGAS (5 métriques LLM-judge) via Ollama, juge `gemma4:12b` (ou `gemma2:2b` en option sur Colab pour aller plus vite) | Standard du secteur, aucun appel API externe |
| Évaluation par composant | `agent.py` instrumenté (capture des états intermédiaires) + RAGAS (`gemma2:2b`) + juge externe custom (`gemma4:12b`) pour le grading | Isole les 6 briques du pipeline agentique, sans jamais relancer retrieval/génération pour re-scorer |

## Structure du projet

```
RAG_Aquila/
├── schema_rag_aquila.html            # Schéma interactif Mermaid de l'architecture complète
├── schema_evaluate_components.html   # Schéma interactif Mermaid de l'évaluation par composant
├── src/
│   ├── ingest.py             # Indexation des documents (à lancer une fois)
│   ├── ask.py                # Pipeline RAG complet (retrieval hybride + génération)
│   ├── agent.py              # Pipeline agentique LangGraph (instrumenté pour l'éval par composant)
│   ├── api_server.py         # Serveur API OpenAI-compatible pour Open WebUI
│   ├── run_agentic_all.py    # Lance le pipeline agentique sur tout le dataset + capture les états intermédiaires
│   ├── evaluate_ragas.py     # Évaluation RAGAS globale (5 métriques, bout-en-bout)
│   ├── eval_common.py        # Utilitaires partagés : ground truth des sources, chunks → contextes RAGAS
│   ├── evaluate_components.py # Évaluation par composant (6 briques), à partir d'agentic_results.json
│   └── debug_question.py     # Script de debug : retrieval verbose sur une question
├── data/
│   ├── questions.json            # Dataset de 50 questions avec réponses de référence
│   ├── agentic_results.json      # Résultats + états intermédiaires du run agentique (run_agentic_all.py)
│   ├── ragas_evaluation.csv      # Scores RAGAS globaux par question (evaluate_ragas.py)
│   └── component_evaluation.csv  # Scores par brique et par question (evaluate_components.py)
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
