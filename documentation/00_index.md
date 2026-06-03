# Documentation — Projet Aquila

## Table des matières

| Fichier | Contenu |
|---|---|
| [01_presentation.md](01_presentation.md) | C'est quoi ce projet ? Différence avec ChatGPT, cas d'usage |
| [02_les_outils.md](02_les_outils.md) | bge-m3, BM25, RRF, pymupdf4llm, gemma2:2b, ChromaDB, LangChain |
| [03_fonctionnement_detaille.md](03_fonctionnement_detaille.md) | Le flux complet, ingestion et question/réponse étape par étape |
| [04_explication_du_code.md](04_explication_du_code.md) | Chaque fonction de `ingest.py`, `ask.py` et `app.py` expliquée ligne par ligne |
| [05_utilisation.md](05_utilisation.md) | Comment installer et lancer le projet |
| [06_problemes_courants.md](06_problemes_courants.md) | Erreurs fréquentes et leurs solutions |

## Par où commencer ?

- Tu découvres le projet → lis **01** puis **02**
- Tu veux comprendre comment ça marche en profondeur → lis **03**
- Tu veux comprendre le code → lis **04**
- Tu veux juste lancer le projet → lis **05**
- Tu as une erreur → lis **06**

## Résumé des choix techniques

| Composant | Choix | Raison |
|---|---|---|
| Extraction PDF | `pymupdf4llm` | Markdown structuré, espaces corrects autour des symboles |
| Modèle d'embedding | `bge-m3` | Multilingue (FR), fenêtre 8192 tokens, vocabulaire scientifique |
| Recherche | Hybride sémantique + BM25 | Sémantique pour les concepts, BM25 pour les termes exacts |
| Fusion | RRF (Reciprocal Rank Fusion) | Équitable entre les deux méthodes, basé sur les rangs |
| Modèle de génération | `gemma2:2b` | Rapide, suffisant pour extraire du contexte fourni |
| Chunks | 1000 caractères, overlap 200 | Assez grand pour contenir une définition mathématique complète |
| Interface | Streamlit (`app.py`) | Interface web légère, lancée avec `python -m streamlit run src/app.py` |
