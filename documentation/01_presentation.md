# 01 — Présentation du projet

## C'est quoi ce projet ?

**RAG Aquila** est un système de type **RAG** (*Retrieval-Augmented Generation*).

En français simple : un programme qui permet de **poser des questions à une IA sur tes propres documents**, et d'obtenir des réponses basées uniquement sur le contenu de ces documents — sans connexion internet, sans envoyer tes données sur un serveur externe.

---

## La différence avec ChatGPT

| ChatGPT | Projet RAG Aquila |
|---|---|
| Connaît des milliards de pages internet | Ne connaît que TES documents |
| Tourne sur les serveurs d'OpenAI | Tourne entièrement sur ta machine |
| Tes données partent sur internet | Tes données restent privées |
| Peut inventer des réponses sur n'importe quel sujet | Ne répond que si l'info est dans tes fichiers |
| Interface fournie par OpenAI | Interface Streamlit hébergée en local |

---

## Le cas d'usage concret

Tu as des fichiers PDF de cours universitaires :
- `Brochure-2024-2025.pdf` (ENS DMA)
- `Brochure Master2526_1.pdf` (Sorbonne)

Tu poses la question dans l'interface Streamlit : *"Quels sont les cours obligatoires de L3 à l'ENS DMA ?"*

Le système :
1. Cherche dans tes PDFs les passages les plus pertinents (recherche hybride : sémantique + mots-clés)
2. Fusionne les deux listes de résultats avec la méthode RRF (Reciprocal Rank Fusion)
3. Re-classe les 10 meilleurs passages avec un cross-encoder (re-ranker) plus précis
4. Envoie les 5 meilleurs passages au LLM gemma2:2b
5. Le LLM rédige une réponse basée uniquement sur ces passages

---

## Pourquoi "RAG" ?

- **R**etrieval = *Récupération* — on cherche les bons passages dans les documents
- **A**ugmented = *Augmenté* — on enrichit la question avec ces passages avant de l'envoyer au LLM
- **G**eneration = *Génération* — le LLM génère une réponse à partir du contexte fourni

Sans le "R", le LLM répondrait de mémoire (et inventerait). Avec le "R", il est contraint de répondre uniquement à partir des passages fournis.

---

## Les quatre modes d'utilisation

| Mode | Commande | Usage |
|---|---|---|
| Interface web | `python -m streamlit run src/app.py --server.headless true` | Usage normal |
| Terminal | `python src/ask.py` | Debug — affiche tous les logs de retrieval |
| Indexation | `python src/ingest.py` | À relancer si tu changes tes documents |
| Évaluation | `python src/evaluate.py` | Mesure la qualité du pipeline sur 40 questions |

---

## Ce que le projet N'est PAS

- Ce n'est pas ChatGPT — il ne répond pas à des questions générales hors documents
- Ce n'est pas un moteur de recherche — il génère une réponse rédigée, pas une liste de liens
- Ce n'est pas un RAG agentique — le pipeline est fixe, il ne décide pas dynamiquement de chercher différemment
- Ce n'est pas infaillible — si l'information n'est pas dans les documents, il ne peut pas répondre correctement

---

## Ce qui rend ce RAG avancé

La plupart des RAG basiques font : question → recherche sémantique → LLM. Ce projet va plus loin :

```
Question
    │
    ├── Recherche sémantique (20 candidats)
    ├── Recherche BM25/lexicale (20 candidats)
    │
    ▼
Fusion RRF → 10 candidats
    │
    ▼
Re-ranker CrossEncoder → 5 meilleurs
    │
    ▼
LLM → Réponse
```

Chaque étape améliore la précision par rapport à la précédente.
