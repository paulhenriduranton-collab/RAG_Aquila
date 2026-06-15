# 01 — Présentation du projet

## C'est quoi ce projet ?

**RAG Aquila** est un système de type **RAG** (*Retrieval-Augmented Generation*) **agentique**.

En français simple : un programme qui permet de **poser des questions à une IA sur tes propres documents**, et d'obtenir des réponses basées uniquement sur le contenu de ces documents — sans connexion internet, sans envoyer tes données sur un serveur externe.

---

## La différence avec ChatGPT

| ChatGPT | Projet RAG Aquila |
|---|---|
| Connaît des milliards de pages internet | Ne connaît que TES documents |
| Tourne sur les serveurs d'OpenAI | Tourne entièrement sur ta machine |
| Tes données partent sur internet | Tes données restent privées |
| Peut inventer des réponses sur n'importe quel sujet | Ne répond que si l'info est dans tes fichiers |
| Interface fournie par OpenAI | Interface Open WebUI hébergée en local |

---

## Le cas d'usage concret

Tu as des fichiers PDF de brochures universitaires :
- `Brochure-2024-2025.pdf` (ENS DMA)
- `Brochure Master2526_1.pdf` (Sorbonne)

Tu poses la question dans l'interface Open WebUI : *"Quels sont les cours obligatoires de L3 à l'ENS DMA ?"*

Le système agentique :
1. Identifie que la question concerne `Brochure-2024-2025.pdf` (ENS)
2. Génère une réponse fictive (HyDE) pour mieux orienter la recherche sémantique
3. Cherche dans ce fichier les passages les plus pertinents (recherche hybride : sémantique MMR + BM25)
4. Fusionne les deux listes de résultats avec RRF (Reciprocal Rank Fusion)
5. Re-classe les 10 meilleurs passages avec un cross-encoder
6. Évalue si les 5 chunks retenus sont suffisants pour répondre
7. Si non : reformule la requête et relance une recherche ciblée
8. Génère la réponse en s'appuyant uniquement sur les passages trouvés

---

## Pourquoi "RAG" ?

- **R**etrieval = *Récupération* — on cherche les bons passages dans les documents
- **A**ugmented = *Augmenté* — on enrichit la question avec ces passages avant de l'envoyer au LLM
- **G**eneration = *Génération* — le LLM génère une réponse à partir du contexte fourni

Sans le "R", le LLM répondrait de mémoire (et inventerait). Avec le "R", il est contraint de répondre uniquement à partir des passages fournis.

---

## Les cinq modes d'utilisation

| Mode | Commande | Usage |
|---|---|---|
| Interface chat | `uvicorn api_server:app` + Open WebUI | Usage normal |
| Terminal RAG classique | `python src/ask.py` | Debug — affiche tous les logs de retrieval |
| Terminal agentique | `python src/agent.py` | Debug agentique — plus lent, plus précis |
| Indexation | `python src/ingest.py` | À relancer si tu changes tes documents |
| Évaluation | `python src/run_agentic_all.py` puis `python src/evaluate_ragas.py` | Mesure la qualité sur 40 questions |

---

## Ce que le projet N'est PAS

- Ce n'est pas ChatGPT — il ne répond pas à des questions générales hors documents
- Ce n'est pas un moteur de recherche — il génère une réponse rédigée, pas une liste de liens
- Ce n'est pas infaillible — si l'information n'est pas dans les documents, il ne peut pas répondre correctement

---

## Ce qui rend ce RAG avancé

### Pipeline de retrieval

La plupart des RAG basiques font : question → recherche sémantique → LLM. Ce projet va plus loin :

```
Question
    │
    ├── HyDE : génère une réponse fictive pour orienter la recherche sémantique
    │
    ├── Recherche sémantique MMR (20 candidats diversifiés)
    ├── Recherche BM25/lexicale normalisée (20 candidats)
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

### Pipeline agentique (LangGraph)

Au-delà du RAG classique, le pipeline agentique ajoute une boucle de contrôle :

```
Question
    │
    ▼
identify_sources
    → le LLM choisit quel(s) document(s) concernent la question
    │
    ▼
retrieve
    → pipeline RAG hybride sur la/les source(s) ciblée(s)
    │
    ▼
grade_documents
    → le LLM évalue si les chunks sont suffisants pour répondre
    │
    ├── OUI → generate → Réponse
    │
    └── NON (si < MAX_ATTEMPTS=2 tentatives)
            │
            ▼
        rewrite_query
            → le LLM reformule la requête sur ce qui manque
            │
            └── retrieve (nouvelle tentative)
```

### Contextual retrieval

Chaque chunk est préfixé d'une ligne de contexte avant d'être indexé :
```
[Brochure-2024-2025.pdf | p.12 | DMA > Organisation > Cours | cours obligatoires, ECTS, L3]

## Cours communs de L3
Les quatre cours obligatoires sont...
```

Cette ligne permet à l'embedding ET à BM25 de comprendre le contexte structurel d'un chunk isolé — sans elle, un chunk extrait de son document ne dit pas de lui-même dans quel établissement ou quelle section il se trouve.
