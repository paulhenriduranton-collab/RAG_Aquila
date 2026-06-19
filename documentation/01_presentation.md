# 01 — Présentation du projet

## C'est quoi ce projet ?

**RAG Aquila** est un système de type **RAG** (*Retrieval-Augmented Generation*) **agentique**.

En français simple : un programme qui permet de **poser des questions à une IA sur tes propres documents**, et d'obtenir des réponses basées uniquement sur le contenu de ces documents — sans connexion internet, sans envoyer tes données sur un serveur externe.

---

## La différence avec ChatGPT

| ChatGPT | Projet RAG Aquila |
|---|---|
| Connaît des milliards de pages internet | Ne connaît que TES documents |
| Tourne sur les serveurs d'OpenAI | Tourne entièrement sur ta machine (ou sur Colab) |
| Tes données partent sur internet | Tes données restent privées |
| Peut inventer des réponses sur n'importe quel sujet | Ne répond que si l'info est dans tes fichiers |
| Interface fournie par OpenAI | Interface Open WebUI locale ou Gradio sur Colab |

---

## Le cas d'usage concret

Tu as des fichiers PDF de brochures universitaires :
- `ENS.pdf` (brochure ENS DMA 2024-2025)
- `SORBONNE.pdf` (brochure Master Sorbonne 2025-2026)

Tu poses la question dans l'interface Open WebUI : *"Quels sont les cours obligatoires de L3 à l'ENS DMA ?"*

Le système agentique :
1. Identifie que la question concerne `ENS.pdf` et classifie la difficulté (ici : niveau 2, synthèse)
2. Génère une réponse fictive (HyDE) pour mieux orienter la recherche sémantique
3. Cherche dans ce fichier les passages les plus pertinents (recherche hybride : sémantique MMR + BM25)
4. Fusionne les deux listes de résultats avec RRF (Reciprocal Rank Fusion)
5. Supprime les quasi-doublons (Jaccard > 80 %)
6. Re-classe les meilleurs passages avec un cross-encoder (seuil de pertinence 0.5)
7. Évalue si les 5 chunks retenus sont suffisants pour répondre
8. Si non : remonte la difficulté, reformule la requête et relance une recherche ciblée
9. Génère la réponse en s'appuyant uniquement sur les passages trouvés

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
| Interface chat | `uvicorn api_server:app` + Open WebUI | Usage normal en local |
| Interface Gradio | `colab_run.ipynb` étape 4b | Usage interactif sur Colab, avec surlignage PDF |
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
    │          (désactivé pour les questions factuelles de difficulté 1)
    │
    ├── Recherche sémantique MMR (25 candidats diversifiés parmi 100)
    ├── Recherche BM25/lexicale normalisée (25 candidats)
    │
    ▼
Fusion RRF → 15 candidats
    │
    ▼
Déduplication Jaccard (seuil 80 %) → ~10-12 candidats
    │
    ▼
Re-ranker CrossEncoder (seuil 0.5) → 5 meilleurs
    │
    ▼
LLM → Réponse
```

### Pipeline agentique (LangGraph)

Au-delà du RAG classique, le pipeline agentique ajoute une boucle de contrôle intelligente :

```
Question
    │
    ▼
identify_sources
    → le LLM choisit quel(s) document(s) et classifie la difficulté (1/2/3)
    → si difficulté 3 : décompose en sous-requêtes indépendantes
    │
    ▼
retrieve
    → pipeline RAG hybride sur la/les source(s) ciblée(s)
    → si difficulté 3 : un retrieval par sous-requête, puis fusion + re-rank global
    │
    ├── difficulté 1 → vérifie la troncature des chunks
    │     → pas de troncature → generate directement (pas d'appel LLM de grading)
    │     → troncature détectée → upgrade_difficulty → rewrite_query → retrieve
    │
    └── difficulté 2/3 → grade_documents
          → le LLM évalue si les chunks sont suffisants pour répondre
          │
          ├── OUI (ou max tentatives atteint) → generate → Réponse
          │
          └── NON + tentatives restantes
                  │
                  ├── difficulté < 3 → upgrade_difficulty → rewrite_query → retrieve
                  └── difficulté 3 → rewrite_query → retrieve (2ème et dernier)
```

### Découpage agentique (et pas page par page)

Une page de PDF n'est pas une unité de sens — une section peut commencer en bas d'une page et continuer sur la suivante. Le pipeline d'ingestion concatène donc toutes les pages d'un document en un texte continu, puis découpe sur la structure logique (titres Markdown, puis un LLM qui affine la coupe entre sous-sections cohérentes). Un chunk peut donc couvrir plusieurs pages d'origine ; sa métadonnée `page` retient simplement la page où sa section commence. Si les chunks d'un document ressemblent malgré tout à un découpage page par page, c'est souvent parce que la brochure elle-même place un titre par page — pas parce que le pipeline est borné à la page. Voir [03_fonctionnement_detaille.md](03_fonctionnement_detaille.md) pour le détail.

### Contextual retrieval

Chaque chunk est préfixé d'une ligne de contexte avant d'être indexé :
```
[ENS.pdf | p.12 | Cours communs de L3 | cours obligatoires, ects, l3]

## Cours communs de L3
Les quatre cours obligatoires sont...
```

Cette ligne est construite **entièrement sans LLM** : le nom de fichier et la page viennent des métadonnées, le chemin de titres est extrait par regex sur les `#`/`##`/`###` présents dans le chunk, et les mots-clés sont les mots/bigrammes les plus fréquents du chunk après suppression des stopwords français (comptage de fréquence, zéro hallucination possible).

Cette ligne permet à l'embedding ET à BM25 de comprendre le contexte structurel d'un chunk isolé — sans elle, un chunk extrait de son document ne dit pas de lui-même dans quel établissement ou quelle section il se trouve.
