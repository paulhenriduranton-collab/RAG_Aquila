# 01 — Présentation du projet

## C'est quoi ce projet ?

**Aquila** est un système de type **RAG** (*Retrieval-Augmented Generation*).

En français simple : un programme qui permet de **poser des questions à une IA sur tes propres documents**, et d'obtenir des réponses basées uniquement sur le contenu de ces documents — sans connexion internet, sans envoyer tes données sur un serveur externe.

---

## La différence avec ChatGPT

| ChatGPT | Projet Aquila |
|---|---|
| Connaît des milliards de pages internet | Ne connaît que TES documents |
| Tourne sur les serveurs d'OpenAI | Tourne entièrement sur ta machine |
| Tes données partent sur internet | Tes données restent privées |
| Peut inventer des réponses sur n'importe quel sujet | Ne répond que si l'info est dans tes fichiers |
| Interface web fournie par OpenAI | Interface Open WebUI hébergée en local |

---

## Le cas d'usage concret

Tu as des fichiers PDF de cours :
- `ENS.pdf`
- `SORBONNE.pdf`

Tu poses la question dans l'interface Open WebUI : *"Qu'est-ce qu'un espace de Banach ?"*

Le système :
1. Cherche dans tes PDFs les passages qui parlent d'espaces de Banach
2. Sélectionne les 5 passages les plus pertinents (via recherche hybride BM25 + vectorielle)
3. Demande à `gemma2:2b` de rédiger une réponse à partir de ces passages
4. Affiche la réponse dans l'interface, en streaming (comme ChatGPT)

---

## Pourquoi "RAG" ?

- **R**etrieval = *Récupération* — on cherche les bons passages dans les documents
- **A**ugmented = *Augmenté* — on enrichit la question avec ces passages avant de l'envoyer au LLM
- **G**eneration = *Génération* — le LLM génère une réponse à partir du contexte fourni

---

## Les trois modes d'utilisation

| Mode | Commande | Usage |
|---|---|---|
| Interface web (Open WebUI) | `docker compose up -d` + `uvicorn src.api:app` | Usage normal, interface type ChatGPT |
| Terminal | `python src/ask.py` | Debug — affiche les logs de retrieval |
| Évaluation | `python evaluation/eval_retrieval.py` | Mesure la qualité du retrieval |

---

## Ce que le projet N'est PAS

- Ce n'est pas ChatGPT — il ne répond pas à des questions générales hors documents
- Ce n'est pas un moteur de recherche — il génère une réponse rédigée, pas une liste de liens
- Ce n'est pas infaillible — si l'information n'est pas dans les documents, il ne peut pas répondre correctement
