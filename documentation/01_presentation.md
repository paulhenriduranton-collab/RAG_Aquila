# 01 — Présentation du projet

## C'est quoi ce projet ?

Ce projet s'appelle **Aquila**. C'est un système de type **RAG** (*Retrieval-Augmented Generation*).

En français simple : c'est un programme qui te permet de **poser des questions à une intelligence artificielle sur tes propres documents**, et d'obtenir des réponses précises basées uniquement sur le contenu de ces documents.

---

## La différence avec ChatGPT

| ChatGPT | Projet Aquila |
|---|---|
| Connaît des milliards de pages internet | Ne connaît que TES documents |
| Tourne sur les serveurs d'OpenAI | Tourne sur TA machine |
| Tes données partent sur internet | Tes données restent privées |
| Peut inventer des réponses sur n'importe quel sujet | Ne répond que si l'info est dans tes fichiers |

---

## Le cas d'usage concret

Tu as 3 fichiers PDF de cours de maths :
- `Poly-L3-StatMath.pdf`
- `analyse_fonctionnelle.pdf`
- `calcul_diff.pdf`

Tu poses la question : *"Qu'est-ce qu'un espace de Banach ?"*

Le système :
1. Cherche dans tes 3 PDFs les passages qui parlent d'espaces de Banach
2. Extrait les 4 passages les plus pertinents
3. Demande à l'IA de rédiger une réponse à partir de ces passages
4. Affiche la réponse dans une interface web

---

## Pourquoi ce nom "RAG" ?

- **R**etrieval = *Récupération* — on cherche les bons passages dans les documents
- **A**ugmented = *Augmenté* — on enrichit la question avec ces passages
- **G**eneration = *Génération* — l'IA génère une réponse à partir de tout ça

---

## Ce que le projet N'est PAS

- Ce n'est pas ChatGPT — il ne sait pas répondre à des questions générales hors documents
- Ce n'est pas un moteur de recherche — il génère une réponse rédigée, pas une liste de liens
- Ce n'est pas infaillible — si l'information n'est pas dans les documents, il ne peut pas répondre correctement
