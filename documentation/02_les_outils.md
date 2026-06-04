# 02 — Les outils utilisés

## Vue d'ensemble

| Outil | Rôle dans le projet |
|---|---|
| Ollama | Héberge les modèles IA en local |
| bge-m3 | Transforme le texte en vecteurs (embeddings) |
| BM25 | Recherche par mots-clés exacts |
| pymupdf4llm | Extrait les PDFs en Markdown propre |
| gemma2:2b | Génère les réponses en langage naturel |
| ChromaDB | Stocke et recherche les vecteurs |
| LangChain | Colle tous les composants ensemble |
| FastAPI | Expose le pipeline RAG comme une API web |
| Open WebUI | Interface utilisateur type ChatGPT |
| RAGAS | Évalue la qualité des réponses générées |

---

## 1. Ollama

**Ce que c'est :** Un logiciel qui fait tourner des modèles d'IA directement sur ton ordinateur, sans connexion internet.

**Ce qu'il fait ici :** Il héberge deux modèles :
- `bge-m3` pour transformer du texte en vecteurs
- `gemma2:2b` pour générer les réponses

**Analogie :** C'est un serveur local — il reçoit des requêtes (`embed ce texte`, `génère une réponse`) et les envoie au bon modèle.

---

## 2. bge-m3 (modèle d'embedding)

**Ce que c'est :** Un modèle spécialisé dans la transformation de texte en vecteurs, développé par BAAI (Beijing Academy of AI).

**Ce qu'il fait ici :** Pour chaque morceau de texte, il produit une liste de **1024 nombres** qui représentent le *sens* du texte. Deux textes qui parlent du même sujet auront des vecteurs proches.

**Pourquoi bge-m3 ?**

| Modèle | Langues | Fenêtre contexte | Adapté maths FR |
|---|---|---|---|
| nomic-embed-text | Anglais surtout | 8192 tokens | Non |
| mxbai-embed-large | Multilingue | **512 tokens** | Partiel (trop court) |
| **bge-m3** | **100+ langues dont FR** | **8192 tokens** | **Oui** |

**Important :** Si tu changes de modèle d'embedding, tu dois supprimer `vector_db/` et relancer `ingest.py`. Les vecteurs produits par bge-m3 (1024 dimensions) sont incompatibles avec ceux de nomic-embed-text (768 dimensions).

---

## 3. BM25 (recherche lexicale)

**Ce que c'est :** Un algorithme de recherche par mots-clés, utilisé dans des moteurs de recherche comme Elasticsearch.

**Ce qu'il fait ici :** En parallèle de la recherche sémantique, BM25 cherche les chunks qui contiennent exactement les mots de la question.

**La complémentarité avec la recherche sémantique :**

| Recherche sémantique (bge-m3) | Recherche lexicale (BM25) |
|---|---|
| Cherche par *sens* | Cherche par *mots exacts* |
| "espace complet" → trouve "Banach" | "différentielle" → trouve "différentielle" |
| Bonne sur les concepts | Bonne sur les termes techniques et formules |

---

## 4. pymupdf4llm (extraction PDF)

**Ce que c'est :** Une extension de PyMuPDF conçue pour produire du Markdown propre depuis les PDFs, optimisée pour les LLMs.

**Pourquoi pas PyMuPDF brut ?**

```
# PyMuPDF brut
"SoientX⊂R d un ouvert etf:X→R m d2fdx2"

# pymupdf4llm
"Soient X ⊂ R^d un ouvert et f : X → R^m"
```

pymupdf4llm préserve les titres de sections (`## Chapitre 3`), les tableaux, et place correctement les espaces autour des symboles mathématiques.

**Limite :** Il ne reconnaît pas les formules LaTeX. Les intégrales et fractions restent en texte brut.

---

## 5. gemma2:2b (modèle de génération)

**Ce que c'est :** Un LLM de Google, 2 milliards de paramètres, tournant en local via Ollama.

**Ce qu'il fait ici :** Il reçoit le prompt (question + 5 passages + instructions) et génère la réponse en français, uniquement à partir du contexte fourni.

**Comparaison :**

| Modèle | Taille | Qualité | Vitesse |
|---|---|---|---|
| gemma2:2b | 1.6 GB | Correcte | Très rapide |
| llama3.1:8b | 4.7 GB | Bonne | Moyen |
| gemma2:9b | 5.5 GB | Très bonne | Lent |

`gemma2:2b` est suffisant pour extraire et reformuler des informations depuis un contexte fourni. Pour des questions de raisonnement complexe, `llama3.1:8b` donne de meilleurs résultats.

---

## 6. ChromaDB

**Ce que c'est :** Une base de données spécialisée dans le stockage et la recherche de vecteurs.

**Ce qu'il fait ici :** Stocke les 1024 nombres de chaque chunk dans `vector_db/chroma.sqlite3`. Quand tu poses une question, il calcule les 20 vecteurs les plus proches du vecteur de ta question.

**La différence avec une base classique :**
- Base classique : cherche "Banach" → trouve les lignes qui contiennent exactement "Banach"
- ChromaDB : cherche "espace complet" → trouve les passages sur "Banach", "Cauchy", "convergence"

---

## 7. LangChain

**Ce que c'est :** Une librairie Python qui sert de colle entre tous les composants.

**Ce qu'il fournit ici :**
- `RecursiveCharacterTextSplitter` → découpe le texte
- `OllamaEmbeddings` → appelle bge-m3 via Ollama
- `Chroma` → gère la base vectorielle
- `OllamaLLM` → appelle gemma2:2b via Ollama

---

## 8. FastAPI + Open WebUI

**FastAPI** est un framework Python qui expose le pipeline RAG comme une API web compatible avec le format OpenAI. Quand Open WebUI envoie une requête, FastAPI fait tourner le retrieval + génération et renvoie les tokens en streaming.

**Open WebUI** est une interface web (lancée via Docker) qui ressemble à ChatGPT. Elle se connecte à l'API FastAPI locale et affiche les réponses.

```
Navigateur (Open WebUI :3000)
    ↓  POST /v1/chat/completions
API FastAPI (:8000)
    ↓  retrieval hybride
ChromaDB + BM25
    ↓  prompt construit
Ollama / gemma2:2b
    ↓  tokens streamés
Navigateur (réponse en temps réel)
```

---

## 9. RAGAS (évaluation)

**Ce que c'est :** Un framework d'évaluation automatique pour les systèmes RAG. Il utilise un LLM comme juge pour noter la qualité des réponses.

**Ce qu'il mesure :**

| Métrique | Question posée |
|---|---|
| Faithfulness | La réponse invente-t-elle des informations absentes des chunks ? |
| Answer Relevancy | La réponse répond-elle vraiment à la question posée ? |
| Context Precision | Les chunks récupérés sont-ils utiles à la réponse ? |

**Important :** RAGAS évalue la qualité des *réponses*. Pour évaluer la qualité du *retrieval* (est-ce que les bons chunks sont récupérés ?), on utilise `evaluation/eval_retrieval.py` qui calcule Recall@K et MRR.
