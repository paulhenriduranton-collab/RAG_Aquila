# 02 — Les outils utilisés

## Vue d'ensemble

| Outil | Rôle dans le projet |
|---|---|
| Ollama | Héberge les modèles IA en local |
| bge-m3 | Transforme le texte en vecteurs (embeddings) |
| BM25 | Recherche par mots-clés exacts |
| pymupdf4llm | Extrait les PDFs en Markdown propre, page par page |
| ftfy | Répare les encodages cassés dans les textes extraits de PDF |
| gemma4:12b | Split agentique (ingestion), HyDE, grading, reformulation, génération, évaluation RAGAS |
| gemma2:2b | Optionnel : juge RAGAS allégé sur Colab (remplace gemma4:12b pour aller plus vite) |
| ChromaDB | Stocke et recherche les vecteurs |
| LangChain | Colle tous les composants ensemble |
| LangGraph | Orchestre le pipeline agentique (machine à états) |
| CrossEncoder BAAI/bge-reranker-v2-m3 | Re-classe les chunks par pertinence réelle (re-ranking) |
| RAGAS | Évalue la qualité du pipeline sur 5 métriques standards |
| Open WebUI + FastAPI | Interface chat locale, serveur API compatible OpenAI |
| Gradio | Interface web interactive sur Colab avec surlignage PDF |

---

## 1. Ollama

**Ce que c'est :** Un logiciel qui fait tourner des modèles d'IA directement sur ton ordinateur, sans connexion internet.

**Ce qu'il fait ici :** Il héberge les modèles utilisés par le projet :
- `bge-m3` pour transformer du texte en vecteurs (embeddings)
- `gemma4:12b` pour le split agentique des chunks (ingestion), HyDE, le grading des chunks, la reformulation de requêtes et la génération finale
- `gemma2:2b`, optionnel, comme juge RAGAS plus rapide sur Colab (non utilisé en local)

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

**Important :** Le modèle d'embedding doit être **identique** dans `ingest.py` et `ask.py`. Si tu changes de modèle, tu dois supprimer la base vectorielle et relancer `ingest.py`. Les vecteurs produits par bge-m3 (1024 dimensions) sont incompatibles avec ceux d'un autre modèle.

---

## 3. BM25 (recherche lexicale)

**Ce que c'est :** Un algorithme de recherche par mots-clés, utilisé dans des moteurs de recherche comme Elasticsearch.

**Ce qu'il fait ici :** En parallèle de la recherche sémantique, BM25 cherche les chunks qui contiennent exactement les mots de la question. Il construit un index lexical à partir de tous les chunks stockés dans ChromaDB (une seule fois par session, puis mis en cache).

**La normalisation BM25 dans ce projet :**

La fonction `_tokenize()` applique une normalisation avant indexation :
- mise en minuscules
- suppression des points médians (`étudiant·e·s` → `etudiantes`)
- décomposition Unicode (NFD) + suppression des diacritiques (accents)
- extraction des tokens de 2+ caractères (filtre la ponctuation)

Sans cette normalisation, BM25 raterait les mots accentués selon le PDF et ne ferait pas le lien entre "etudiant" et "étudiant".

**La complémentarité avec la recherche sémantique :**

| Recherche sémantique (bge-m3) | Recherche lexicale (BM25) |
|---|---|
| Cherche par *sens* | Cherche par *mots exacts* |
| "espace complet" → trouve "Banach" | "différentielle" → trouve "différentielle" |
| Bonne sur les concepts | Bonne sur les termes techniques et sigles |

---

## 4. pymupdf4llm (extraction PDF)

**Ce que c'est :** Une extension de PyMuPDF conçue pour produire du Markdown propre depuis les PDFs, optimisée pour les LLMs.

**Comment ça fonctionne ici :**

```python
# Extraction page par page : chaque page = un Document indépendant
pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
```

L'option `page_chunks=True` est importante : chaque page est retournée comme un document indépendant. Cela garantit qu'un tableau qui tient sur une page ne sera jamais coupé entre deux morceaux lors du découpage en chunks.

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

## 5. ftfy (réparation d'encodage)

**Ce que c'est :** Une bibliothèque Python spécialisée dans la détection et la réparation des problèmes d'encodage de texte.

**Ce qu'il fait ici :** Appliqué immédiatement après l'extraction par pymupdf4llm, sur chaque page :

```python
# Répare les accents cassés avant tout traitement
text = ftfy.fix_text(page["text"])
```

Certains PDFs contiennent des accents mal encodés (`a → à`, `´e → é`). ftfy les détecte et les corrige automatiquement. Sans cette étape, BM25 raterait les mots accentués mal encodés.

---

## 6. gemma4:12b (modèle principal)

**Ce que c'est :** Un LLM de Google, 12 milliards de paramètres, tournant en local via Ollama.

**Ce qu'il fait ici :** Cinq rôles distincts :

1. **Split agentique** — Dans `ingest.py`, à l'intérieur de chaque bloc thématique (issu du pré-découpage par titres), insère des marqueurs `===SPLIT===` entre les sous-sections logiquement distinctes. Une validation déterministe (≥60 % des mots du texte d'origine doivent être retrouvés dans la réponse) rejette les réponses où le LLM aurait reformulé ou résumé le texte au lieu de se limiter à y insérer des marqueurs ; dans ce cas, le bloc original est conservé tel quel sans découpe.

2. **HyDE** — Dans `ask.py`, génère une réponse fictive stylistiquement proche des brochures indexées. Cette réponse est utilisée à la place de la question pour la recherche sémantique (meilleure similarité cosinus). Désactivé pour les questions factuelles de difficulté 1.

3. **Grading** — Dans `agent.py`, évalue si les chunks récupérés sont suffisants pour répondre à la question, et identifie précisément ce qui manque.

4. **Reformulation** — Dans `agent.py`, reformule la requête en ciblant ce qui manque selon le verdict du grading.

5. **Génération** — Reçoit le prompt (question + 5 passages + instructions) et génère la réponse en français, uniquement à partir du contexte fourni. Paramétré avec `temperature=0` pour des réponses déterministes.

**Évaluation RAGAS** — Dans `evaluate_ragas.py`, joue par défaut le rôle de juge LLM pour les 5 métriques RAGAS (`EVAL_LLM = "gemma4:12b"`). Sur Colab, le notebook peut le remplacer par `gemma2:2b` (3-5x plus rapide) pour accélérer l'évaluation — ce dernier n'a aucun autre rôle dans le projet.

**Comparaison :**

| Modèle | Taille | Qualité | Vitesse |
|---|---|---|---|
| gemma2:2b | 1.6 GB | Correcte (suffisant comme juge RAGAS rapide) | Très rapide |
| llama3.1:8b | 4.7 GB | Bonne | Moyen |
| **gemma4:12b** | **~8 GB** | **Très bonne** | **Lent (Colab GPU recommandé)** |

`gemma4:12b` est nécessaire pour les tâches de raisonnement complexes (split agentique, grade_documents, rewrite_query, génération finale). Pour une utilisation sans GPU, `llama3.1:8b` est un bon compromis.

---

## 7. ChromaDB

**Ce que c'est :** Une base de données spécialisée dans le stockage et la recherche de vecteurs.

**Ce qu'il fait ici :** Stocke les 1024 nombres de chaque chunk dans un fichier `chroma.sqlite3`. Quand tu poses une question, il calcule les vecteurs les plus proches du vecteur de ta question (recherche MMR ou cosinus).

**Double chemin de la base vectorielle :**

| Contexte | Chemin | Raison |
|---|---|---|
| Local (ingest.py) | `C:/vector_db_aquila` | Hors OneDrive — SQLite corrompu par la synchro cloud |
| Local (ask.py) / Colab | `vector_db/` (dans le repo) | Versionnée dans git pour Colab |

**Pourquoi hors OneDrive en local ?** SQLite (le moteur de ChromaDB) et la synchronisation cloud sont incompatibles : OneDrive peut verrouiller le fichier `.sqlite3` en cours d'écriture, ce qui corrompt la base.

**La différence avec une base classique :**
- Base classique : cherche "Banach" → trouve les lignes qui contiennent exactement "Banach"
- ChromaDB : cherche "espace complet" → trouve les passages sur "Banach", "Cauchy", "convergence"

**L'index BM25 est construit séparément** depuis les textes stockés dans ChromaDB (`vector_db._collection.get(...)`), une seule fois par session.

---

## 8. LangChain

**Ce que c'est :** Une librairie Python qui sert de colle entre tous les composants.

**Ce qu'il fournit ici :**

| Classe | Usage |
|---|---|
| `RecursiveCharacterTextSplitter` | Fallback déterministe : redécoupe sans overlap les blocs encore > 1500 caractères après le split agentique |
| `OllamaEmbeddings` | Appelle bge-m3 via Ollama pour calculer les embeddings |
| `Chroma` | Gère la base vectorielle (écriture depuis ingest.py, lecture depuis ask.py) |
| `OllamaLLM` | Appelle gemma4:12b (split agentique côté ingestion ; HyDE, grading, reformulation, génération côté ask.py/agent.py) |
| `TextLoader` | Charge les fichiers .txt |
| `Docx2txtLoader` | Charge les fichiers .docx |

---

## 9. LangGraph (pipeline agentique)

**Ce que c'est :** Une librairie de LangChain pour construire des pipelines IA sous forme de graphes de nœuds et d'arêtes, avec état partagé et transitions conditionnelles.

**Ce qu'il fait ici :** Orchestre le pipeline agentique dans `agent.py`. Le graphe :

```
START
  │
  ▼
identify_sources  →  retrieve  ─┬─  [diff. 1] → troncation? ─┬─ non → generate → END
                        ↑       │                              └─ oui → upgrade_difficulty
                        │       └─  [diff. 2/3] → grade_documents
                        │                              │
                        │                    OUI      ▼
                        │               generate  ←  suffisant?
                        │                              │
                        │                             NON
                        │                              ▼
                        └── rewrite_query  ←  upgrade_difficulty (si diff < 3)
```

L'état du graphe (`AgentState`) est un dictionnaire partagé entre tous les nœuds : question, requête courante, sources ciblées, difficulté, sous-requêtes, pool de chunks, verdict de suffisance, nombre de tentatives, réponse finale.

**Pourquoi LangGraph et pas un simple `if` ?** LangGraph permet de visualiser le graphe, de l'interrompre et de l'inspecter à tout moment, et de gérer proprement les boucles conditionnelles avec un état partagé immuable entre les nœuds.

---

## 10. CrossEncoder BAAI/bge-reranker-v2-m3 (re-ranker)

**Ce que c'est :** Un modèle de la bibliothèque `sentence-transformers`, téléchargé automatiquement depuis HuggingFace au premier lancement (~471 Mo).

**Nom complet :** `BAAI/bge-reranker-v2-m3` — modèle multilingue du même laboratoire que bge-m3 (BAAI), cohérent pour une utilisation ensemble.

**Ce qu'il fait ici :** Re-classe les candidats issus de la fusion RRF (après déduplication). Il reçoit des paires `(question, chunk)` et prédit un score de pertinence pour chacune :

```python
# Le re-ranker lit chaque paire conjointement — pas séparément comme les embeddings
pairs = [("Quels sont les cours obligatoires ?", "Les quatre cours communs sont..."),
         ("Quels sont les cours obligatoires ?", "La bibliothèque est ouverte..."), ...]
scores = reranker.predict(pairs)  # → [8.4, -1.2, ...]
```

Les scores sont des logits centrés sur 0 (pas des probabilités). Un score négatif signifie que le modèle juge le chunk hors-sujet. Les chunks sous le seuil `RERANK_THRESHOLD = 0.5` sont écartés.

**Pourquoi un seuil de 0.5 (et pas 0.0) ?** Pour bge-reranker-v2-m3, sigmoid(0) = 0.5 = pertinence "moyenne". Un seuil de 0.5 est plus sélectif que 0.0 : il ne garde que les chunks jugés clairement pertinents, pas les cas limites.

**Pourquoi c'est plus précis qu'un embedding ?**

Un embedding encode question et chunk **séparément** — il mesure leur proximité dans l'espace vectoriel mais sans voir les deux ensembles. Le CrossEncoder lit la question et le chunk **ensemble** dans un seul passage, ce qui lui permet de comprendre des relations subtiles ("ce passage répond-il vraiment à cette question ?").

**Où il est chargé :** Une seule fois au démarrage, en variable globale dans `ask.py` :
```python
# Chargé une fois, réutilisé pour toutes les questions de la session
reranker = CrossEncoder(RERANK_MODEL)
```

---

## 11. RAGAS (évaluation)

**Ce que c'est :** Une librairie Python standard pour évaluer les pipelines RAG avec des métriques LLM-judge.

**Ce qu'il fait ici :** Dans `evaluate_ragas.py`, il évalue chaque résultat du pipeline agentique sur 5 métriques :

| Métrique | Question posée au juge | Ground truth nécessaire ? |
|---|---|---|
| **Faithfulness** | La réponse invente-t-elle des choses absentes des chunks ? | Non |
| **AnswerRelevancy** | La réponse répond-elle à la question posée ? | Non |
| **ContextPrecision** | Les chunks récupérés sont-ils pertinents pour cette question ? | Non |
| **ContextRecall** | Les chunks couvrent-ils tout ce que contient la réponse de référence ? | Oui |
| **AnswerCorrectness** | La réponse est-elle correcte par rapport à la référence ? | Oui |

RAGAS utilise Ollama (via une API compatible OpenAI sur `http://localhost:11434/v1`) — aucun appel à OpenAI ou à un service externe.

---

## 12. Open WebUI + FastAPI (interface locale)

**Open WebUI** : Interface web de chat type ChatGPT, hébergée en local. Se connecte à n'importe quelle API compatible OpenAI.

**FastAPI (`api_server.py`)** : Serveur minimal qui expose `ask_question_agentic()` via deux routes :
- `GET /v1/models` — retourne la liste des modèles disponibles
- `POST /v1/chat/completions` — reçoit un message, appelle le pipeline agentique, renvoie la réponse au format OpenAI

Cette architecture permet de brancher Open WebUI sur le RAG agentique sans modifier aucun fichier existant.

**Lancement :**
```powershell
# Depuis src/ — expose le pipeline agentique sur le port 8001
uvicorn api_server:app --host 0.0.0.0 --port 8001
```

L'interface est accessible sur **http://localhost:3000** après lancement d'Open WebUI.

---

## 13. Gradio (interface Colab)

**Ce que c'est :** Une librairie Python pour créer des interfaces web interactives en quelques lignes.

**Ce qu'il fait ici :** Dans `colab_run.ipynb` (étape 4b), Gradio expose une interface web avec trois panneaux :
1. **Réponse** — le texte généré par le pipeline agentique
2. **Chunks récupérés** — détail HTML des chunks avec source, page et score de re-ranking
3. **Pages PDF** — galerie d'images des pages sources avec le chunk **surligné en bleu**

Le surlignage utilise PyMuPDF (`fitz`) pour chercher des fragments du chunk dans la page PDF et les marquer en bleu.

**Lien public** : Gradio génère automatiquement un lien `.gradio.live` accessible pendant 72h, ce qui permet de partager l'interface avec quelqu'un sans qu'il ait besoin de Colab.
