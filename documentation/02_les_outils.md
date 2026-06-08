# 02 — Les outils utilisés

## Vue d'ensemble

| Outil | Rôle dans le projet |
|---|---|
| Ollama | Héberge les modèles IA en local |
| bge-m3 | Transforme le texte en vecteurs (embeddings) |
| BM25 | Recherche par mots-clés exacts |
| pymupdf4llm | Extrait les PDFs en Markdown propre, page par page |
| ftfy | Répare les encodages cassés dans les textes extraits de PDF |
| gemma2:2b | Génère les réponses ET joue le rôle de juge dans l'évaluation |
| ChromaDB | Stocke et recherche les vecteurs |
| LangChain | Colle tous les composants ensemble |
| CrossEncoder mmarco | Re-classe les chunks par pertinence réelle (re-ranking) |
| Streamlit | Interface web locale pour poser des questions |

---

## 1. Ollama

**Ce que c'est :** Un logiciel qui fait tourner des modèles d'IA directement sur ton ordinateur, sans connexion internet.

**Ce qu'il fait ici :** Il héberge deux modèles :
- `bge-m3` pour transformer du texte en vecteurs
- `gemma2:2b` pour générer les réponses et évaluer les métriques

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

**Ce qu'il fait ici :** En parallèle de la recherche sémantique, BM25 cherche les chunks qui contiennent exactement les mots de la question. Il construit un index lexical à partir de tous les chunks stockés dans ChromaDB (une seule fois par session, puis mis en cache).

**La complémentarité avec la recherche sémantique :**

| Recherche sémantique (bge-m3) | Recherche lexicale (BM25) |
|---|---|
| Cherche par *sens* | Cherche par *mots exacts* |
| "espace complet" → trouve "Banach" | "différentielle" → trouve "différentielle" |
| Bonne sur les concepts | Bonne sur les termes techniques et formules |

**Implémentation :** La bibliothèque `rank_bm25` implémente `BM25Okapi`. Chaque chunk est tokenisé en minuscules (`t.lower().split()`), ce qui signifie que la recherche n'est pas sensible à la casse.

---

## 4. pymupdf4llm (extraction PDF)

**Ce que c'est :** Une extension de PyMuPDF conçue pour produire du Markdown propre depuis les PDFs, optimisée pour les LLMs.

**Comment ça fonctionne ici :**

```python
pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
```

L'option `page_chunks=True` est importante : chaque page est retournée comme un document indépendant. Cela garantit qu'un tableau qui tient sur une page ne sera jamais coupé en deux morceaux lors du découpage en chunks.

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
text = ftfy.fix_text(page["text"])
```

Certains PDFs contiennent des accents mal encodés (`a → à`, `´e → é`). ftfy les détecte et les corrige automatiquement avant tout traitement. Sans cette étape, BM25 raterait les mots accentués mal encodés.

---

## 6. gemma2:2b (modèle de génération et juge d'évaluation)

**Ce que c'est :** Un LLM de Google, 2 milliards de paramètres, tournant en local via Ollama.

**Ce qu'il fait ici :** Deux rôles distincts :

1. **Génération** — Reçoit le prompt (question + 5 passages + instructions) et génère la réponse en français, uniquement à partir du contexte fourni. Paramétré avec `temperature=0` pour des réponses déterministes, et `num_ctx=4096` pour le contexte.

2. **Juge d'évaluation** — Dans `evaluate.py`, le même LLM évalue la qualité des réponses en répondant à des prompts d'évaluation. Il retourne un score entre 0.0 et 1.0 pour chaque métrique. C'est le même principe que RAGAS, implémenté sans dépendance externe.

**Comparaison :**

| Modèle | Taille | Qualité | Vitesse |
|---|---|---|---|
| gemma2:2b | 1.6 GB | Correcte | Très rapide |
| llama3.1:8b | 4.7 GB | Bonne | Moyen |
| gemma2:9b | 5.5 GB | Très bonne | Lent |

`gemma2:2b` est suffisant pour extraire et reformuler des informations depuis un contexte fourni. Pour des questions de raisonnement complexe, `llama3.1:8b` donne de meilleurs résultats.

---

## 7. ChromaDB

**Ce que c'est :** Une base de données spécialisée dans le stockage et la recherche de vecteurs.

**Ce qu'il fait ici :** Stocke les 1024 nombres de chaque chunk dans `vector_db/chroma.sqlite3`. Quand tu poses une question, il calcule les 20 vecteurs les plus proches du vecteur de ta question (similarité cosinus).

**La différence avec une base classique :**
- Base classique : cherche "Banach" → trouve les lignes qui contiennent exactement "Banach"
- ChromaDB : cherche "espace complet" → trouve les passages sur "Banach", "Cauchy", "convergence"

**L'index BM25 est construit séparément** depuis les textes stockés dans ChromaDB (`vector_db._collection.get(include=["documents", "metadatas"])`), une seule fois par session.

---

## 8. LangChain

**Ce que c'est :** Une librairie Python qui sert de colle entre tous les composants.

**Ce qu'il fournit ici :**

| Classe | Usage |
|---|---|
| `MarkdownHeaderTextSplitter` | Découpe le texte sur les titres ## et ###, en conservant le contexte hiérarchique |
| `RecursiveCharacterTextSplitter` | Découpe les sections longues en chunks de 1000 caractères (overlap 200) |
| `OllamaEmbeddings` | Appelle bge-m3 via Ollama pour calculer les embeddings |
| `Chroma` | Gère la base vectorielle (écriture depuis ingest.py, lecture depuis ask.py) |
| `OllamaLLM` | Appelle gemma2:2b via Ollama pour la génération et l'évaluation |
| `TextLoader` | Charge les fichiers .txt |
| `Docx2txtLoader` | Charge les fichiers .docx |

---

## 9. CrossEncoder mmarco-mMiniLMv2 (re-ranker)

**Ce que c'est :** Un modèle de la bibliothèque `sentence-transformers` (~471 Mo, téléchargé automatiquement depuis HuggingFace au premier lancement).

**Nom complet :** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` — modèle multilingue entraîné sur des paires (question, passage) pour estimer leur pertinence réelle.

**Ce qu'il fait ici :** Re-classe les 10 candidats issus de la fusion RRF. Il reçoit des paires `(question, chunk)` et prédit un score de pertinence pour chacune :

```python
pairs = [("Quels sont les cours obligatoires ?", "Les quatre cours communs sont..."),
         ("Quels sont les cours obligatoires ?", "La bibliothèque est ouverte..."), ...]
scores = reranker.predict(pairs)
# → [8.4, 0.2, ...]
```

**Pourquoi c'est plus précis qu'un embedding ?**

Un embedding encode question et chunk **séparément** — il mesure leur proximité dans l'espace vectoriel mais sans voir les deux ensembles. Le CrossEncoder lit la question et le chunk **ensemble** dans un seul passage, ce qui lui permet de comprendre des relations subtiles ("ce passage répond-il vraiment à cette question ?").

**Où il est chargé :** Une seule fois au démarrage du programme, en variable globale dans `ask.py` :
```python
reranker = CrossEncoder(RERANK_MODEL)
```

---

## 10. Streamlit (interface web)

**Ce que c'est :** Un framework Python pour créer des interfaces web interactives sans écrire de HTML/CSS.

**Ce qu'il fait ici :** Fournit l'interface principale dans `app.py` :
- Un champ texte pour saisir la question
- Un bouton "Envoyer" qui déclenche le pipeline RAG
- Un spinner de chargement pendant le traitement
- L'affichage de la réponse générée

**Lancement :**
```powershell
python -m streamlit run src/app.py --server.headless true --server.fileWatcherType none
```

- `--server.headless true` : nécessaire avec Python 3.14 (comportement différent de Streamlit)
- `--server.fileWatcherType none` : supprime les warnings `torchvision`

L'interface est accessible sur **http://localhost:8501**.
