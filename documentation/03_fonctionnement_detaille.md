# 03 — Fonctionnement détaillé

## Le flux complet

```
PHASE 1 : INGESTION (une seule fois, ou après ajout de documents)
──────────────────────────────────────────────────────────────────
Fichiers PDF/TXT/DOCX  (documents/)
        │
        ▼ Pour les PDFs :
   Extraction page par page en Markdown  (pymupdf4llm, page_chunks=True)
   → chaque page = un Document indépendant avec métadonnée "page"
   → réparation des encodages cassés  (ftfy.fix_text)
   → chevauchement inter-pages : début de la page suivante (300 chars)
     ajouté à la fin de chaque page — évite de couper listes/sections
        │
        ▼ Étape 0 — Filtre des tables des matières
   _is_toc_page  (ratio lignes avec "..." ou ". . ." > 30 %)
   → les pages de TdM ne contiennent que des numéros de pages
     et polluent la recherche — elles sont ignorées
        │
        ▼ Étape 1 — Découpage par titres
   MarkdownHeaderTextSplitter  (# → h1, ## → h2, ### → h3)
   → strip_headers=False : les titres restent dans le texte des chunks
   → chaque chunk sait dans quelle section il se trouve (métadonnée h1/h2/h3)
        │
        ▼ Étape 2 — Fusion des micro-chunks
   _merge_small_chunks  (MIN_CHUNK_SIZE = 400 caractères)
   → tout chunk < 400 caractères est fusionné avec son voisin
   → évite les fragments trop courts (lignes de calendrier, etc.)
        │
        ▼ Étape 3 — Découpage par taille
   RecursiveCharacterTextSplitter  (chunk_size=1000, overlap=200)
   → séparateurs dans l'ordre : \n## > \n### > \n\n > \n| > \n > espace
   → \n| protège les lignes de tableaux Markdown contre la coupure
        │
        ▼ Étape 4 — Filtre des chunks trop courts
   → tout chunk dont le contenu utile < 30 caractères est écarté
     (numéros de page isolés, symboles seuls)
        │
        ▼ Étape 5 — Contextual retrieval (ajout d'un préfixe LLM)
   _contextualize_chunks  (gemma4:12b via Ollama)
   → chaque chunk est préfixé d'une ligne de contexte :
     [source | p.X | h1 > h2 > h3 | mots-clés LLM]
   → le LLM ne génère que les mots-clés (3-6 tokens, pas de résumé)
   → les informations structurelles (source, page, section) sont
     ajoutées sans LLM pour éviter les hallucinations
   → checkpoint pickle après chaque page (reprise sur crash)
        │
        ▼
   Transformation en vecteurs de 1024 nombres  (bge-m3 via Ollama)
   → envoi par lots de 50 chunks pour éviter les timeouts Ollama
        │
        ▼
   Sauvegarde  (ChromaDB → C:/vector_db_aquila/chroma.sqlite3)


PHASE 2 : QUESTION/RÉPONSE AGENTIQUE (à chaque question)
──────────────────────────────────────────────────────────────────
Question de l'utilisateur
        │
        ▼ [LangGraph] identify_sources
   Le LLM choisit quel(s) fichier(s) de documents/ concernent la question.
   → Si la question mentionne un établissement précis : filtre sur ce fichier
   → Si la question compare plusieurs établissements : cherche partout
   → Si la réponse ne correspond à aucun fichier : cherche partout
        │
        ▼ [LangGraph] retrieve_node
   → HyDE : génère une réponse fictive pour la recherche sémantique
   ┌─────────────────────────────────────────┐
   │                                         │
   ▼                                         ▼
Recherche SÉMANTIQUE MMR              Recherche LEXICALE (BM25)
→ réponse fictive vectorisée (bge-m3) → question normalisée (sans accents)
→ MMR : 20 chunks pertinents + diversifiés → 20 chunks avec mots exacts
   (parmi 80 candidats, lambda=0.5)    BM25 normalisé : accents, points
K_RETRIEVE = 20                        médians, tokens 2+ chars
   │                                         │
   └──────────────────┬──────────────────────┘
                      ▼
             Fusion RRF (Reciprocal Rank Fusion)
             score(chunk) = 1/(60 + rang_sémantique) + 1/(60 + rang_BM25)
             → filtre diversité : max 3 chunks par source (si multi-sources)
             → 10 candidats sélectionnés  (K_RERANK = 10)
                      │
                      ▼
             Re-ranking CrossEncoder (BAAI/bge-reranker-v2-m3)
             → lit chaque paire (question, chunk) ENSEMBLE
             → score logit : > 0 = pertinent, < 0 = hors-sujet
             → chunks sous RERANK_THRESHOLD = 0.0 écartés
             → 5 meilleurs gardés  (K_FINAL = 5)
        │
        ▼ [LangGraph] grade_documents
   Le LLM évalue si les chunks accumulés sont suffisants :
   → "OUI" → passe directement à generate
   → "NON — [ce qui manque]" → passe à rewrite_query si tentatives < MAX_ATTEMPTS=2
        │
        ├── OUI ou max tentatives atteint ──────────┐
        │                                            ▼
        │                                  [LangGraph] generate_node
        │                                  Prompt = question + 5 chunks + instructions
        │                                  gemma4:12b, temperature=0, num_ctx=4096
        │                                  → Réponse (Open WebUI ou terminal)
        │
        └── NON (< 2 tentatives)
                │
                ▼
        [LangGraph] rewrite_query
        Le LLM reformule la requête sur ce qui manque précisément
                │
                └──→ retrieve_node (nouvelle tentative)


PHASE 3 : ÉVALUATION (à la demande)
──────────────────────────────────────────────────────────────────
python src/run_agentic_all.py
→ charge data/questions.json (40 questions avec réponses de référence)
→ pour chaque question : lance ask_question_agentic() avec verbose=True
→ capture les logs de l'agent (_Tee : écrit sur console ET buffer)
→ sauvegarde après chaque question dans data/agentic_results.json (crash-safe)

python src/evaluate_ragas.py
→ charge data/agentic_results.json + data/questions.json
→ construit un EvaluationDataset RAGAS (SingleTurnSample par question)
→ initialise gemma4:12b et bge-m3 via API Ollama (OpenAI-compatible)
→ score chaque sample individuellement avec 5 métriques (async)
→ exporte data/ragas_evaluation.csv
→ affiche le récapitulatif par question et par niveau
```

---

## Phase 1 — Ingestion en détail

### Extraction Markdown (pymupdf4llm + ftfy)

`pymupdf4llm` est appelé avec l'option `page_chunks=True` : chaque page du PDF est extraite séparément. Après extraction, `ftfy` répare les accents mal encodés :

```python
# Chaque page du PDF devient un Document LangChain indépendant
pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
for page in pages:
    text = ftfy.fix_text(page["text"])   # répare les encodages cassés (a → à, ´e → é)
    if not text.strip():  # ignore les pages vides (couvertures, pages blanches)
        continue
    page_num = page["metadata"].get("page_number", 0)
    documents.append(Document(
        page_content=text,
        metadata={"source": pdf_path.name, "page": page_num + 1},  # 1-indexé
    ))
```

**Pourquoi page par page ?** Un tableau peut couvrir toute une page. Si on extrait le PDF d'un seul bloc puis qu'on découpe par taille, le tableau sera scindé au milieu. Avec `page_chunks=True`, le tableau reste dans un seul document avant le découpage.

### Chevauchement inter-pages (PAGE_OVERLAP_CHARS = 300)

Une liste ou une section peut commencer en bas d'une page et se terminer en haut de la suivante. Sans chevauchement, le chunk du bas de la page ne contient que le début de la liste, et le chunk du haut ne contient que la suite — aucun des deux ne contient l'information complète.

```python
# Les 300 premiers caractères de chaque page sont ajoutés à la fin de la précédente
for i in range(len(documents) - 1):
    next_start = documents[i + 1].page_content[:PAGE_OVERLAP_CHARS]
    documents[i].page_content += "\n\n" + next_start
```

### Détection des tables des matières (_is_toc_page)

Les pages de TdM listent uniquement des numéros de pages (ex: `1.1  Objectifs . . . . . . . . . 7`). Si plus de 30 % des lignes contiennent `...` ou `. . .`, la page est ignorée :

```python
# Proportion de lignes avec des points de suspension > seuil → c'est une TdM
dot_lines = sum(1 for l in lines if ". . ." in l or "..." in l)
return dot_lines / len(lines) > TOC_DOT_RATIO  # TOC_DOT_RATIO = 0.3
```

### Étape 1 — Découpage par titres (MarkdownHeaderTextSplitter)

```python
# Coupe à chaque titre Markdown — chaque section garde son chemin de titres
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=False,  # conserve les titres dans le texte du chunk
)
```

Ce splitter coupe le texte à chaque titre Markdown. Chaque section produit un chunk qui contient :
- Le titre lui-même (parce que `strip_headers=False`)
- Le texte sous ce titre jusqu'au prochain titre de niveau égal ou supérieur
- Les métadonnées `h1`, `h2`, `h3` — le chemin hiérarchique de la section

**Pourquoi garder les titres dans le texte ?** Deux sections de cours différents peuvent avoir le même intitulé — par exemple deux sections "Organisation" dans deux brochures. En conservant `## Organisation du DMA` dans le texte du chunk, l'embedding distingue les deux lors de la recherche.

### Étape 2 — Fusion des micro-chunks (_merge_small_chunks)

Certains chunks sont très courts après le découpage par titres — par exemple une ligne de calendrier sous son propre titre :

```
## Fin des cours
Vendredi 17 janvier 2025
```

Ce chunk fait ~50 caractères. Un chunk aussi court est trop pauvre en mots pour être bien classé. La fonction `_merge_small_chunks` parcourt tous les chunks et fusionne tout chunk de moins de 400 caractères avec son voisin suivant :

```python
MIN_CHUNK_SIZE = 400  # seuil de fusion en caractères

buffer = None
for chunk in chunks:
    buffer = chunk if buffer is None else Document(
        page_content=buffer.page_content + "\n\n" + chunk.page_content,
        metadata=buffer.metadata,   # garde les métadonnées du premier chunk
    )
    if len(buffer.page_content) >= MIN_CHUNK_SIZE:
        merged.append(buffer)
        buffer = None
# reliquat final trop court → rattaché au dernier chunk déjà validé
```

### Étape 3 — Découpage par taille (RecursiveCharacterTextSplitter)

Certaines sections, après fusion, dépassent 1000 caractères. Elles sont redécoupées :

```python
# Coupe en chunks de 1000 chars max, avec 200 chars de chevauchement
size_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n## ", "\n### ", "\n\n", "\n|", "\n", " ", ""],
)
```

**Ordre de priorité des séparateurs :**
1. `\n## ` et `\n### ` — coupe en priorité entre sections Markdown
2. `\n\n` — coupe entre paragraphes
3. `\n|` — coupe avant une ligne de tableau, **protège les tableaux Markdown** : une ligne `| col1 | col2 |` ne sera jamais coupée au milieu
4. `\n` — coupe entre lignes
5. espace puis caractère — en dernier recours

**Le chevauchement de 200 caractères** évite de couper une idée en deux :

```
Chunk 1 : "...un espace vectoriel normé est dit complet si toute suite de
           Cauchy converge. On appelle un tel espace un espace de [FIN]"

Chunk 2 : "[DÉBUT] espace de Banach. Les espaces de Banach jouent un rôle
           central en analyse fonctionnelle..."
```

### Étape 5 — Contextual retrieval (_contextualize_chunks)

L'embedding et BM25 ne lisent que le texte du chunk — pas ses métadonnées. Un chunk isolé ne dit pas de lui-même dans quel établissement ou quelle section il se trouve.

La contextualisation ajoute en tête de chaque chunk une ligne de contexte construite en deux parties :

**Partie déterministe (sans LLM, sans hallucination) :**
- Nom du fichier source (`Brochure-2024-2025.pdf`)
- Numéro de page
- Chemin de titres capturé par MarkdownHeaderTextSplitter (`DMA > Organisation > Cours`)

**Partie LLM :**
- 3 à 6 mots-clés sur le sujet précis du chunk (ex: `cours obligatoires, ECTS, L3`)
- Le LLM ne voit que le chunk, pas le document entier — il ne peut donc générer que des mots-clés sur le contenu, pas sur le contexte global. Les informations structurelles sont ajoutées sans lui.

Résultat :
```
[Brochure-2024-2025.pdf | p.12 | DMA > Organisation > Cours | cours obligatoires, ECTS, L3]

## Cours communs de L3
Les quatre cours communs obligatoires sont...
```

**Checkpoint pickle :** La contextualisation appelle le LLM pour chaque chunk (~700 appels pour deux brochures). En cas de crash llama-server en cours de route, la progression est sauvegardée après chaque page. Au prochain lancement d'`ingest.py`, la contextualisation repart du dernier point de sauvegarde.

### Stockage dans ChromaDB

Les chunks sont envoyés à ChromaDB par **lots de 50** pour éviter les timeouts Ollama :

```python
batch_size = 50
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    if db is None:
        # Premier lot : crée la base Chroma
        db = Chroma.from_documents(batch, embeddings, persist_directory=...)
    else:
        # Lots suivants : ajoute à la base existante
        db.add_documents(batch)
```

Chaque chunk est transformé en vecteur de 1024 nombres par bge-m3 (~1-2 secondes par chunk). Pour ~700 chunks : **15 à 25 minutes**. Sur GPU Colab : bien plus rapide.

---

## Phase 2 — Question/Réponse en détail

### Étape 0 : Redémarrage préventif d'Ollama

Le KV-cache de llama-server s'accumule au fil des questions et dégrade la latence (101s → 1211s observés sur 9 questions sans redémarrage). Ollama est redémarré de façon préventive toutes les 2 questions :

```python
QUESTION_RESTART_INTERVAL = 2  # fréquence de redémarrage préventif

def _maybe_restart_ollama(verbose: bool = True):
    global _question_count
    _question_count += 1
    if _question_count % QUESTION_RESTART_INTERVAL == 0:
        _restart_ollama()  # tue et relance ollama serve
```

### Étape 1 : HyDE (Hypothetical Document Embeddings)

La recherche sémantique calcule la similarité cosinus entre le vecteur de la question et les vecteurs des chunks. Problème : une question (`"Quels sont les cours ?"`) et une réponse (`"Les cours sont..."`) ne sont pas stylistiquement similaires, même si elles parlent du même sujet.

HyDE génère une réponse fictive stylistiquement proche des chunks avant de faire la recherche :

```python
# Génère une réponse fictive pour améliorer la recherche sémantique
HYDE_PROMPT = """Tu es un extrait de brochure universitaire. Réponds à cette
question en 2-3 phrases courtes et factuelles, comme si tu étais le passage
d'une brochure qui y répond directement : {question}"""

hyde_query = _invoke_with_retry(HYDE_PROMPT.format(question=question)).strip()
# → "Les cours obligatoires de L3 comprennent Algèbre 1, Analyse complexe..."
```

BM25 (étape 2) continue d'utiliser la **question originale** pour les correspondances exactes.

### Étape 2 : Recherche sémantique (MMR)

La question fictive (HyDE) est vectorisée par bge-m3. ChromaDB retourne 20 chunks avec MMR (Maximal Marginal Relevance) :

```python
# MMR : pertinents ET diversifiés — évite de retourner 20 extraits du même paragraphe
mmr_docs = vector_db.max_marginal_relevance_search(
    hyde_query,
    k=K_RETRIEVE,          # K_RETRIEVE = 20 — nombre de résultats voulus
    fetch_k=MMR_FETCH_K,   # MMR_FETCH_K = 80 — candidats initiaux examinés
    lambda_mult=MMR_LAMBDA, # MMR_LAMBDA = 0.5 — équilibre pertinence/diversité
    filter=chroma_filter,   # filtre sur la source si identify_sources l'a précisé
)
```

MMR pioche parmi 80 candidats et en sélectionne 20 qui maximisent à la fois la pertinence (proches de la question) et la diversité (éloignés les uns des autres). Cela évite que les 20 résultats soient 20 extraits du même paragraphe.

### Étape 3 : Recherche BM25 (lexicale)

L'index BM25 est construit **une seule fois par session** depuis tous les chunks stockés dans ChromaDB. La question originale est normalisée puis scorée :

```python
# Normalise la question avec la même fonction que lors de l'indexation
bm25_scores = bm25.get_scores(_tokenize(question))
top_bm25 = sorted(candidate_indices, key=lambda i: bm25_scores[i], reverse=True)[:K_RETRIEVE]
```

### Étape 4 : Fusion RRF (Reciprocal Rank Fusion)

Les deux listes de 20 résultats sont fusionnées. La formule RRF :

```
score(chunk) = 1/(60 + rang_sémantique) + 1/(60 + rang_BM25)
```

Un chunk bien classé dans les deux listes obtient un score élevé. La constante 60 (`RRF_K`) est la valeur standard de la littérature — elle empêche les premiers rangs de dominer trop fortement.

**Pourquoi RRF plutôt qu'une somme pondérée ?** RRF est indépendant des valeurs brutes des scores (qui varient selon les modèles) — il ne regarde que les positions dans le classement.

**Filtre de diversité :** maximum 3 chunks par document source quand on cherche dans plusieurs sources — évite que les 10 slots soient saturés par des extraits du même PDF. Ce plafond est désactivé (porté à K_RERANK=10) quand `identify_sources` a restreint la recherche à une seule source.

### Étape 5 : Re-ranking (CrossEncoder)

Le re-ranker reçoit les 10 candidats RRF. Pour chaque chunk, il forme la paire `(question, chunk)` et la lit ensemble :

```
Paires envoyées au re-ranker :
("Quels sont les cours obligatoires ?", "Les quatre cours communs sont...")  → logit  8.4
("Quels sont les cours obligatoires ?", "La bibliothèque est ouverte...")    → logit -1.2
```

Les 5 chunks avec les scores les plus élevés sont gardés. Les chunks sous le seuil `RERANK_THRESHOLD = 0.0` sont écartés même s'ils font partie des 5 premiers.

### Étape 6 : Évaluation des chunks (grade_documents)

Le pipeline agentique ajoute une étape que le RAG classique n'a pas : demander au LLM si les chunks trouvés sont suffisants pour répondre :

```python
# Prompt envoyé au LLM avec les chunks accumulés
GRADE_PROMPT = """Ces extraits contiennent-ils l'information nécessaire ?
- Si oui, réponds uniquement : OUI
- Si non, réponds : NON — [explique en 1 phrase ce qui manque]"""

# Exemples de réponse NON :
# "NON — les extraits mentionnent le stage mais n'indiquent pas sa durée minimale"
```

Si insuffisant et si le nombre de tentatives est sous `MAX_ATTEMPTS=2`, le pipeline reformule la requête et relance un retrieval.

### Étape 7 : Génération

Les chunks finaux sont assemblés en contexte :

```python
# Chaque chunk est accompagné de sa source pour que le LLM puisse citer
context = "\n\n---\n\n".join(
    f"Source : {doc.metadata.get('source', '?')}\n{doc.page_content}"
    for doc in final_docs
)
```

Le template `prompts/rag_prompt.txt` est chargé, les variables `{question}` et `{context}` sont remplacées. Le LLM gemma4:12b génère la réponse avec `temperature=0` (réponses déterministes) et `num_ctx=4096`.

Si aucun chunk n'a été trouvé :
> *"Je ne trouve pas cette information dans les documents fournis."*

---

## Phase 3 — Évaluation en détail

### Étape 1 : run_agentic_all.py

Ce script passe les 40 questions au pipeline agentique de façon séquentielle et sauvegarde les résultats :

```python
# Structure d'un résultat dans data/agentic_results.json
{
    "id": "L1_ENS_001",
    "question": "Qui dirige le DMA en 2024-2025 ?",
    "reponse_attendue": "...",  # ground truth
    "reponse_llm": "...",        # réponse générée par l'agent
    "duree_secondes": 42.3,
    "logs": "[Agent] Source(s) ciblée(s) : Brochure-2024-2025.pdf\n...",
    "chunks": [
        {"source": "Brochure-2024-2025.pdf", "page": 3, "content": "..."}
    ]
}
```

**Reprise sur interruption :** Si le run est interrompu (Ctrl+C, crash), le fichier est lu au prochain lancement et seules les questions manquantes sont traitées.

### Étape 2 : evaluate_ragas.py

Le script charge `agentic_results.json`, construit un `EvaluationDataset` RAGAS et score chaque échantillon individuellement :

```python
# Chaque question devient un SingleTurnSample RAGAS
sample = SingleTurnSample(
    user_input=r["question"],          # question posée
    response=r["reponse_llm"],         # réponse générée
    retrieved_contexts=contexts,        # liste de chunks (strings)
    reference=r["reponse_attendue"],   # ground truth
)
```

RAGAS utilise gemma4:12b et bge-m3 via l'API OpenAI-compatible d'Ollama (`http://localhost:11434/v1`). Aucun appel à l'extérieur.

### Le dataset d'évaluation

40 questions réparties en 3 niveaux :

| Niveau | Type | Exemple |
|---|---|---|
| 1 | Factuel simple | "Qui dirige le DMA en 2024-2025 ?" |
| 2 | Synthèse intra-document | "Comment fonctionne le système de tutorat ?" |
| 3 | Comparaison multi-documents | "Comparez les stages ENS et Sorbonne" |

Chaque question a un `id` structuré (`L1_ENS_001` = Niveau 1, source ENS, séquence 001), une `reponse_attendue` (ground truth), et des métadonnées (`type_source`, `difficulte_rag`, `pages`, `section`).

### Les 5 métriques RAGAS

| Métrique | Ce qu'elle mesure | Ground truth ? |
|---|---|---|
| **Faithfulness** | La réponse invente-t-elle des choses absentes des chunks ? | Non |
| **AnswerRelevancy** | La réponse répond-elle à la question posée ? | Non |
| **ContextPrecision** | Les chunks récupérés sont-ils pertinents pour cette question ? | Non |
| **ContextRecall** | Les chunks couvrent-ils tout ce que contient la réponse de référence ? | Oui |
| **AnswerCorrectness** | La réponse est-elle correcte par rapport à la référence ? | Oui |

Toutes les métriques retournent un score entre 0.0 et 1.0.

### Interpréter les scores

```
Faithfulness faible     → le LLM hallucine, il invente des infos non présentes dans les chunks
AnswerRelevancy faible  → le LLM répond à côté, il ne comprend pas bien la question
ContextPrecision faible → le retrieval ramène des chunks hors sujet
ContextRecall faible    → le retrieval rate des informations importantes
AnswerCorrectness faible → la réponse est incorrecte par rapport aux documents
```
