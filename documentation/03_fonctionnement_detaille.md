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
        │
        ▼ Étape 1 — Découpage par titres
   MarkdownHeaderTextSplitter  (# → h1, ## → h2, ### → h3)
   → strip_headers=False : les titres restent dans le texte des chunks
   → chaque chunk sait dans quelle section il se trouve
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
        ▼
   Transformation en vecteurs de 1024 nombres  (bge-m3 via Ollama)
   → envoi par lots de 50 chunks pour éviter les timeouts Ollama
        │
        ▼
   Sauvegarde  (ChromaDB → vector_db/chroma.sqlite3)


PHASE 2 : QUESTION/RÉPONSE (à chaque question)
──────────────────────────────────────────────────────────────────
Question de l'utilisateur
        │
        ├─────────────────────────────────────────┐
        ▼                                         ▼
   Recherche SÉMANTIQUE                  Recherche LEXICALE (BM25)
   → question vectorisée par bge-m3       → question tokenisée en mots minuscules
   → 20 chunks les plus proches (cosinus) → 20 chunks avec les mots exacts
   K_RETRIEVE = 20                        K_RETRIEVE = 20
        │                                         │
        └──────────────────┬──────────────────────┘
                           ▼
                  Fusion RRF (Reciprocal Rank Fusion)
                  → score(chunk) = 1/(60 + rang_sémantique) + 1/(60 + rang_BM25)
                  → filtre diversité : max 3 chunks par document source
                  → 10 candidats sélectionnés  (K_RERANK = 10)
                           │
                           ▼
                  Re-ranking CrossEncoder
                  → lit chaque paire (question, chunk) ensemble
                  → note la pertinence réelle de chaque chunk
                  → 5 meilleurs gardés  (K_FINAL = 5)
                           │
                           ▼
                  Construction du prompt
                  (question + 5 chunks avec leur source + instructions strictes)
                           │
                           ▼
                  Génération  (gemma2:2b via Ollama, temperature=0, num_ctx=4096)
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
            Terminal           Interface Streamlit
            (ask.py)           (app.py → localhost:8501)


PHASE 3 : ÉVALUATION (à la demande)
──────────────────────────────────────────────────────────────────
python src/evaluate.py
→ charge data/questions.json (40 questions avec réponses de référence)
→ pour chaque question : lance le pipeline RAG complet (verbose=False)
→ envoie 5 prompts d'évaluation au LLM (gemma2:2b joue le rôle de juge)
→ extrait le score numérique (0.0 à 1.0) de chaque réponse du juge
→ affiche un tableau de résultats par question et par niveau
→ sauvegarde dans data/results.json après chaque question (Ctrl+C sûr)
```

---

## Phase 1 — Ingestion en détail

### Extraction Markdown (pymupdf4llm + ftfy)

`pymupdf4llm` est appelé avec l'option `page_chunks=True`, ce qui signifie que chaque page du PDF est extraite séparément :

```python
pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
for page in pages:
    text = ftfy.fix_text(page["text"])   # réparation encodage
    page_num = page["metadata"]["page_number"]
    documents.append(Document(
        page_content=text,
        metadata={"source": pdf_path.name, "page": page_num + 1}
    ))
```

**Pourquoi page par page ?** Un tableau peut couvrir toute une page. Si on extrait le PDF d'un seul bloc, puis qu'on découpe par taille, le tableau sera scindé au milieu. Avec `page_chunks=True`, le tableau reste dans un seul document avant le découpage.

**ftfy** répare les accents cassés (`a → à`, `´e → é`) qui apparaissent fréquemment dans les PDFs mal encodés. Sans cette étape, BM25 ne retrouverait pas les mots accentués.

**Pages vides** (couvertures, pages blanches) : les pages dont le texte est vide après nettoyage sont ignorées (`if not text.strip(): continue`).

Cette étape s'applique uniquement aux PDFs. Les `.txt` sont chargés par `TextLoader`, les `.docx` par `Docx2txtLoader` — sans passage par page.

---

### Étape 1 — Découpage par titres (MarkdownHeaderTextSplitter)

```python
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=False,
)
```

Ce splitter coupe le texte à chaque titre Markdown (`#`, `##`, `###`). Chaque section produit un chunk qui contient :
- Le titre lui-même (parce que `strip_headers=False`)
- Le texte sous ce titre jusqu'au prochain titre de niveau égal ou supérieur

**Pourquoi garder les titres dans le texte (`strip_headers=False`) ?** Deux sections de cours différents peuvent avoir des contenus similaires — par exemple deux sections "Organisation" dans deux brochures différentes. En conservant le titre `## Organisation du DMA` dans le texte du chunk, le modèle d'embedding comprend le contexte et les distingue lors de la recherche.

**Après ce premier découpage**, les métadonnées `source` et `page` du document d'origine sont recopiées dans chaque chunk produit (`hc.metadata.update(doc.metadata)`).

---

### Étape 2 — Fusion des micro-chunks (_merge_small_chunks)

Après le découpage par titres, certains chunks sont très courts — par exemple une simple ligne de calendrier isolée sous son propre titre :

```
## Fin des cours
Vendredi 17 janvier 2025
```

Ce chunk fait ~50 caractères. Un chunk aussi court est trop pauvre en mots pour être bien classé par BM25 ou par la recherche sémantique — il se fait systématiquement éclipser par des chunks plus longs et plus riches en mots-clés, même issus d'un autre document.

La fonction `_merge_small_chunks` parcourt tous les chunks et fusionne tout chunk de moins de 400 caractères avec son voisin suivant :

```python
MIN_CHUNK_SIZE = 400  # en caractères

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

---

### Étape 3 — Découpage par taille (RecursiveCharacterTextSplitter)

Certaines sections, après fusion, dépassent encore 1000 caractères. Elles sont alors redécoupées :

```python
size_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n## ", "\n### ", "\n\n", "\n|", "\n", " ", ""],
)
```

**chunk_size = 1000** — assez grand pour contenir une définition mathématique complète.

**chunk_overlap = 200** — les chunks se chevauchent pour éviter de couper une idée en deux :

```
Chunk 1 : "...un espace vectoriel normé est dit complet si toute suite de
           Cauchy converge. On appelle un tel espace un espace de [FIN]"

Chunk 2 : "[DÉBUT] espace de Banach. Les espaces de Banach jouent un rôle
           central en analyse fonctionnelle..."
```

**Ordre de priorité des séparateurs :**
1. `\n## ` et `\n### ` — coupe en priorité entre sections Markdown
2. `\n\n` — coupe entre paragraphes
3. `\n|` — coupe avant une ligne de tableau, ce qui **protège les tableaux Markdown** : une ligne `| col1 | col2 |` ne sera jamais coupée au milieu
4. `\n` — coupe entre lignes
5. espace puis caractère — en dernier recours

---

### Stockage dans ChromaDB

Les chunks sont envoyés à ChromaDB par **lots de 50** pour éviter les timeouts Ollama sur de grandes bases :

```python
batch_size = 50
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    if db is None:
        db = Chroma.from_documents(batch, embeddings, persist_directory=...)
    else:
        db.add_documents(batch)
```

Chaque chunk est transformé en vecteur de 1024 nombres par bge-m3 (~1-2 secondes par chunk). Pour ~700 chunks : **15 à 25 minutes**.

---

## Phase 2 — Question/Réponse en détail

### Étape 1 : Recherche sémantique

La question est vectorisée par bge-m3. ChromaDB calcule la similarité cosinus entre ce vecteur et les ~700 vecteurs stockés. Les 20 plus proches (`K_RETRIEVE = 20`) sont retournés avec leur score.

**Avantage :** trouve des passages qui parlent du même concept même avec des mots différents.
**Limite :** peut rater des passages contenant des termes exacts spécifiques (noms propres, codes, sigles).

### Étape 2 : Recherche BM25

L'index BM25 est construit **une seule fois par session** depuis tous les chunks stockés dans ChromaDB. La question est tokenisée en mots minuscules. BM25 score chaque chunk selon la fréquence et la rareté des mots. Les 20 meilleurs indices sont retournés.

```python
bm25_scores = bm25.get_scores(question.lower().split())
top_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:K_RETRIEVE]
```

**Avantage :** trouve les passages contenant exactement les mots de la question.
**Limite :** ne comprend pas le sens, peut rater des synonymes.

### Étape 3 : Fusion RRF (Reciprocal Rank Fusion)

Les deux listes de 20 résultats sont fusionnées. La formule RRF :

```
score(chunk) = 1/(60 + rang_sémantique) + 1/(60 + rang_BM25)
```

Un chunk bien classé dans les deux listes obtient un score élevé. Un chunk excellent dans une seule liste peut quand même bien se classer. La constante 60 (`RRF_K`) est la valeur standard de la littérature — elle empêche les premiers rangs de dominer trop fortement.

**Pourquoi RRF plutôt qu'une somme pondérée ?** RRF est indépendant des valeurs brutes des scores (qui varient selon les modèles) — il ne regarde que les positions dans le classement.

**Filtre de diversité :** maximum 3 chunks par document source pour éviter que les 10 slots soient saturés par des extraits du même PDF. Ce filtre est appliqué après le tri par score RRF :

```python
source_count: dict[str, int] = {}
for key, score in ranked:
    source = doc_map[key].metadata.get("source", "?")
    if source_count.get(source, 0) < 3:
        source_count[source] += 1
        top.append((key, score))
    if len(top) == n:   # n = K_RERANK = 10
        break
```

### Étape 4 : Re-ranking (CrossEncoder)

Le re-ranker reçoit les 10 candidats RRF. Pour chaque chunk, il forme la paire `(question, chunk)` et la lit **ensemble** — contrairement aux embeddings qui calculent question et chunk séparément.

```
Paires envoyées au re-ranker :
("Quels sont les cours obligatoires ?", "Les quatre cours communs sont...")  → score 8.4
("Quels sont les cours obligatoires ?", "La bibliothèque est ouverte...")    → score 0.2
...
```

Les 5 chunks avec les scores les plus élevés sont gardés pour le LLM.

**Pourquoi ajouter un re-ranker ?** Un chunk peut être bien classé par RRF mais ne pas vraiment répondre à la question. Le CrossEncoder comprend mieux la pertinence réelle grâce à la lecture conjointe question + chunk.

### Étape 5 : Génération

Les 5 chunks sont assemblés en un contexte avec leur source :

```python
context = "\n\n---\n\n".join(
    f"Source : {doc.metadata.get('source', '?')}\n{doc.page_content}"
    for doc in final_docs
)
```

Le template `prompts/rag_prompt.txt` est chargé et les variables `{question}` et `{context}` sont remplacées. Le LLM gemma2:2b génère la réponse avec `temperature=0` (réponses déterministes, pas d'aléatoire) et `num_ctx=4096`.

Si aucun chunk n'a été trouvé (liste vide après le pipeline), la réponse retournée est :
> *"Je ne trouve pas cette information dans les documents fournis."*

---

## Phase 3 — Évaluation en détail

### Le dataset synthétique

40 questions ont été générées par un LLM à partir des brochures ENS et Sorbonne, réparties en 3 niveaux :

| Niveau | Type | Exemple |
|---|---|---|
| 1 | Factuel simple | "Qui dirige le DMA en 2024-2025 ?" |
| 2 | Synthèse intra-document | "Comment fonctionne le système de tutorat ?" |
| 3 | Comparaison multi-documents | "Comparez la philosophie ENS vs Sorbonne" |

Chaque question a une `reponse_attendue` (ground truth) qui sert à calculer les métriques avec référence.

### Le système d'évaluation LLM-judge

Il n'y a **aucun package d'évaluation externe** dans ce projet. Toutes les métriques sont calculées via des appels directs au LLM (gemma2:2b), qui joue le rôle de juge. C'est le même principe que RAGAS, implémenté directement sans dépendance externe.

Pour chaque métrique, un prompt spécifique est envoyé au LLM. Le LLM est instruit de répondre **uniquement par un nombre entre 0.0 et 1.0**. La fonction `_score()` extrait ce nombre :

```python
def _score(prompt: str) -> float:
    raw = llm.invoke(prompt).strip()
    for token in raw.replace(",", ".").split():   # remplace virgules françaises
        try:
            return max(0.0, min(1.0, float(token)))  # clamp entre 0 et 1
        except ValueError:
            continue
    return 0.0   # si le LLM ne donne pas de nombre valide
```

### Les 5 métriques

Toutes calculées par appels LLM :

| Métrique | Question posée au juge | Ground truth ? |
|---|---|---|
| **Faithfulness** | La réponse invente-t-elle des choses non présentes dans les chunks ? | Non |
| **Answer Relevancy** | La réponse répond-elle à la question posée ? | Non |
| **Context Quality** | Les chunks récupérés sont-ils pertinents pour cette question ? | Non |
| **Context Recall** | Les chunks couvrent-ils tout ce que contient la réponse de référence ? | Oui |
| **Answer Correctness** | La réponse est-elle factuellement correcte par rapport à la référence ? | Oui |

Par question : **5 appels LLM** (1 par métrique). Pour 40 questions : **200 appels LLM** au total → durée estimée 30 à 60 minutes.

### Traitement incrémental

`evaluate.py` charge les résultats existants de `data/results.json` et ne réévalue que les questions absentes :

```python
already_done = {r["id"] for r in results}
to_evaluate = [e for e in dataset if e["id"] not in already_done]
```

Le fichier est réécrit après chaque question réussie. Un Ctrl+C en cours d'évaluation ne perd pas les résultats déjà calculés.

### Interpréter les scores

```
Faithfulness faible  → le LLM hallucine, il invente des infos non présentes dans les chunks
Relevancy faible     → le LLM répond à côté, il ne comprend pas bien la question
Context Quality faible → le retrieval ramène des chunks hors sujet
Context Recall faible  → le retrieval rate des informations importantes
Correctness faible     → la réponse est incorrecte par rapport aux documents
```
