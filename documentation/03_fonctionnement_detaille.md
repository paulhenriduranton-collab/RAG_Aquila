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
        ▼ Étape 1 — Concaténation des pages (_concat_pages)
   → les pages de table des matières sont exclues (_is_toc_page :
     ratio lignes avec "..." ou ". . ." > 30 %)
   → toutes les pages restantes d'un même document sont recollées en
     UN SEUL texte continu, avec un marqueur <!-- PAGE X --> au début
     de chaque page d'origine
   → c'est cette étape qui permet à une section thématique de chevaucher
     plusieurs pages — il n'y a plus de frontière de page dans le texte
        │
        ▼ Étape 2 — Pré-découpe par titres (_presplit_by_headers)
   → découpe le texte concaténé à chaque titre Markdown # ou ##
   → chaque bloc retient la page de son PREMIER marqueur <!-- PAGE X -->
     (_get_page_for_position) — c'est la métadonnée "page" finale du chunk,
     même si son contenu déborde sur la page suivante
   → les blocs < MIN_SECTION_SIZE_FOR_SPLIT (800 caractères) sont fusionnés
     avec le bloc suivant, pour ne pas envoyer de micro-section au LLM
        │
        ▼ Étape 3 — Split agentique (_agentic_split_section, LLM gemma4:12b)
   → chaque bloc thématique (≥ 800 caractères) est envoyé au LLM, qui
     insère des marqueurs ===SPLIT=== entre sous-sections distinctes
     (consigne stricte : jamais au milieu d'un tableau Markdown)
   → validation anti-hallucination : si moins de 60% des mots du bloc
     d'origine sont retrouvés dans la réponse du LLM, le découpage est
     rejeté et le bloc original est gardé intact (fallback)
   → un bloc qui ne contient qu'un seul sujet peut ressortir sans aucun
     marqueur (segment unique)
        │
        ▼ Étape 4 — Fallback déterministe (_fallback_split_large)
   RecursiveCharacterTextSplitter  (chunk_size=1500, overlap=0)
   → ne s'applique qu'aux segments encore > MAX_CHUNK_SIZE (1500 car.)
     après le split agentique — un filet de sécurité, pas le découpage
     principal
   → séparateurs dans l'ordre : \n## > \n### > \n\n > \n| > \n > espace
   → \n| protège les lignes de tableaux Markdown contre la coupure
        │
        ▼ Étape 5 — Filtre des chunks trop courts
   → tout chunk < MIN_CONTENT_SIZE (30 caractères) est écarté
     (numéros de page isolés, symboles seuls)
        │
        ▼ Étape 6 — chunk_index par page
   → chaque chunk reçoit un numéro d'ordre (0, 1, 2...) parmi les chunks
     qui partagent la même métadonnée "page" — utilisé par agent.py pour
     retrouver "le chunk suivant" en cas de troncature (_expand_truncated)
        │
        ▼ Étape 7 — Contextualisation déterministe (zéro LLM)
   → chemin de titres extrait par regex sur les #/##/### du chunk
     (_extract_section_path)
   → mots-clés = mots/bigrammes les plus fréquents du chunk après
     suppression des stopwords français (_extract_keywords_deterministic) —
     comptage de fréquence, aucun appel LLM, aucune hallucination possible
   → préfixe inséré en tête du chunk :
     [source | p.X | chemin de titres | mots-clés]
   → checkpoint pickle après chaque document traité (reprise sur crash)
        │
        ▼
   Transformation en vecteurs de 1024 nombres  (bge-m3 via Ollama)
   → envoi par lots de 50 chunks pour éviter les timeouts Ollama
        │
        ▼
   Sauvegarde  (ChromaDB)
   → en local : C:/vector_db_aquila (hors OneDrive)
   → sur Colab : /content/RAG_Aquila/vector_db (override notebook)


PHASE 2 : QUESTION/RÉPONSE AGENTIQUE (à chaque question)
──────────────────────────────────────────────────────────────────
Question de l'utilisateur
        │
        ▼ [LangGraph] identify_sources
   Le LLM (gemma4:12b) choisit quel(s) fichier(s) et classifie la difficulté.
   → SOURCES : fichier(s) concerné(s) ou TOUS
   → DIFFICULTE : 1 (factuel), 2 (synthèse), 3 (complexe/comparaison)
   → SOUS-REQUETES : décomposition si difficulté 3 (1 à 2 sous-requêtes)
   → Fallback : sources=None, difficulté=2 si le LLM ne suit pas le format
        │
        ▼ [LangGraph] retrieve_node
   │
   ├── Difficulté 3, 1er retrieval : un retrieval par sous-requête (HyDE off)
   │   → fusion des résultats + re-rank global sur la question originale
   │
   ├── Difficulté 1 : HyDE désactivé (économise un appel LLM)
   │
   └── Difficulté 2 : HyDE activé → génère une réponse fictive (gemma4:12b)
   │
   ┌─────────────────────────────────────────┐
   │                                         │
   ▼                                         ▼
Recherche SÉMANTIQUE MMR              Recherche LEXICALE (BM25)
→ réponse fictive vectorisée (bge-m3) → question normalisée (sans accents)
→ MMR : 25 chunks pertinents + diversifiés → 25 chunks avec mots exacts
   (parmi 100 candidats, lambda=0.5)   BM25 normalisé : accents, points
K_RETRIEVE = 25                        médians, tokens 2+ chars
   │                                         │
   └──────────────────┬──────────────────────┘
                      ▼
             Fusion RRF (Reciprocal Rank Fusion)
             score(chunk) = 1/(60 + rang_sémantique) + 1/(60 + rang_BM25)
             → filtre diversité : max 3 chunks par source (si multi-sources)
               (désactivé si la recherche est restreinte à une seule source)
             → 15 candidats sélectionnés  (K_RERANK = 15)
                      │
                      ▼
             Déduplication (Jaccard sur tokens normalisés)
             → compare le CONTENU sans la ligne de contexte '[...]'
             → deux chunks partageant > 80 % de leurs tokens uniques
               → seul le mieux classé RRF est conservé
             → évite de gaspiller des slots de re-ranking sur des passages répétés
             → K_RERANK = 15 compense les 3-5 chunks retirés en moyenne
                      │
                      ▼
             Re-ranking CrossEncoder (BAAI/bge-reranker-v2-m3)
             → lit chaque paire (requête courante, chunk) ENSEMBLE
             → score logit : > 0.5 = pertinent, < 0.5 = hors-sujet
             → chunks sous RERANK_THRESHOLD = 0.5 écartés
             → 5 meilleurs gardés  (K_FINAL = 5)
                      │
                      ▼
             Fusion des pools (seulement au 2ème retrieval)
             → 3 slots : meilleurs chunks du 1er retrieval (déjà triés)
             → 2 slots réservés : top 2 des nouveaux chunks,
               re-rankés sur la requête reformulée (pas la question originale)
             → garantit que l'info ciblée arrive au LLM même si elle est
               dans un chunk court qui perdrait face aux chunks généraux
        │
        ▼ [LangGraph] _route_after_retrieve
   │
   ├── Difficulté 1 : pas de grade_documents (économise un appel LLM)
   │   → vérifie si un chunk se termine par un marqueur de coupure de page
   │     (ex: "PAGE 19 SUR 78")
   │   → si troncature détectée : upgrade_difficulty → rewrite_query → retrieve
   │   → sinon : generate directement
   │
   └── Difficulté 2/3 : grade_documents
        │
        ▼ [LangGraph] grade_documents
   Le LLM (gemma4:12b) évalue si les chunks accumulés sont suffisants :
   → les chunks tronqués sont annotés [⚠ TRONQUÉ] pour le LLM
   → "OUI" → passe directement à generate
   → "NON — [ce qui manque]" → passe à la boucle de reformulation
        │
        ├── OUI ou max tentatives atteint (MAX_ATTEMPTS=1)
        │                                            ▼
        │                                  [LangGraph] generate_node
        │                                  Prompt = question + 5 chunks + instructions
        │                                  gemma4:12b, temperature=0, num_ctx=4096
        │                                  → Réponse (Open WebUI ou terminal)
        │
        └── NON + tentatives restantes
                │
                ├── difficulté < 3 : upgrade_difficulty → difficulté passe à 3
                │
                ▼
        [LangGraph] rewrite_query
        Le LLM (gemma4:12b) reformule la requête sur ce qui manque précisément
        → reçoit : question originale, requête actuelle, verdict du grade
                │
                └──→ retrieve_node (2ème et dernier retrieval)


PHASE 3 : ÉVALUATION (à la demande)
──────────────────────────────────────────────────────────────────
python src/run_agentic_all.py
→ charge data/questions.json (40 questions avec réponses de référence)
→ pour chaque question : lance ask_question_agentic() avec verbose=True
→ capture les logs de l'agent (_Tee : écrit sur console ET buffer)
→ sauvegarde après chaque question dans data/agentic_results.json (crash-safe)
→ push sur GitHub toutes les 5 questions (PUSH_EVERY = 5)

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
        metadata={"source": pdf_path.name, "page": page_num + 1},  # 1-indexé pour l'affichage
    ))
```

**Pourquoi page par page ?** Un tableau peut couvrir toute une page. Si on extrait le PDF d'un seul bloc puis qu'on découpe par taille, le tableau sera scindé au milieu. Avec `page_chunks=True`, le tableau reste dans un seul document avant le découpage.

### Concaténation des pages (_concat_pages)

Une page de PDF n'est pas une unité de sens : une section peut commencer en bas d'une page et continuer sur la suivante, surtout dans une brochure mise en page librement. Plutôt que de découper page par page puis de gérer des chevauchements artificiels, le pipeline recolle d'abord **toutes** les pages d'un même document en un seul texte continu :

```python
def _concat_pages(pages: list[Document]) -> tuple[str, str]:
    source = pages[0].metadata["source"]
    parts = []
    for doc in pages:
        if _is_toc_page(doc.page_content):   # pages de TdM exclues ici
            continue
        page_num = doc.metadata["page"]
        cleaned = _clean_text(doc.page_content)
        if cleaned:
            parts.append(f"<!-- PAGE {page_num} -->\n{cleaned}")
    return "\n\n".join(parts), source
```

Chaque page conserve un marqueur `<!-- PAGE X -->` à son point de jonction. Ce marqueur n'est jamais montré au LLM ni à l'embedding — il sert uniquement, en interne, à retrouver le numéro de page d'une position donnée dans le texte concaténé (`_get_page_for_position`), puis il est retiré du texte final (`_strip_page_markers`).

`_clean_text` déduplique les lignes consécutives identiques (artefacts fréquents de l'extraction PDF) et réduit les sauts de ligne en excès.

### Détection des tables des matières (_is_toc_page)

Les pages de TdM listent uniquement des numéros de pages (ex: `1.1  Objectifs . . . . . . . . . 7`). Si plus de 30 % des lignes contiennent `...` ou `. . .`, la page est ignorée avant même la concaténation :

```python
# Proportion de lignes avec des points de suspension > seuil → c'est une TdM
dot_lines = sum(1 for l in lines if ". . ." in l or "..." in l)
return dot_lines / len(lines) > TOC_DOT_RATIO  # TOC_DOT_RATIO = 0.3
```

### Pré-découpe par titres (_presplit_by_headers)

Le texte concaténé est ensuite découpé sur chaque titre Markdown `#` ou `##` :

```python
header_pattern = re.compile(r"^(#{1,2})\s+.+", re.MULTILINE)
split_positions = [m.start() for m in header_pattern.finditer(full_text)]
```

Chaque bloc obtenu hérite de la page de son **premier** marqueur `<!-- PAGE X -->` (donc la page où il commence — pas forcément la seule page qu'il couvre). C'est la métadonnée `page` qui sera attachée au chunk final.

Les blocs trop courts pour être de vraies sections (`< MIN_SECTION_SIZE_FOR_SPLIT` = 800 caractères, ex: un titre suivi d'une seule ligne de calendrier) sont fusionnés avec le bloc suivant, en conservant les métadonnées du premier :

```python
MIN_SECTION_SIZE_FOR_SPLIT = 800

buffer = None
for section in sections:
    buffer = section if buffer is None else Document(
        page_content=buffer.page_content + "\n\n" + section.page_content,
        metadata=buffer.metadata,
    )
    if len(buffer.page_content) >= MIN_SECTION_SIZE_FOR_SPLIT:
        merged.append(buffer)
        buffer = None
```

S'il n'y a aucun titre `#`/`##` dans le document, tout le texte forme un seul bloc.

### Split agentique (_agentic_split_section, LLM gemma4:12b)

Chaque bloc thématique (≥ 800 caractères) part vers le LLM avec une consigne précise : insérer le marqueur `===SPLIT===` entre chaque sous-section logiquement distincte, **sans modifier le texte**, et jamais à l'intérieur d'un tableau Markdown.

```python
SPLIT_PROMPT = """Texte d'une section de brochure universitaire :
---
{section_text}
---

Insère le marqueur ===SPLIT=== entre chaque sous-section logique distincte
(changement de sujet, nouveau paragraphe thématique).
Ne modifie pas le texte. Insère uniquement des marqueurs ===SPLIT===
aux endroits appropriés.

RÈGLE ABSOLUE : ne coupe JAMAIS à l'intérieur d'un tableau...
Si la section entière porte sur un seul sujet, ne mets aucun marqueur."""
```

**Garde-fou anti-hallucination :** un LLM peut reformuler, résumer ou tronquer le texte au lieu de se contenter d'y insérer des marqueurs. Pour s'en protéger, le pipeline compare les mots du texte d'origine à ceux de la réponse :

```python
original_words = set(section_text.lower().split())
response_words = set(response.lower().replace(SPLIT_MARKER.lower(), "").split())
preserved = len(original_words & response_words) / len(original_words)

if preserved < 0.6:          # moins de 60% des mots d'origine retrouvés
    return [section_text.strip()]   # → on rejette le découpage, bloc gardé intact
```

C'est une validation purement déterministe (comptage d'ensembles de mots) — aucun appel LLM supplémentaire pour vérifier le LLM.

**Conséquence directe pour la métadonnée `page` :** un bloc peut très bien contenir le texte de la page 11 et de la page 12 si le LLM n'a inséré aucun marqueur entre les deux (même sujet) ou si le titre concerné chevauchait les deux pages dans le PDF d'origine. Dans ce cas, le chunk garde la métadonnée `page` du **premier** marqueur `<!-- PAGE X -->` qu'il contient, c'est-à-dire la page où le bloc a commencé. Si les chunks obtenus ressemblent malgré tout à un découpage page par page sur une brochure donnée, c'est en général parce que cette brochure place un titre `#`/`##` par page — pas parce que le pipeline est borné à la page.

### Fallback déterministe (_fallback_split_large)

Filet de sécurité pour les segments encore trop gros (> `MAX_CHUNK_SIZE` = 1500 caractères) après le split agentique — par exemple si le LLM n'a mis aucun marqueur sur un très long bloc :

```python
MAX_CHUNK_SIZE = 1500

splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHUNK_SIZE,
    chunk_overlap=0,   # pas d'overlap : ce n'est qu'un filet de sécurité, pas le découpage principal
    separators=["\n## ", "\n### ", "\n\n", "\n|", "\n", " "],
)
```

**Ordre de priorité des séparateurs :**
1. `\n## ` et `\n### ` — coupe en priorité entre sections Markdown
2. `\n\n` — coupe entre paragraphes
3. `\n|` — coupe avant une ligne de tableau, **protège les tableaux Markdown** : une ligne `| col1 | col2 |` ne sera jamais coupée au milieu
4. `\n` — coupe entre lignes
5. espace — en dernier recours

Sans overlap, contrairement à une ancienne version du pipeline : la cohérence sémantique est désormais assurée par le split agentique en amont, pas par un recouvrement systématique de caractères.

### Filtre des micro-chunks et numérotation par page

Tout chunk dont le contenu fait moins de `MIN_CONTENT_SIZE` (30 caractères) est écarté — numéros de page isolés, symboles seuls.

Chaque chunk reçoit ensuite un `chunk_index` : son rang (0, 1, 2...) parmi les chunks qui partagent la même métadonnée `page`. Cette numérotation est utilisée par `agent.py` (`_expand_truncated`) pour retrouver, sans appel LLM, le premier chunk de la page suivante d'un document quand un chunk se termine par un marqueur de coupure de page.

### Contextualisation déterministe (_contextualize_chunk)

L'embedding et BM25 ne lisent que le texte du chunk — pas ses métadonnées. Un chunk isolé ne dit pas de lui-même dans quel établissement ou quelle section il se trouve. Contrairement à une version antérieure du pipeline, **aucun LLM n'intervient à cette étape** : tout est construit par regex et comptage de fréquence.

**Chemin de titres (_extract_section_path) :** extrait par regex les titres `#`/`##`/`###` présents dans le texte du chunk lui-même (puisqu'ils y sont restés depuis l'extraction Markdown) :

```python
def _extract_section_path(text: str) -> str:
    headers = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,3})\s+(.+)", line)
        if match:
            title = match.group(2).strip().strip("*")
            if title and title not in headers:
                headers.append(title)
    return " > ".join(headers)
```

**Mots-clés (_extract_keywords_deterministic) :** comptage de fréquence des mots et bigrammes du chunk après suppression des stopwords français et du préfixe de contexte s'il existe déjà, des titres Markdown et de la ponctuation — les 5 termes les plus fréquents sont retenus :

```python
words = re.findall(r"[a-zàâäéèêëïîôùûüÿçœæ]{3,}", clean.lower())
words = [w for w in words if w not in _STOPWORDS]
bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
freq = Counter(bigrams) + Counter(words)
top = [term for term, _ in freq.most_common(8) if freq[term] >= 1][:5]
```

**Pourquoi déterministe plutôt qu'un LLM ?** Un comptage de fréquence ne peut pas halluciner — les mots-clés produits sont garantis présents dans le chunk. C'est aussi instantané (zéro appel réseau) là où un LLM générant des mots-clés pour ~700 chunks ajoutait auparavant 15 à 60 minutes au temps d'ingestion.

Résultat :
```
[ENS.pdf | p.12 | Cours communs de L3 | cours obligatoires, ects, l3]

## Cours communs de L3
Les quatre cours obligatoires sont...
```

**Checkpoint pickle :** seul le split agentique (étape coûteuse, un appel LLM par bloc thématique) appelle Ollama pendant l'ingestion. En cas de crash llama-server en cours de route, la progression est sauvegardée après chaque **document source** complet (`done_sources`, `pipeline_version: 3`). Au prochain lancement d'`ingest.py`, les documents déjà traités sont sautés et l'ingestion repart des documents restants.

### Stockage dans ChromaDB

Avant d'écrire dans ChromaDB, `ingest.py` crée le dossier cible s'il n'existe pas (`VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)`) — les bindings Rust de ChromaDB ne créent pas le dossier automatiquement, ce qui causerait une erreur `SQLITE_CANTOPEN` sur Colab.

Les chunks sont envoyés à ChromaDB par **lots de 50** pour éviter les timeouts Ollama :

```python
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)  # nécessaire pour les bindings Rust de ChromaDB
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

**HyDE est désactivé pour les questions de difficulté 1** (factuelles). Ces questions cherchent une valeur unique (nom, date, nombre) — la question brute est suffisante pour la recherche sémantique, et on économise un appel LLM.

BM25 (étape 2) continue d'utiliser la **question originale** pour les correspondances exactes.

### Étape 2 : Recherche sémantique (MMR)

La question fictive (HyDE) ou la question brute est vectorisée par bge-m3. ChromaDB retourne 25 chunks avec MMR (Maximal Marginal Relevance) :

```python
# MMR : pertinents ET diversifiés — évite de retourner 25 extraits du même paragraphe
mmr_docs = vector_db.max_marginal_relevance_search(
    hyde_query,
    k=K_RETRIEVE,          # K_RETRIEVE = 25 — nombre de résultats voulus
    fetch_k=MMR_FETCH_K,   # MMR_FETCH_K = 100 — candidats initiaux examinés (K_RETRIEVE × 4)
    lambda_mult=MMR_LAMBDA, # MMR_LAMBDA = 0.5 — équilibre pertinence/diversité
    filter=chroma_filter,   # filtre sur la source si identify_sources l'a précisé
)
```

MMR pioche parmi 100 candidats et en sélectionne 25 qui maximisent à la fois la pertinence (proches de la question) et la diversité (éloignés les uns des autres). Cela évite que les 25 résultats soient 25 extraits du même paragraphe.

### Étape 3 : Recherche BM25 (lexicale)

L'index BM25 est construit **une seule fois par session** depuis tous les chunks stockés dans ChromaDB. La question originale est normalisée puis scorée :

```python
# Normalise la question avec la même fonction que lors de l'indexation
bm25_scores = bm25.get_scores(_tokenize(question))
# Restreint aux sources demandées si identify_sources en a choisi
top_bm25 = sorted(candidate_indices, key=lambda i: bm25_scores[i], reverse=True)[:K_RETRIEVE]
```

### Étape 4 : Fusion RRF (Reciprocal Rank Fusion)

Les deux listes de 25 résultats sont fusionnées. La formule RRF :

```
score(chunk) = 1/(60 + rang_sémantique) + 1/(60 + rang_BM25)
```

Un chunk bien classé dans les deux listes obtient un score élevé. La constante 60 (`RRF_K`) est la valeur standard de la littérature — elle empêche les premiers rangs de dominer trop fortement.

**Pourquoi RRF plutôt qu'une somme pondérée ?** RRF est indépendant des valeurs brutes des scores (qui varient selon les modèles) — il ne regarde que les positions dans le classement.

**Filtre de diversité :** maximum 3 chunks par document source quand on cherche dans plusieurs sources — évite que les 15 slots soient saturés par des extraits du même PDF. Ce plafond est désactivé (porté à K_RERANK=15) quand `identify_sources` a restreint la recherche à une seule source.

### Étape 4bis : Déduplication (Jaccard)

Avant le re-ranking, les quasi-doublons RRF sont supprimés. La comparaison se fait sur le **contenu seul** (sans la ligne de contexte `[source | p.X | ...]`), grâce à la fonction `_body()` qui retire le préfixe. Deux chunks dont la similarité de Jaccard sur leurs tokens normalisés dépasse `DEDUP_THRESHOLD = 0.8` sont considérés identiques — seul le mieux classé par RRF est gardé :

```python
# Retire le préfixe de contexte pour comparer le contenu pur
tokens_A = set(_tokenize(_body(chunk_A.page_content)))
tokens_B = set(_tokenize(_body(chunk_B.page_content)))
jaccard = len(tokens_A & tokens_B) / len(tokens_A | tokens_B)
# Si jaccard >= 0.8 → chunk_B écarté (chunk_A mieux classé RRF est déjà dans kept)
```

**Pourquoi comparer sans le préfixe ?** Deux chunks identiques dont le LLM a généré des mots-clés différents (ex: `filière, maths` vs `Ce passage décrit...`) auraient un Jaccard réduit par les tokens du préfixe et échapperaient à la déduplication.

**Pourquoi déduplication avant re-ranking ?** La déduplication retire en moyenne 3 à 5 chunks quasi-identiques. Sans elle, ces slots gaspillés réduisent la diversité des chunks envoyés au CrossEncoder. C'est pour compenser cette perte que `K_RERANK` a été augmenté de 10 à 15 : on passe 15 candidats au re-ranker sachant qu'environ 3 à 5 seront des quasi-doublons supprimés en amont.

### Étape 5 : Re-ranking (CrossEncoder)

Le re-ranker reçoit les candidats RRF après déduplication (jusqu'à 15, en pratique ~10-12 après dédup). Pour chaque chunk, il forme la paire `(question originale, chunk)` et la lit ensemble :

```
Paires envoyées au re-ranker :
("Quels sont les cours obligatoires ?", "Les quatre cours communs sont...")  → logit  8.4
("Quels sont les cours obligatoires ?", "La bibliothèque est ouverte...")    → logit -1.2
```

Les 5 chunks avec les scores les plus élevés sont gardés. Les chunks sous le seuil `RERANK_THRESHOLD = 0.5` sont écartés même s'ils font partie des 5 premiers.

### Étape 5bis : Fusion des pools lors du 2ème retrieval

Au 2ème retrieval, le pipeline ne fait pas un re-rank global sur l'ensemble des chunks anciens + nouveaux. Il **réserve des slots** par retrieval :

```
Pool final (K_FINAL = 5 slots) :
  3 slots → meilleurs chunks du 1er retrieval  (déjà triés par re-rank, on prend [:3])
  2 slots → top 2 des nouveaux chunks, re-rankés sur la requête reformulée
```

**Pourquoi ne pas re-ranker tout le pool ensemble ?** Le CrossEncoder score chaque paire `(requête, chunk)`. Si la requête est la question originale (large), les chunks riches du 1er retrieval dominent toujours. Un chunk court qui contient l'info précise manquante — par exemple une seule ligne `"La durée minimale du stage est de 3 mois"` — perd systématiquement face à des chunks denses sur le sujet général, même s'il répond exactement à ce qui manquait.

**Pourquoi re-ranker les nouveaux chunks sur la requête reformulée (et non la question originale) ?** La requête reformulée est précisément ciblée sur l'info manquante (ex: `"durée minimale stage obligatoire"`). Le CrossEncoder peut alors correctement scorer `(requête ciblée, chunk court)` comme très pertinent, alors que la même paire avec la question originale donnerait un score médiocre.

```python
# agent.py — retrieve_node au 2ème retrieval
old_top = state["docs"][:K_FINAL - 2]                                # top 3 du 1er retrieval
new_top = _rerank(state["current_query"], new_docs_deduped, n=2)     # top 2 du 2ème retrieval, scorés sur la requête reformulée
final   = old_top + [d for d in new_top if d.page_content not in seen]  # 5 chunks au total
```

### Étape 6 : Évaluation des chunks (grade_documents)

Le pipeline agentique ajoute une étape que le RAG classique n'a pas : demander au LLM si les chunks trouvés sont suffisants pour répondre. Les chunks tronqués (détectés par le regex `TRUNCATION_RE`) sont annotés `[⚠ TRONQUÉ]` pour que le LLM identifie les listes incomplètes :

```python
# Prompt envoyé au LLM avec les chunks accumulés
GRADE_PROMPT = """Ces extraits contiennent-ils l'information nécessaire ?
- Si oui, réponds uniquement : OUI
- Si non, réponds : NON — [explique en 1 phrase ce qui manque]
- Si un extrait est marqué [⚠ TRONQUÉ], considère que la liste est incomplète"""

# Exemples de réponse NON :
# "NON — les extraits mentionnent le stage mais n'indiquent pas sa durée minimale"
```

Si insuffisant et si le nombre de tentatives est sous `MAX_ATTEMPTS=1`, le pipeline peut reformuler la requête et relancer un retrieval. Il y a donc au maximum 2 retrievals : l'initial et une reformulation ciblée.

**Court-circuit pour difficulté 1 :** Les questions factuelles simples ne passent pas par `grade_documents` (économie d'un appel LLM). Le seul cas où la boucle se déclenche est si un chunk est tronqué (marqueur de coupure de page détecté).

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
    "logs": "[Agent] Source(s) ciblée(s) : ENS.pdf\n...",
    "chunks": [
        {"source": "ENS.pdf", "page": 3, "rerank_score": 8.412, "content": "..."}
    ]
}
```

**Reprise sur interruption :** Si le run est interrompu (Ctrl+C, crash), le fichier est lu au prochain lancement et seules les questions manquantes sont traitées.

**Push automatique :** Sur Colab, les résultats sont pushés sur GitHub toutes les 5 questions (`PUSH_EVERY = 5`) pour ne pas perdre la progression si Colab coupe la session.

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

RAGAS utilise gemma4:12b (ou gemma2:2b sur Colab pour la rapidité) et bge-m3 via l'API OpenAI-compatible d'Ollama (`http://localhost:11434/v1`). Aucun appel à l'extérieur.

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
