# 03 — Fonctionnement détaillé

## Le flux complet

```
PHASE 1 : INGESTION (une seule fois, ou après ajout de documents)
──────────────────────────────────────────────────────────────────
Fichiers PDF/TXT/DOCX  (documents/)
        │
        ▼
   Extraction en Markdown structuré  (pymupdf4llm)
   → titres conservés, symboles mathématiques bien espacés
        │
        ▼
   Découpage en chunks de 1000 caractères  (RecursiveCharacterTextSplitter)
   → overlap de 200 caractères entre chunks consécutifs
   → coupe en priorité sur les séparateurs Markdown (##, \n\n, \n)
        │
        ▼
   Transformation en vecteurs de 1024 nombres  (bge-m3 via Ollama)
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
   → question vectorisée par bge-m3       → question découpée en mots
   → 20 chunks les plus proches (cosinus) → 20 chunks avec les mots exacts
        │                                         │
        └──────────────────┬──────────────────────┘
                           ▼
                  Fusion RRF (Reciprocal Rank Fusion)
                  → score = Σ 1/(60 + rang) pour chaque chunk
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
                  (question + 5 chunks + instructions strictes)
                           │
                           ▼
                  Génération  (gemma2:2b via Ollama, 4096 tokens)
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
            Terminal           Interface Streamlit
            (ask.py)           (app.py → localhost:8501)


PHASE 3 : ÉVALUATION (à la demande)
──────────────────────────────────────────────────────────────────
python src/evaluate.py
→ charge data/questions.json (40 questions avec réponses de référence)
→ pour chaque question : lance le pipeline RAG complet
→ évalue avec 5 métriques via appels LLM
→ affiche un tableau de résultats par question et par niveau
→ sauvegarde dans data/results.json (incrémental — Ctrl+C ne perd rien)
```

---

## Phase 1 — Ingestion en détail

### Extraction Markdown (pymupdf4llm)

`pymupdf4llm` convertit chaque PDF en un document Markdown unique. Cela résout un problème spécifique aux PDFs mathématiques où PyMuPDF brut colle les symboles aux mots :

```
# PyMuPDF brut
"SoientX⊂R d un ouvert etf:X→R m. d2fdx2"

# pymupdf4llm
"Soient X ⊂ R^d un ouvert et f : X → R^m."
```

Les titres (`## Chapitre 3`) et la structure du document sont préservés, ce qui améliore la qualité du découpage en chunks.

### Découpage (chunking)

`RecursiveCharacterTextSplitter` coupe le texte dans cet ordre de priorité :
1. Titres Markdown (`## `, `### `) — coupe entre sections
2. Paragraphes (`\n\n`)
3. Lignes (`\n`)
4. Espaces (entre mots)
5. Caractères (en dernier recours)

**chunk_size = 1000** — assez grand pour contenir une définition mathématique complète.

**chunk_overlap = 200** — les chunks se chevauchent pour éviter de couper une idée en deux :

```
Chunk 1 : "...un espace vectoriel normé est dit complet si toute suite de
           Cauchy converge. On appelle un tel espace un espace de [FIN]"

Chunk 2 : "[DÉBUT] espace de Banach. Les espaces de Banach jouent un rôle
           central en analyse fonctionnelle..."
```

Sans chevauchement, la définition serait fragmentée entre deux chunks.

### Embeddings et stockage

`bge-m3` transforme chaque chunk en 1024 nombres (un vecteur). Ce calcul prend ~1-2 secondes par chunk. Pour ~700 chunks : 15-25 minutes. C'est pour ça qu'on le fait une seule fois et qu'on sauvegarde dans `vector_db/`.

**Attention :** si tu changes de modèle d'embedding, la dimension des vecteurs change. ChromaDB refusera les nouveaux vecteurs. Il faut supprimer `vector_db/` et tout recréer.

---

## Phase 2 — Question/Réponse en détail

### Étape 1 : Recherche sémantique

La question est vectorisée par bge-m3. ChromaDB calcule la similarité cosinus entre ce vecteur et les ~700 vecteurs stockés. Les 20 plus proches (`K_RETRIEVE = 20`) sont retournés.

**Avantage :** trouve des passages qui parlent du même concept même avec des mots différents.
**Limite :** peut rater des passages contenant des termes exacts spécifiques (noms propres, codes, sigles).

### Étape 2 : Recherche BM25

La question est tokenisée en mots minuscules. BM25 score chaque chunk selon la fréquence et la rareté des mots. Les 20 meilleurs sont retournés.

**Avantage :** trouve les passages contenant exactement les mots de la question.
**Limite :** ne comprend pas le sens, peut rater des synonymes.

### Étape 3 : Fusion RRF (Reciprocal Rank Fusion)

Les deux listes de 20 résultats sont fusionnées. La formule RRF :

```
score(chunk) = 1/(60 + rang_sémantique) + 1/(60 + rang_BM25)
```

Un chunk bien classé dans les deux listes obtient un score élevé. Un chunk excellent dans une seule liste peut quand même bien se classer.

**Pourquoi RRF plutôt qu'une somme pondérée ?** RRF est indépendant des valeurs brutes des scores (qui varient selon les modèles) — il ne regarde que les rangs.

**Filtre de diversité :** maximum 3 chunks par document source pour éviter que les 10 slots soient saturés par des extraits du même PDF.

### Étape 4 : Re-ranking (CrossEncoder)

Le re-ranker reçoit les 10 candidats RRF. Pour chaque chunk, il forme la paire `(question, chunk)` et la lit **ensemble** — contrairement aux embeddings qui calculent question et chunk séparément.

```
Paires envoyées au re-ranker :
("Quels sont les cours obligatoires ?", "Les quatre cours communs sont...")  → score 8.4
("Quels sont les cours obligatoires ?", "La bibliothèque est ouverte...")    → score 0.2
...
```

Les 5 chunks avec les scores les plus élevés sont gardés pour le LLM.

**Pourquoi ajouter un re-ranker ?** Un chunk peut être bien classé par RRF mais ne pas vraiment répondre à la question. Le re-ranker comprend mieux la pertinence réelle grâce à la lecture conjointe question+chunk.

### Étape 5 : Génération

Les 5 chunks sont assemblés en un contexte avec leur source. Le template `rag_prompt.txt` est chargé et les variables `{question}` et `{context}` sont remplacées. Le LLM gemma2:2b génère la réponse.

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

### Les 5 métriques

Toutes les métriques sont calculées par appels LLM — le LLM joue le rôle de juge.

| Métrique | Question posée | Ground truth ? |
|---|---|---|
| **Faithfulness** | La réponse invente-t-elle des choses non présentes dans les chunks ? | Non |
| **Answer Relevancy** | La réponse répond-elle à la question posée ? | Non |
| **Context Quality** | Les chunks récupérés sont-ils pertinents pour cette question ? | Non |
| **Context Recall** | Les chunks couvrent-ils tout ce que contient la réponse de référence ? | Oui |
| **Answer Correctness** | La réponse est-elle factuellement correcte par rapport à la référence ? | Oui |

### Interpréter les scores

```
Faithfulness faible  → le LLM hallucine, il invente des infos non présentes dans les chunks
Relevancy faible     → le LLM répond à côté, il ne comprend pas bien la question
Context Quality faible → le retrieval ramène des chunks hors sujet
Context Recall faible  → le retrieval rate des informations importantes
Correctness faible     → la réponse est incorrecte par rapport aux documents
```

### Les résultats sont sauvegardés

`data/results.json` est mis à jour après chaque question. Si tu fais Ctrl+C en cours d'évaluation, les résultats déjà calculés sont conservés.
