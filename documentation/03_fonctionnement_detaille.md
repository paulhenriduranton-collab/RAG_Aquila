# 03 — Fonctionnement détaillé

## Le flux complet

```
PHASE 1 : INGESTION (une seule fois, ou après ajout de documents)
──────────────────────────────────────────────────────────────────
Fichiers PDF/TXT/DOCX  (documents/)
        │
        ▼
   Extraction en Markdown structuré  (pymupdf4llm)
   → espaces corrects autour des symboles, titres conservés
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
        ├─────────────────────────────────────────────┐
        ▼                                             ▼
   Recherche SÉMANTIQUE                      Recherche LEXICALE (BM25)
   → question vectorisée par bge-m3           → question découpée en mots
   → 20 chunks les plus proches               → 20 chunks avec les mots exacts
        │                                             │
        └──────────────────┬──────────────────────────┘
                           ▼
                  Fusion des scores
                  (sémantique + BM25 normalisé × BM25_WEIGHT)
                  → filtre de diversité : 1 chunk max par page source
                  → top 5 chunks sélectionnés  (K_FINAL = 5)
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
            Terminal           API FastAPI (:8000)
            (ask.py)                │
                                    ▼
                              Open WebUI (:3000)
                              streaming token par token


PHASE 3 : ÉVALUATION (à la demande, pour mesurer et améliorer)
──────────────────────────────────────────────────────────────────
python evaluation/generate_dataset.py
→ génère ~60 paires (question, chunk_de_référence) depuis les chunks existants
→ sauvegarde dans evaluation/dataset.json
→ à nettoyer manuellement (garder 25-40 questions de qualité)

python evaluation/eval_retrieval.py
→ lance le retrieval sur chaque question du dataset
→ vérifie si le chunk de référence est dans les top-K résultats
→ calcule Recall@1, Recall@3, Recall@5, Recall@10, MRR
→ affiche les questions non trouvées pour diagnostic
```

---

## Phase 1 — Ingestion en détail

### Extraction Markdown (pymupdf4llm)

`pymupdf4llm` convertit chaque PDF en un document Markdown unique. Cela résout un problème spécifique aux PDFs de maths où PyMuPDF brut colle les symboles aux mots :

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

**chunk_size = 1000** — assez grand pour contenir une définition mathématique complète (600-900 caractères typiquement).

**chunk_overlap = 200** — les chunks se chevauchent pour éviter de couper une idée en deux :

```
Chunk 1 : "...un espace vectoriel normé est dit complet si toute suite de
           Cauchy converge. On appelle un tel espace un espace de [FIN]"

Chunk 2 : "[DÉBUT] espace de Banach. Les espaces de Banach jouent un rôle
           central en analyse fonctionnelle..."
```

Sans chevauchement, la définition serait fragmentée entre deux chunks.

### Embeddings et stockage

`bge-m3` transforme chaque chunk en 1024 nombres. Ce calcul prend environ 1-2 secondes par chunk (pour ~700 chunks : 15-25 minutes). C'est pour ça qu'on le fait une seule fois et qu'on sauvegarde dans `vector_db/`.

**Attention :** si tu changes de modèle d'embedding, la dimension des vecteurs change (768 pour nomic, 1024 pour bge-m3). ChromaDB refusera d'insérer des vecteurs de dimension différente de celle de la collection existante. Il faut supprimer `vector_db/` et tout recréer.

---

## Phase 2 — Question/Réponse en détail

### Les deux recherches parallèles

**Recherche sémantique :**
La question est vectorisée par bge-m3. ChromaDB calcule la similarité cosinus entre ce vecteur et les ~700 vecteurs stockés. Les 20 plus proches (`K_RETRIEVE = 20`) sont retournés avec leur score (0 à 1).

**Recherche BM25 :**
La question est tokenisée en mots minuscules. BM25 score chaque chunk selon la fréquence et la rareté de ces mots dans la base. Les 20 meilleurs sont retournés.

### Fusion des scores

Les deux listes sont fusionnées par somme pondérée :

```
score_final = score_sémantique + (score_BM25_normalisé × BM25_WEIGHT)
```

`BM25_WEIGHT = 0.5` par défaut — les deux méthodes ont un poids similaire.

Un chunk présent dans les deux listes cumule les contributions des deux méthodes et remonte en tête du classement.

Le filtre de diversité limite à **1 chunk par page source** pour éviter que les 5 slots soient saturés par des extraits du même paragraphe.

### Construction du prompt

Le prompt est lu depuis `prompts/rag_prompt.txt`. Les variables `{question}` et `{context}` sont remplacées :

```
Tu es un assistant documentaire strict...

CONTEXTE :
Source : ENS.pdf
[chunk 1...]

---

Source : SORBONNE.pdf
[chunk 2...]

---
[etc. jusqu'à 5 chunks]

QUESTION : Qu'est-ce qu'un espace de Banach ?
```

Le modèle est explicitement interdit d'inventer ou d'utiliser ses connaissances propres.

---

## Phase 3 — Évaluation en détail

### Pourquoi évaluer ?

Sans métriques, il est impossible de savoir si une modification du pipeline (chunk_size, BM25_WEIGHT, seuil sémantique) améliore ou détériore les résultats. L'évaluation permet de comparer objectivement avant/après.

### Le dataset synthétique

`generate_dataset.py` échantillonne des chunks depuis ChromaDB et demande au LLM de générer une question dont la réponse se trouve dans ce chunk. Le résultat est sauvegardé dans `evaluation/dataset.json`.

Il faut ensuite nettoyer manuellement le dataset : supprimer les questions trop vagues ou incompréhensibles hors contexte. Conserver 25-40 questions de qualité.

### Les métriques

`eval_retrieval.py` exécute le retrieval sur chaque question et mesure :

- **Recall@K** : le chunk de référence est-il parmi les K premiers résultats ?
- **MRR** (Mean Reciprocal Rank) : à quelle position moyenne apparaît le bon chunk ?

```
Recall@5 < 40%  → problème sérieux de retrieval (chunking, embedding)
Recall@5 > 65%  → retrieval solide, optimiser la génération
MRR < 0.30      → bon chunk trouvé mais mal classé (tuner la fusion)
```

Pour tester un paramètre sans modifier le code :

```powershell
python evaluation/eval_retrieval.py --k-retrieve 30 --bm25-weight 0.3 --threshold 0.45
```

---

## Ce qui se passe si tu ajoutes un nouveau document

1. Copie le fichier dans `documents/`
2. Supprime l'ancienne base : `Remove-Item -Recurse -Force vector_db`
3. Relance `python src/ingest.py`
4. Regénère le dataset d'évaluation si nécessaire : `python evaluation/generate_dataset.py`
