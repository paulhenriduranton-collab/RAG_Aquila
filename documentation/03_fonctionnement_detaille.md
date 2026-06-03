# 03 — Fonctionnement détaillé

## Le flux complet

```
PHASE 1 : INGESTION (une seule fois)
─────────────────────────────────────────────────────────
Fichiers PDF/TXT/DOCX
        │
        ▼
   Extraction en Markdown structuré (pymupdf4llm)
   → espaces corrects, titres conservés, tableaux en Markdown
        │
        ▼
   Découpage en morceaux de 1000 caractères (RecursiveCharacterTextSplitter)
   → chevauchement de 200 caractères entre morceaux consécutifs
   → coupe d'abord sur les séparateurs Markdown (##, ###, \n\n, \n)
        │
        ▼
   Transformation en vecteurs de 1024 nombres (bge-m3 via Ollama)
        │
        ▼
   Sauvegarde dans la base vectorielle (ChromaDB → vector_db/chroma.sqlite3)


PHASE 2 : QUESTION/RÉPONSE (à chaque question)
─────────────────────────────────────────────────────────
Ta question ("Qu'est-ce qu'une différentielle d'ordre 2 ?")
        │
        ├──────────────────────────────────────────────────┐
        ▼                                                  ▼
   Recherche SÉMANTIQUE (dense)                  Recherche LEXICALE (BM25)
   → question transformée en vecteur              → question découpée en mots
   → 20 chunks les plus proches (bge-m3)          → 20 chunks avec les mots exacts
        │                                                  │
        └──────────────────┬───────────────────────────────┘
                           ▼
                  Fusion RRF (Reciprocal Rank Fusion)
                  → score = 1/(60+rang_sémantique) + 1/(60+rang_BM25)
                  → top 5 chunks, 1 seul par page source
                           │
                           ▼
                  Construction du prompt
                  (question + 5 chunks + instructions strictes)
                           │
                           ▼
                  Envoi à gemma2:2b (via Ollama)
                           │
                           ▼
                  Réponse affichée dans le terminal
```

---

## Phase 1 — Ingestion en détail

### Étape 1.1 : Extraction Markdown (pymupdf4llm)

On utilise `pymupdf4llm` plutôt que PyMuPDF brut car les PDFs de maths posent un problème spécifique : les symboles mathématiques se collent aux mots environnants lors de l'extraction.

**Avant (PyMuPDF brut) :**
```
"SoientX⊂R d un ouvert etf:X→R m. d2fdx2"
```

**Après (pymupdf4llm) :**
```
"Soient X ⊂ R^d un ouvert et f : X → R^m."
```

pymupdf4llm produit un document Markdown pour le PDF entier, avec :
- Les espaces correctement placés autour des symboles
- Les titres de chapitres et sections conservés (`## Chapitre 3`)
- Les tableaux convertis en Markdown `| col | col |`

**Limite :** Les formules complexes (intégrales, fractions) restent en texte brut — pas en LaTeX. La reconnaissance de formules nécessiterait Nougat (Meta), incompatible avec Python 3.14.

### Étape 1.2 : Le découpage (chunking)

**Pourquoi découper ?** Les modèles d'IA ont une limite de texte qu'ils peuvent traiter en une fois. On ne peut pas donner un PDF de 200 pages entier à l'IA.

**Comment fonctionne le découpage récursif ?**

Le `RecursiveCharacterTextSplitter` essaie de couper aux endroits les plus naturels, dans cet ordre de priorité :
1. Sur les **titres Markdown** (`## `, `### `) — coupe entre sections
2. Sur les **paragraphes** (`\n\n`) — coupe entre blocs de texte
3. Sur les **lignes** (`\n`)
4. Sur les **espaces** (entre mots)
5. En dernier recours, sur les **caractères**

**Les paramètres utilisés :**
- `chunk_size = 1000` → chaque morceau fait au maximum 1000 caractères
- `chunk_overlap = 200` → les morceaux se chevauchent de 200 caractères

**Pourquoi 1000 caractères pour des maths ?** Une définition mathématique complète avec son énoncé formel fait facilement 600-900 caractères. Avec 1000, on s'assure qu'une définition tient dans un seul chunk.

**Pourquoi le chevauchement ?** Pour ne pas couper une idée en deux. Exemple :

```
Chunk 1 : "...un espace vectoriel normé est dit complet si toute suite de
            Cauchy converge. On appelle un tel espace un espace de [FIN]"

Chunk 2 : "[DÉBUT] espace de Banach. Les espaces de Banach jouent un rôle
            central en analyse fonctionnelle..."
```

Sans chevauchement, la définition serait coupée et aucun chunk ne serait complet.

### Étape 1.3 : Transformation en vecteurs (embeddings)

Pour chaque chunk, `bge-m3` génère 1024 nombres. Cette liste de nombres s'appelle un **vecteur** ou **embedding**.

Ces nombres ne sont pas aléatoires : deux textes qui parlent du même sujet auront des vecteurs proches. C'est ce qui permet la recherche par sens.

**Pourquoi bge-m3 ?** C'est le seul modèle disponible via Ollama qui combine :
- Bonne compréhension du français et du vocabulaire scientifique
- Fenêtre de contexte de 8192 tokens (compatible avec des chunks de 1000 caractères)

**Temps de calcul :** Environ 1-2 secondes par chunk. Pour ~600 chunks → 10 à 20 minutes. C'est pour ça qu'on le fait une seule fois et qu'on sauvegarde le résultat dans `vector_db/`.

### Étape 1.4 : Sauvegarde dans ChromaDB

Les vecteurs sont sauvegardés dans `vector_db/chroma.sqlite3`. Ce fichier contient :
- Le texte de chaque chunk
- Son vecteur (1024 nombres)
- Ses métadonnées (nom du fichier source)

---

## Phase 2 — Question/Réponse en détail

### Étape 2.1 : Deux recherches en parallèle

Quand tu poses une question, le système lance **deux recherches simultanées** :

**Recherche sémantique (dense) :**
La question est transformée en vecteur avec bge-m3. ChromaDB calcule la similarité cosinus entre ce vecteur et les 600+ vecteurs stockés. Il retourne les **20 chunks** (`K_RETRIEVE = 20`) dont les vecteurs sont les plus proches.

→ Bonne pour retrouver des concepts même avec des mots différents

**Recherche lexicale BM25 :**
La question est découpée en mots (`["différentielle", "ordre", "2"]`). BM25 calcule un score pour chaque chunk en fonction de la fréquence et de la rareté de ces mots dans la base. Il retourne les **20 chunks** avec les meilleurs scores lexicaux.

→ Bonne pour retrouver les termes techniques exacts et les symboles

### Étape 2.2 : Fusion RRF (Reciprocal Rank Fusion)

Les deux listes de 20 résultats sont fusionnées en une seule via la formule :

```
score_RRF(chunk) = 1/(60 + rang_sémantique) + 1/(60 + rang_BM25)
```

**Pourquoi cette formule ?**
- Un chunk classé #1 en sémantique contribue 1/61 ≈ 0.0164
- Un chunk classé #1 en BM25 contribue aussi 1/61 ≈ 0.0164
- Un chunk présent dans les **deux** listes cumule les deux contributions → il remonte en tête
- Un chunk absent d'une liste contribue 0 pour cette méthode

**Règle de diversité :** Au maximum 1 chunk par page source, pour ne pas envoyer au LLM 5 chunks du même paragraphe.

**Résultat :** Les 5 meilleurs chunks (`K_FINAL = 5`) sont sélectionnés.

### Étape 2.3 : Construction du prompt

Le prompt final est lu depuis `prompts/rag_prompt.txt` et les variables `{question}` et `{context}` sont remplacées :

```
Tu es un assistant documentaire strict. Tu n'as AUCUNE connaissance propre...

CONTEXTE :
Source : calcul_diff.pdf
[chunk 1...]

---

Source : calcul_diff.pdf
[chunk 2...]

---

[etc. jusqu'à 5 chunks]

QUESTION : Différentielle d'ordre 2
```

### Étape 2.4 : Génération de la réponse

`gemma2:2b` reçoit ce prompt et génère une réponse. Il a une fenêtre de contexte de **4096 tokens** (`num_ctx=4096`), ce qui correspond à environ 3000 mots.

Le prompt est **strict** : le modèle est explicitement interdit d'inventer ou d'utiliser ses connaissances propres. S'il ne trouve pas l'information dans les 5 chunks, il répond : *"Je ne trouve pas cette information dans les documents fournis."*

---

## Ce qui se passe si tu ajoutes un nouveau document

1. Tu mets ton nouveau fichier dans `documents/`
2. Tu supprimes `vector_db/` (sinon l'ancien index persiste)
3. Tu relances `python src/ingest.py`
4. La prochaine question utilisera le nouveau document

**Attention :** Si tu ne relances pas `ingest.py`, le nouveau document est ignoré.
