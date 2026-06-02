# 03 — Fonctionnement détaillé

## Le flux complet

```
PHASE 1 : INGESTION (une seule fois)
─────────────────────────────────────────────────────────
Fichiers PDF/TXT/DOCX
        │
        ▼
   Lecture des fichiers (PyPDFLoader, TextLoader, Docx2txtLoader)
        │
        ▼
   Découpage en morceaux de 800 caractères (RecursiveCharacterTextSplitter)
        │
        ▼
   Transformation en vecteurs de 768 nombres (nomic-embed-text via Ollama)
        │
        ▼
   Sauvegarde dans la base vectorielle (ChromaDB → vector_db/chroma.sqlite3)


PHASE 2 : QUESTION/RÉPONSE (à chaque question)
─────────────────────────────────────────────────────────
Ta question ("Qu'est-ce qu'un espace de Banach ?")
        │
        ▼
   Transformation en vecteur de 768 nombres (nomic-embed-text)
        │
        ▼
   Recherche des 4 morceaux les plus proches dans ChromaDB
        │
        ▼
   Construction du prompt (question + 4 morceaux + instructions)
        │
        ▼
   Envoi à gemma:latest (via Ollama)
        │
        ▼
   Réponse affichée dans Streamlit
```

---

## Phase 1 — Ingestion en détail

### Étape 1.1 : Lecture des fichiers

Chaque type de fichier a son propre outil de lecture :
- `.pdf` → `PyPDFLoader` (lit page par page)
- `.txt` → `TextLoader` (lit le fichier entier)
- `.docx` → `Docx2txtLoader` (lit le document Word)

Après lecture, chaque morceau contient :
- Le texte brut
- Des métadonnées : le nom du fichier source, le numéro de page (pour les PDF)

### Étape 1.2 : Le découpage (chunking)

**Pourquoi découper ?** Les modèles d'IA ont une limite de texte qu'ils peuvent traiter en une fois. On ne peut pas donner un PDF de 200 pages entier à l'IA.

**Comment fonctionne le découpage récursif ?**

Le `RecursiveCharacterTextSplitter` essaie de couper aux endroits les plus naturels :
1. D'abord sur les **paragraphes** (`\n\n`)
2. Si un paragraphe est encore trop long, sur les **lignes** (`\n`)
3. Si une ligne est encore trop longue, sur les **espaces** (mots)
4. En dernier recours, sur les **caractères**

**Les paramètres utilisés :**
- `chunk_size = 800` → chaque morceau fait au maximum 800 caractères
- `chunk_overlap = 120` → les morceaux se chevauchent de 120 caractères

**Pourquoi le chevauchement ?** Pour ne pas couper une idée en deux. Exemple :

```
Morceau 1 : "...un espace vectoriel normé est dit complet si toute suite de
             Cauchy converge. On appelle un tel espace un espace de [FIN]"

Morceau 2 : "[DÉBUT] espace de Banach. Les espaces de Banach jouent un rôle
             central en analyse fonctionnelle..."
```

Sans chevauchement, la définition serait coupée en deux morceaux et aucun des deux ne serait complet.

**Résultat sur ton projet :** 3 PDFs → 210 pages lues → 601 morceaux créés.

### Étape 1.3 : Transformation en vecteurs (embeddings)

Pour chaque morceau, `nomic-embed-text` génère 768 nombres. Cette liste de nombres s'appelle un **vecteur** ou **embedding**.

Ces nombres ne sont pas aléatoires : deux textes qui parlent du même sujet auront des vecteurs proches. C'est ce qui permet la recherche par sens.

**Temps de calcul :** Environ 1 seconde par morceau sur un ordinateur sans GPU. Pour 601 morceaux → ~10 à 15 minutes. C'est pour ça qu'on le fait une seule fois et qu'on sauvegarde le résultat.

### Étape 1.4 : Sauvegarde dans ChromaDB

Les vecteurs sont sauvegardés dans `vector_db/chroma.sqlite3`. Ce fichier contient :
- Le texte de chaque morceau
- Son vecteur (768 nombres)
- Ses métadonnées (nom du fichier source)

---

## Phase 2 — Question/Réponse en détail

### Étape 2.1 : Transformation de la question

Ta question est transformée en vecteur avec le **même modèle** (`nomic-embed-text`). C'est important : si on utilisait un modèle différent, les vecteurs ne seraient pas comparables.

### Étape 2.2 : Recherche des morceaux pertinents

ChromaDB calcule la **distance** entre le vecteur de ta question et chacun des 601 vecteurs stockés. Il retourne les 4 morceaux dont les vecteurs sont les plus proches (`k=4` dans le code).

**Qu'est-ce que la "distance" entre vecteurs ?** C'est une mesure mathématique (souvent la similarité cosinus) qui indique à quel point deux vecteurs pointent dans la même "direction". Plus ils pointent dans la même direction, plus les textes ont un sens similaire.

### Étape 2.3 : Construction du prompt

Le prompt final ressemble à ça :

```
Tu es un assistant qui répond uniquement à partir du contexte fourni.

Question utilisateur :
Qu'est-ce qu'un espace de Banach ?

Contexte extrait des documents :
Source : analyse_fonctionnelle.pdf
[morceau 1 du PDF...]

---

Source : analyse_fonctionnelle.pdf
[morceau 2 du PDF...]

---

[etc.]

Règles :
- Réponds uniquement avec les informations présentes dans le contexte.
- Si la réponse n'est pas dans le contexte, dis : "Je ne trouve pas..."
- Réponds de manière claire, concise et professionnelle.
- Cite les documents sources si possible.
```

### Étape 2.4 : Génération de la réponse

`gemma:latest` reçoit ce prompt et génère une réponse. Il a une fenêtre de contexte de **4096 tokens** (`num_ctx=4096`), ce qui correspond à environ 3000 mots — largement suffisant pour la question + les 4 morceaux.

**Qu'est-ce qu'un token ?** Un token est une unité de texte pour l'IA — environ 1 mot en anglais, un peu moins en français. "espace de Banach" = 4 tokens environ.

---

## Ce qui se passe si tu ajoutes un nouveau document

1. Tu mets ton nouveau fichier dans `documents/`
2. Tu relances `python src/ingest.py`
3. Chroma recrée la base entière avec tous les documents (anciens + nouveaux)
4. La prochaine question utilisera le nouveau document

**Attention :** Si tu ne relances pas `ingest.py`, le nouveau document est ignoré.
