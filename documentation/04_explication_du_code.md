# 04 — Explication du code

## ingest.py — Le script d'indexation

### Les imports (lignes 1–9)

```python
from pathlib import Path
```
`pathlib` est une librairie Python standard pour manipuler les chemins de fichiers. Elle fonctionne sur Windows (`\`) et Mac/Linux (`/`) sans qu'on ait à s'en préoccuper.

```python
import pymupdf4llm
```
Extracteur PDF qui produit du Markdown structuré. Remplace `PyMuPDFLoader` de LangChain qui produisait du texte brut avec les espaces manquants autour des formules mathématiques.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```
L'outil de découpage récursif en chunks. Voir `03_fonctionnement_detaille.md` pour le détail.

```python
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_core.documents import Document
```
Interfaces LangChain pour ChromaDB, les embeddings Ollama, et les loaders de fichiers texte/Word.

---

### Les constantes (lignes 11–14)

```python
BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
EMBED_MODEL = "bge-m3"
```

- `__file__` = chemin complet de `ingest.py` lui-même
- `.resolve().parent.parent` = remonte deux niveaux → racine du projet
- `EMBED_MODEL = "bge-m3"` → modèle multilingue avec fenêtre de 8192 tokens, adapté au français scientifique

---

### La fonction `_load_pdf()` (lignes 17–20)

```python
def _load_pdf(pdf_path: Path) -> list[Document]:
    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    return [Document(page_content=md_text, metadata={"source": pdf_path.name})]
```

Convertit un PDF entier en **un seul document Markdown**. Contrairement à PyMuPDFLoader qui créait un Document par page, `pymupdf4llm` retourne le document entier avec sa structure Markdown préservée (titres `##`, `###`, tableaux, listes).

Le `RecursiveCharacterTextSplitter` s'occupera ensuite de le découper intelligemment selon les séparateurs Markdown.

---

### La fonction `load_documents()` (lignes 23–42)

```python
for file_path in sorted(DOCUMENTS_DIR.iterdir()):
    if file_path.name.startswith("."):
        continue
```
Parcourt les fichiers dans l'ordre alphabétique. Ignore les fichiers cachés (`.DS_Store` sur Mac, etc.).

```python
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8")
        loaded = loader.load()
    elif suffix == ".pdf":
        loaded = _load_pdf(file_path)
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(file_path))
        loaded = loader.load()
```
Choisit l'outil selon le format. Les PDFs passent par `_load_pdf()` (pymupdf4llm), les autres formats utilisent les loaders LangChain standard.

```python
    for doc in loaded:
        doc.metadata["source"] = file_path.name
    documents.extend(loaded)
    print(f"  ✓ {file_path.name} ({len(loaded)} doc(s))")
```
Ajoute le nom du fichier dans les métadonnées — ça permettra d'afficher "cette réponse vient de `calcul_diff.pdf`" lors des questions.

---

### La fonction `main()` (lignes 45–73)

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
)
```

- `chunk_size=1000` → max 1000 caractères par chunk. Dimensionné pour les maths : une définition complète fait souvent 600-900 caractères.
- `chunk_overlap=200` → 200 caractères de recouvrement entre chunks consécutifs, pour ne pas couper une définition en deux.
- `separators` → ordre de priorité de découpage : d'abord les titres Markdown (`##`, `###`), puis les paragraphes, puis les lignes, puis les mots. Exploite la structure produite par pymupdf4llm.

```python
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
batch_size = 50
db = None
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    if db is None:
        db = Chroma.from_documents(batch, embeddings, persist_directory=str(VECTOR_DB_DIR))
    else:
        db.add_documents(batch)
```
Traite les chunks par lots de 50 pour éviter les crashs mémoire. Le premier lot crée la base ChromaDB, les suivants y ajoutent des données.

---

## ask.py — Le script de questions

### La différence avec ingest.py

`ingest.py` lit les documents et construit la base — il s'exécute **une seule fois** (ou après ajout de nouveaux documents).
`ask.py` interroge la base et génère une réponse — il s'exécute **à chaque question**.

---

### Les constantes (lignes 8–17)

```python
EMBED_MODEL = "bge-m3"
GEN_MODEL = "gemma2:2b"
K_RETRIEVE = 20    # candidats par méthode avant fusion
K_FINAL = 5        # chunks envoyés au LLM
RRF_K = 60         # constante RRF (standard = 60)
```

- `EMBED_MODEL` doit être le **même modèle** que dans `ingest.py` — les vecteurs stockés et les vecteurs de requête doivent être dans le même espace mathématique.
- `K_RETRIEVE = 20` → on récupère les 20 meilleurs candidats de chaque méthode (sémantique + BM25) avant de fusionner.
- `K_FINAL = 5` → après fusion, on envoie les 5 meilleurs chunks au LLM. Plus que l'ancien système (3 chunks) pour avoir plus de contexte.
- `RRF_K = 60` → constante standard de la formule RRF. Une valeur plus haute lisse les différences de rang, une valeur plus basse les amplifie.

```python
llm = OllamaLLM(model=GEN_MODEL, num_ctx=4096, temperature=0)
```

- `temperature=0` → réponses déterministes, le modèle ne "brode" pas.
- `num_ctx=4096` → fenêtre de contexte de 4096 tokens, suffisant pour 5 chunks + la question.

**Pourquoi `llm` est créé en dehors de la fonction ?** Pour ne pas le recharger à chaque question. Le charger une fois au démarrage est plus efficace.

---

### Le cache BM25 (lignes 20–35)

```python
_bm25_index: BM25Okapi | None = None
_bm25_chunks: list[tuple[str, dict]] | None = None
```
Variables globales qui stockent l'index BM25 entre les questions. L'index est construit la **première question seulement** (il faut lire tous les chunks de ChromaDB), puis réutilisé pour les suivantes.

```python
def _build_bm25_index(vector_db: Chroma):
    result = vector_db._collection.get(include=["documents", "metadatas"])
    texts = result["documents"]
    _bm25_index = BM25Okapi([t.lower().split() for t in texts])
```

- `vector_db._collection.get()` → récupère **tous** les chunks stockés dans ChromaDB (texte + métadonnées).
- `t.lower().split()` → tokenisation simple : on met en minuscules et on découpe sur les espaces. Chaque chunk devient une liste de mots.
- `BM25Okapi` → implémentation de BM25 de la librairie `rank_bm25`.

---

### La fonction `_merge()` — Fusion RRF (lignes 38–75)

```python
def _merge(semantic, bm25_indices, bm25_chunks, n=K_FINAL):
```

Cette fonction reçoit les deux listes de résultats et retourne les `n` meilleurs chunks fusionnés.

```python
    for rank, (doc, _) in enumerate(semantic):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1.0 / (RRF_K + rank + 1)
        doc_map[key] = doc
```
Pour chaque chunk sémantique, on ajoute `1/(60 + rang + 1)` à son score RRF. Le rang commence à 0 donc le premier chunk reçoit `1/61`, le deuxième `1/62`, etc.

On utilise `page_content` comme clé de dictionnaire car un même chunk peut apparaître dans les deux listes (sémantique et BM25) — dans ce cas, les contributions s'additionnent.

```python
    for rank, idx in enumerate(bm25_indices):
        text, meta = bm25_chunks[idx]
        scores[text] = scores.get(text, 0) + 1.0 / (RRF_K + rank + 1)
```
Même chose pour les résultats BM25. `bm25_indices` contient les indices des chunks dans `bm25_chunks`, triés par score BM25 décroissant.

```python
    # Diversité : au plus 1 chunk par page source
    seen_pages: set[tuple] = set()
    for key, score in ranked:
        page_id = (meta.get("source"), meta.get("page"))
        if page_id not in seen_pages:
            seen_pages.add(page_id)
            top.append((key, score))
        if len(top) == n:
            break
```
Règle de diversité : si les 5 meilleurs chunks RRF viennent tous de la même page du même PDF, on n'enverrait qu'une seule vue du document au LLM. Cette règle force la diversité en sélectionnant au maximum 1 chunk par page.

---

### La fonction `ask_question()` (lignes 82–133)

```python
raw_semantic = vector_db.similarity_search_with_relevance_scores(question, k=K_RETRIEVE)
```
Retourne les 20 chunks sémantiquement les plus proches avec leur score de similarité cosinus (entre 0 et 1).

```python
bm25_scores = bm25.get_scores(question.lower().split())
top_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:K_RETRIEVE]
```
`get_scores()` retourne un score BM25 pour **chaque** chunk de la base. On trie ensuite les indices par score décroissant et on garde les 20 meilleurs.

```python
final_docs, rrf_ranking = _merge(raw_semantic, top_bm25, bm25_chunks)
```
La fusion RRF produit les 5 chunks finaux avec leurs scores.

```python
context = "\n\n---\n\n".join(
    f"Source : {doc.metadata.get('source', '?')}\n{doc.page_content}"
    for doc in final_docs
)
prompt = PROMPT_PATH.read_text(encoding="utf-8").format(question=question, context=context)
return llm.invoke(prompt)
```
Les chunks sont formatés avec leur source, séparés par `---`, et injectés dans le template de prompt avant envoi au LLM.

---

## app.py — L'interface web Streamlit

`app.py` est une interface graphique minimaliste qui remplace le terminal pour poser des questions.

```python
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask import ask_question
```

`sys.path.insert` ajoute le dossier `src/` au chemin Python pour pouvoir importer `ask_question` depuis `ask.py` qui est dans le même dossier.

```python
st.title("RAG Aquila")

question = st.text_input("Votre question :")

if st.button("Envoyer") and question:
    with st.spinner("Recherche en cours..."):
        reponse = ask_question(question)
    st.write(reponse)
```

- `st.text_input` → champ de saisie texte
- `st.button("Envoyer")` → bouton qui déclenche la recherche
- `st.spinner` → indicateur de chargement pendant que `ask_question()` tourne
- `st.write` → affiche la réponse dans la page

**Pour lancer l'interface :**
```powershell
python -m streamlit run src/app.py
```

Streamlit ouvre automatiquement un onglet dans le navigateur à `http://localhost:8501`.

---

## rag_prompt.txt — Les instructions pour l'IA

```
Tu es un assistant documentaire strict. Tu n'as AUCUNE connaissance propre : 
tu lis uniquement le CONTEXTE ci-dessous et tu en extrais la réponse.

RÈGLE ABSOLUE : si l'information n'apparaît pas textuellement dans le CONTEXTE, 
réponds exactement : "Je ne trouve pas cette information dans les documents fournis."
N'invente rien, ne complète pas avec tes connaissances, ne déduis pas 
au-delà de ce qui est écrit.

CONTEXTE :
{context}

QUESTION : {question}

RÉPONSE (uniquement à partir du CONTEXTE, sources citées entre parenthèses) :
```

**Pourquoi cette formulation stricte ?** On dit au LLM qu'il n'a "AUCUNE connaissance propre" pour éviter qu'il mélange ce qu'il sait de son entraînement avec ce que les documents disent. Sans cette contrainte, un LLM comme gemma2:2b peut "compléter" une réponse incomplète avec des informations inventées (hallucination).

**Pourquoi ce fichier est séparé du code ?** Pour pouvoir modifier les instructions sans toucher au code Python. Si on veut que l'IA réponde en anglais, soit plus concise, ou cite les numéros de page, on modifie juste ce fichier.

`{question}` et `{context}` sont des **espaces réservés** (placeholders). Python les remplace par les vraies valeurs via `.format()` avant d'envoyer le prompt au modèle.
