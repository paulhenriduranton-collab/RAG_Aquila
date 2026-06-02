# 04 — Explication du code

## ingest.py — Le script d'indexation

### Les imports (lignes 1–10)

```python
from pathlib import Path
```
`pathlib` est une librairie Python standard pour manipuler les chemins de fichiers. Elle fonctionne sur Windows (`\`) et Mac/Linux (`/`) sans qu'on ait à s'en préoccuper.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```
L'outil de découpage récursif. Voir la doc `03_fonctionnement_detaille.md` pour le détail.

```python
from langchain_chroma import Chroma
```
L'interface Python pour parler à la base de données ChromaDB.

```python
from langchain_ollama import OllamaEmbeddings
```
L'interface pour appeler le modèle `nomic-embed-text` via Ollama.

```python
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader,
)
```
Trois outils de lecture : un pour chaque format de fichier supporté.

---

### Les constantes (lignes 12–15)

```python
BASE_DIR = Path(__file__).resolve().parent.parent
```
- `__file__` = le chemin complet de `ingest.py` lui-même
- `.resolve()` = transforme en chemin absolu (ex: `C:\Users\...`)
- `.parent` = remonte d'un dossier (on passe de `src/` à `Projet Aquila/`)
- `.parent` = remonte encore (on est déjà à la racine)

Résultat : `BASE_DIR = C:\Users\paulh\OneDrive\Bureau\Projet Aquila`

```python
DOCUMENTS_DIR = BASE_DIR / "documents"   # → .../Projet Aquila/documents
VECTOR_DB_DIR = BASE_DIR / "vector_db"   # → .../Projet Aquila/vector_db
EMBED_MODEL = "nomic-embed-text"
```

---

### La fonction `load_documents()` (lignes 18–37)

```python
def load_documents():
    documents = []
    for file_path in DOCUMENTS_DIR.iterdir():
```
Parcourt tous les fichiers dans `documents/`. `iterdir()` retourne un objet par fichier.

```python
        if file_path.name.startswith("."):
            continue
```
Ignore les fichiers cachés (`.DS_Store` sur Mac, etc.). `continue` = passe au fichier suivant sans traiter celui-ci.

```python
        suffix = file_path.suffix.lower()
```
Récupère l'extension : `.pdf`, `.txt`, `.docx`... `.lower()` met en minuscules pour éviter les problèmes avec `.PDF` ou `.Pdf`.

```python
        if suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(file_path))
        else:
            print(f"Format ignoré : {file_path.name}")
            continue
```
Choisit le bon outil selon l'extension. Si le format n'est pas reconnu, affiche un message et passe au fichier suivant.

```python
        loaded = loader.load()
```
Charge le fichier. Pour un PDF, retourne autant d'éléments que de pages.

```python
        for doc in loaded:
            doc.metadata["source"] = file_path.name
```
Ajoute le nom du fichier dans les métadonnées de chaque morceau. Ça permettra plus tard de dire "cette réponse vient de `analyse_fonctionnelle.pdf`".

```python
        documents.extend(loaded)
    return documents
```
Ajoute les documents chargés à la liste. `extend` ajoute plusieurs éléments d'un coup (contrairement à `append` qui en ajoute un seul).

---

### La fonction `main()` (lignes 40–58)

```python
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(documents)
```
Découpe tous les documents en morceaux. `chunks` est une liste de ~601 éléments.

```python
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
```
Crée l'objet qui va appeler `nomic-embed-text` via Ollama. À ce stade, aucun calcul n'est encore fait.

```python
    db = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        if db is None:
            db = Chroma.from_documents(batch, embeddings, persist_directory=str(VECTOR_DB_DIR))
        else:
            db.add_documents(batch)
```
Traite les chunks par lots de 50 pour éviter les crashs mémoire. Le premier lot crée la base, les suivants ajoutent dedans.

---

## ask.py — Le script de questions

### La différence avec ingest.py

`ingest.py` lit les documents et construit la base — il s'exécute une seule fois.
`ask.py` interroge la base et génère une réponse — il s'exécute à chaque question.

### Les nouvelles constantes

```python
GEN_MODEL = "gemma:latest"
llm = OllamaLLM(model=GEN_MODEL, num_ctx=4096)
```
`OllamaLLM` est le modèle de génération (différent de `OllamaEmbeddings` qui transforme en vecteurs). `num_ctx=4096` = fenêtre de contexte de 4096 tokens.

**Pourquoi `llm` est créé en dehors de la fonction ?** Pour ne pas le recharger à chaque question. Le charger une fois au démarrage est plus efficace.

---

### La fonction `ask_question(question)` ligne par ligne

```python
    vector_db = Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings,
    )
```
Ouvre la base existante (créée par `ingest.py`). Ne recrée rien, lit juste le fichier `chroma.sqlite3`.

```python
    docs = vector_db.as_retriever(search_kwargs={"k": 4}).invoke(question)
```
- `.as_retriever()` = transforme la base en "chercheur"
- `search_kwargs={"k": 4}` = retourne les 4 morceaux les plus proches
- `.invoke(question)` = lance la recherche avec ta question

```python
    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "source inconnue")
        context_parts.append(f"Source : {source}\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)
```
Formate les 4 morceaux trouvés en un bloc de texte avec le nom du fichier source devant chacun. Les morceaux sont séparés par `---` pour que le LLM comprenne où commence chaque extrait.

```python
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.format(question=question, context=context)
```
Lit le fichier `rag_prompt.txt` et remplace `{question}` et `{context}` par les vraies valeurs.

```python
    return llm.invoke(prompt)
```
Envoie le prompt à gemma et retourne sa réponse.

---

## app.py — L'interface Streamlit

```python
st.set_page_config(page_title="RAG simple", page_icon="📄")
```
Configure le titre et l'icône de l'onglet du navigateur.

```python
st.title("RAG simple — Questions Maths :")
st.write("Pose une question...")
```
Affiche du texte sur la page.

```python
question = st.text_input("Votre question")
```
Crée un champ texte. Streamlit stocke ce que l'utilisateur tape dans la variable `question`.

```python
if st.button("Répondre"):
    if not question.strip():
        st.warning("Merci d'écrire une question simple.")
    else:
        with st.spinner("Recherche dans les documents..."):
            answer = ask_question(question)
        st.subheader("Réponse")
        st.write(answer)
```
- `st.button()` retourne `True` quand on clique dessus
- `question.strip()` = enlève les espaces au début/fin pour vérifier que ce n'est pas vide
- `st.spinner()` = affiche une animation de chargement pendant le calcul
- `st.subheader()` = titre de niveau 2
- `st.write()` = affiche le texte de la réponse

---

## rag_prompt.txt — Les instructions pour l'IA

```
Tu es un assistant qui répond uniquement à partir du contexte fourni.

Question utilisateur :
{question}

Contexte extrait des documents :
{context}

Règles :
- Réponds uniquement avec les informations présentes dans le contexte.
- Si la réponse n'est pas dans le contexte, dis : "Je ne trouve pas cette information dans les documents fournis."
- Réponds de manière claire, concise et professionnelle.
- Cite les documents sources si possible.
```

`{question}` et `{context}` sont des **espaces réservés** (placeholders). Python les remplace par les vraies valeurs avant d'envoyer le prompt au modèle.

**Pourquoi ce fichier est séparé du code ?** Pour pouvoir modifier les instructions sans toucher au code Python. Si on veut que l'IA réponde en anglais ou soit plus concise, on modifie juste ce fichier.
