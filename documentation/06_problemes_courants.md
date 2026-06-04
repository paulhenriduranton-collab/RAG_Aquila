# 06 — Problèmes courants et solutions

## `Collection expecting embedding with dimension 768, got 1024`

**Cause :** La base `vector_db/` a été créée avec `nomic-embed-text` (768 dimensions), mais le code utilise maintenant `bge-m3` (1024 dimensions). Les deux sont incompatibles.

**Solution :**
```powershell
Remove-Item -Recurse -Force vector_db
python src/ingest.py
```

---

## `model "bge-m3" not found, try pulling it first`

**Cause :** Le modèle n'est pas encore téléchargé dans Ollama.

**Solution :**
```powershell
ollama pull bge-m3
```

---

## `uvicorn` n'est pas reconnu

**Cause :** L'environnement virtuel n'est pas activé, ou uvicorn n'est pas installé.

**Solution :**
```powershell
# Activer le venv
venv\Scripts\activate

# Vérifier que uvicorn est installé
pip install "uvicorn[standard]"

# Relancer
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

---

## L'IA répond "Je ne trouve pas cette information dans les documents fournis"

**Cause 1 :** Le passage demandé n'est dans aucun des 5 chunks sélectionnés.

**Solutions :**
1. Lance `python src/ask.py` et regarde les logs `[Fusion] Top 5` — les bons chunks sont-ils sélectionnés ?
2. Reformule la question avec des mots présents dans les documents
3. Lance `python evaluation/eval_retrieval.py` pour mesurer objectivement le Recall@5

**Cause 2 :** `ingest.py` n'a pas été relancé après l'ajout d'un document.

**Solution :** `python src/ingest.py`

---

## `ingest.py` se bloque sans afficher d'erreur

**Cause probable :** Ollama plante ou sature en mémoire pendant le calcul des embeddings.

**Solutions :**
1. Vérifie qu'Ollama tourne : `ollama list`
2. Redémarre Ollama depuis la barre des tâches
3. Réduis le `batch_size` dans `ingest.py` (passer de 50 à 20)

---

## `Le processus ne peut pas accéder au fichier chroma.sqlite3`

**Cause :** Un processus Python tourne encore en arrière-plan et utilise le fichier.

**Solution :**
```powershell
Stop-Process -Name python -Force
Remove-Item -Recurse -Force "vector_db"
python src/ingest.py
```

---

## Open WebUI affiche "Connexion impossible" ou pas de modèle disponible

**Cause :** Le serveur FastAPI n'est pas démarré, ou Docker ne trouve pas l'hôte.

**Solutions :**
1. Vérifie que le serveur RAG tourne : `uvicorn src.api:app --host 0.0.0.0 --port 8000`
2. Teste directement l'API : ouvre `http://localhost:8000/v1/models` dans ton navigateur — tu dois voir `{"data":[{"id":"aquila-rag"...}]}`
3. Si Open WebUI tourne dans Docker mais ne trouve pas le serveur : l'URL doit être `http://host.docker.internal:8000/v1` (pas `localhost`)

---

## La recherche sémantique retourne des chunks du mauvais cours

**Cause :** Limitation connue des modèles d'embedding sur du texte mathématique. bge-m3 peut confondre des matières si les formules sont peu lisibles.

**Ce qui compense :** BM25 retrouve les bons chunks par mots-clés exacts. La fusion donne la priorité aux chunks présents dans les deux listes. Vérifie les logs `[Fusion]` — le résultat final devrait être correct même si `[Sémantique]` est bruité.

**Solution durable :** Mesure le Recall@5 avec `eval_retrieval.py`, puis expérimente : augmenter `K_RETRIEVE`, ajuster `BM25_WEIGHT`, ajouter un reranker `sentence-transformers`.

---

## Le conflit git sur `chroma.sqlite3`

**Cause :** `vector_db/` est versionné dans git alors qu'il ne devrait pas l'être.

**Solution :**
```powershell
git rm -r --cached -f vector_db/
```
Puis vérifier que `vector_db/` est bien dans `.gitignore`.

---

## L'environnement virtuel n'est pas activé

**Symptôme :** `uvicorn`, `python`, ou les librairies ne sont pas trouvés. `(venv)` n'apparaît pas au début du terminal.

**Solution :**
```powershell
venv\Scripts\activate
```
