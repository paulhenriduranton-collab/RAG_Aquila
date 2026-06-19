# 06 — Problèmes courants et solutions

## `InternalError: Database error: (code: 14) unable to open database file`

**Cause :** Les bindings Rust de ChromaDB (versions récentes) ne créent pas automatiquement le dossier `vector_db/` si celui-ci n'existe pas. SQLite retourne l'erreur `SQLITE_CANTOPEN` (code 14) quand il ne peut pas ouvrir le fichier `.sqlite3` dans un dossier inexistant.

**Ce qui arrive typiquement :** Sur Colab, la cellule supprime l'ancienne `vector_db/` avec `shutil.rmtree`, puis `ingest.main()` essaie de créer une base ChromaDB dans ce dossier supprimé.

**Solution (déjà corrigée dans le code) :** `ingest.py` crée le dossier avant d'appeler ChromaDB :
```python
# Ligne 354 de ingest.py
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
```

Si l'erreur réapparaît, vérifie que tu as bien la dernière version du code (`git pull`).

---

## `Collection expecting embedding with dimension 768, got 1024`

**Cause :** La base vectorielle a été créée avec un modèle d'embedding à 768 dimensions (ex: `nomic-embed-text`), mais le code utilise maintenant `bge-m3` (1024 dimensions). Les deux sont incompatibles.

**Solution :**
```powershell
# Supprimer l'ancienne base puis réindexer
Remove-Item -Recurse -Force "C:\vector_db_aquila"
python src/ingest.py
```

---

## `model "bge-m3" not found` ou `model "gemma4:12b" not found`

**Cause :** Le modèle n'est pas encore téléchargé dans Ollama.

**Solution :**
```powershell
ollama pull bge-m3      # embeddings (indexation + recherche)
ollama pull gemma4:12b  # split agentique (ingestion), HyDE, grading, génération finale
```

`gemma2:2b` n'est nécessaire qu'optionnellement sur Colab (juge RAGAS plus rapide) — inutile en local.

---

## `uvicorn` n'est pas reconnu

**Cause :** L'environnement virtuel n'est pas activé, ou uvicorn n'est pas installé.

**Solution :**
```powershell
# Activer le venv
venv\Scripts\activate

# Vérifier que uvicorn est installé
pip install "uvicorn[standard]"

# Lancer depuis le dossier src/
cd src
uvicorn api_server:app --host 0.0.0.0 --port 8001
```

---

## L'IA répond "Je ne trouve pas cette information dans les documents fournis"

**Cause 1 :** Le passage demandé n'est dans aucun des chunks sélectionnés après re-ranking.

**Solutions :**
1. Lance `python src/ask.py` et regarde les logs `[Top 5 final]` — les bons chunks sont-ils sélectionnés ?
2. Regarde le score du re-ranker dans les logs `[Re-ranking]` — les chunks sont-ils écartés (score < 0.5) ?
3. Reformule la question avec des mots présents dans les documents
4. Utilise le pipeline agentique (`python src/agent.py`) — il reformule automatiquement si le retrieval est insuffisant
5. Utilise `python src/debug_question.py` pour analyser le retrieval en détail

**Cause 2 :** `ingest.py` n'a pas été relancé après l'ajout d'un document.

**Solution :** `python src/ingest.py`

---

## `ingest.py` se bloque ou crash en cours de route

**Cause probable :** llama-server (le backend d'Ollama) crash sur les longs runs du split agentique (un appel LLM par section thématique, sur les deux brochures). C'est un comportement connu sur Colab et Windows.

**Ce qui se passe automatiquement :** `_invoke_with_retry` tente jusqu'à 3 fois en redémarrant Ollama entre chaque tentative. Si le crash a lieu entre deux documents, le checkpoint pickle sauvegarde la progression (au niveau du document source, pas de la page).

**Solutions :**
1. Relance `python src/ingest.py` — il repart du checkpoint automatiquement
2. Vérifie qu'Ollama tourne : `ollama list`
3. Si Ollama est complètement bloqué, redémarre-le depuis la barre des tâches puis relance

**Si le checkpoint est corrompu :**
```powershell
# Supprimer le checkpoint pour repartir de zéro
Remove-Item "C:\ingest_checkpoint.pkl"
```

---

## `Le processus ne peut pas accéder au fichier chroma.sqlite3`

**Cause :** Un processus Python tourne encore en arrière-plan et utilise le fichier de base de données.

**Solution :**
```powershell
# Tue tous les processus Python en cours
Stop-Process -Name python -Force

# Supprimer la base et réindexer
Remove-Item -Recurse -Force "C:\vector_db_aquila"
python src/ingest.py
```

---

## Open WebUI affiche "Connexion impossible" ou pas de modèle disponible

**Cause :** Le serveur FastAPI n'est pas démarré, ou Open WebUI ne trouve pas l'hôte.

**Solutions :**
1. Vérifie que le serveur RAG tourne :
   ```powershell
   cd src
   uvicorn api_server:app --host 0.0.0.0 --port 8001
   ```
2. Teste directement l'API — ouvre `http://localhost:8001/v1/models` dans ton navigateur. Tu dois voir :
   ```json
   {"object":"list","data":[{"id":"rag-aquila-agentic","object":"model","owned_by":"rag-aquila"}]}
   ```
3. Si Open WebUI tourne dans Docker mais ne trouve pas le serveur : l'URL doit être `http://host.docker.internal:8001/v1` (pas `localhost`)

---

## La latence augmente fortement au fil des questions

**Cause :** Le KV-cache de llama-server s'accumule entre les appels et n'est pas libéré. Latences observées sur 9 questions sans redémarrage : 101s → 1211s.

**Solution automatique :** Le code redémarre Ollama toutes les 2 questions (`QUESTION_RESTART_INTERVAL = 2` dans `ask.py`). Ce comportement est actif par défaut.

**Si tu veux ajuster :**
```python
# Dans src/ask.py, ligne ~64 — augmenter pour moins de redémarrages
QUESTION_RESTART_INTERVAL = 2  # redémarre Ollama toutes les N questions
```

---

## La recherche sémantique retourne des chunks du mauvais cours

**Cause :** Les brochures ENS et Sorbonne ont des sections avec des intitulés similaires (ex: "Organisation", "Calendrier"). bge-m3 peut les confondre si les mots-clés sont proches.

**Ce qui compense :**
1. Le **contextual retrieval** : le préfixe `[ENS.pdf | p.X | ...]` dans chaque chunk précise son établissement — l'embedding le voit
2. BM25 retrouve les chunks par mots-clés exacts de l'établissement mentionné
3. Le **pipeline agentique** : `identify_sources` choisit le bon fichier avant de lancer le retrieval

**Si le problème persiste :** Lance `python src/agent.py` au lieu d'`ask.py` — l'agent filtre la recherche sur le bon document.

---

## L'évaluation RAGAS échoue avec `RuntimeError: Event loop is already running`

**Cause :** Jupyter / Colab a déjà une event loop asyncio active, incompatible avec `asyncio.run()`.

**Solution automatique :** `evaluate_ragas.py` détecte ce cas et utilise `nest_asyncio` :
```python
# Déjà géré dans _run_scoring() — pas d'action nécessaire
import nest_asyncio
nest_asyncio.apply()
```

Si l'erreur persiste, vérifie que `nest_asyncio` est installé :
```powershell
pip install nest_asyncio
```

---

## `rejected ... fetch first` lors du push depuis Colab

**Cause :** Le remote GitHub a des commits que le clone Colab n'a pas (par exemple un fix poussé depuis ta machine locale).

**Solution :** Ajouter `git pull --rebase` avant le push :
```python
# Pull les commits distants et rejoue les commits locaux par-dessus
subprocess.run(["git", "-C", "/content/RAG_Aquila", "pull", "--rebase"], check=True)
# Puis push
subprocess.run(["git", "-C", "/content/RAG_Aquila", "push"], check=True)
```

---

## L'environnement virtuel n'est pas activé

**Symptôme :** `uvicorn`, `python`, ou les librairies ne sont pas trouvés. `(venv)` n'apparaît pas au début du terminal.

**Solution :**
```powershell
venv\Scripts\activate
```

---

## `ragas` ou `langgraph` non trouvés

**Cause :** Les dépendances n'ont pas été installées, ou le venv n'est pas activé.

**Solution :**
```powershell
venv\Scripts\activate
pip install -r requirements.txt
```
