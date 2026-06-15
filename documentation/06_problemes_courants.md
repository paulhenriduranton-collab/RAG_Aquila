# 06 — Problèmes courants et solutions

## `Collection expecting embedding with dimension 768, got 1024`

**Cause :** La base `C:/vector_db_aquila` a été créée avec un modèle d'embedding à 768 dimensions (ex: `nomic-embed-text`), mais le code utilise maintenant `bge-m3` (1024 dimensions). Les deux sont incompatibles.

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
ollama pull bge-m3
ollama pull gemma4:12b
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

# Lancer depuis le dossier src/
cd src
uvicorn api_server:app --host 0.0.0.0 --port 8001
```

---

## L'IA répond "Je ne trouve pas cette information dans les documents fournis"

**Cause 1 :** Le passage demandé n'est dans aucun des chunks sélectionnés après re-ranking.

**Solutions :**
1. Lance `python src/ask.py` et regarde les logs `[Top 5 final]` — les bons chunks sont-ils sélectionnés ?
2. Regarde le score du re-ranker dans les logs `[Re-ranking]` — les chunks sont-ils écartés (score < 0) ?
3. Reformule la question avec des mots présents dans les documents
4. Utilise le pipeline agentique (`python src/agent.py`) — il reformule automatiquement si le retrieval est insuffisant

**Cause 2 :** `ingest.py` n'a pas été relancé après l'ajout d'un document.

**Solution :** `python src/ingest.py`

---

## `ingest.py` se bloque ou crash en cours de route

**Cause probable :** llama-server (le backend d'Ollama) crash sur les longs runs de contextualisation (~700 appels LLM). C'est un comportement connu sur Colab et Windows.

**Ce qui se passe automatiquement :** `_invoke_with_retry` tente jusqu'à 3 fois en redémarrant Ollama entre chaque tentative. Si le crash a lieu entre deux pages, le checkpoint pickle sauvegarde la progression.

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
# Dans src/ask.py, ligne ~57 — augmenter pour moins de redémarrages
QUESTION_RESTART_INTERVAL = 2  # redémarre Ollama toutes les N questions
```

---

## La recherche sémantique retourne des chunks du mauvais cours

**Cause :** Les brochures ENS et Sorbonne ont des sections avec des intitulés similaires (ex: "Organisation", "Calendrier"). bge-m3 peut les confondre si les mots-clés sont proches.

**Ce qui compense :**
1. Le **contextual retrieval** : le préfixe `[source | p.X | ...]` dans chaque chunk précise son établissement — l'embedding le voit
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

## Le conflit git sur `chroma.sqlite3`

**Cause :** `C:/vector_db_aquila` ou un ancien `vector_db/` est versionné dans git alors qu'il ne devrait pas l'être.

**Solution :**
```powershell
# Retirer du suivi git (sans supprimer les fichiers)
git rm -r --cached -f vector_db/
```

Puis vérifier que `vector_db/` et `C:/vector_db_aquila` sont dans `.gitignore`.

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
