# 06 — Problèmes courants et solutions

## L'IA répond n'importe quoi

**Cause probable :** Le modèle hallucine — il invente une réponse au lieu de s'appuyer sur les documents.

**Solutions :**
1. Vérifie que `ingest.py` a bien été lancé et que `vector_db/` existe
2. Utilise `gemma:latest` plutôt que `gemma:2b` (plus petit et moins fiable)
3. Reformule ta question avec des mots présents dans tes documents

---

## `ModuleNotFoundError: No module named 'langchain.text_splitter'`

**Cause :** La librairie a changé de nom dans les versions récentes.

**Solution :**
```powershell
pip install langchain-text-splitters
```
Et dans le code, remplace :
```python
# Ancien (ne fonctionne plus)
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Nouveau
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

---

## `ingest.py` se bloque sans afficher d'erreur

**Cause probable :** Ollama plante ou sature en mémoire en traitant trop de chunks d'un coup.

**Solutions :**
1. Vérifie qu'Ollama tourne : `ollama list`
2. Redémarre Ollama depuis la barre des tâches
3. Le code traite déjà les chunks par lots de 50 — si ça continue, réduis le `batch_size` dans `ingest.py`

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

## Le conflit git sur `chroma.sqlite3` (fichier rouge `AA`)

**Cause :** Les deux branches git ont chacune leur version du fichier binaire. Git ne peut pas fusionner des fichiers binaires automatiquement.

**Solution :** `vector_db/` doit être dans `.gitignore` et ne jamais être versionné. C'est un fichier généré.

```powershell
git rm -r --cached -f vector_db/
```
Puis ajouter dans `.gitignore` :
```
vector_db/
```

---

## Streamlit ne s'ouvre pas dans le navigateur

**Solution :** Ouvre manuellement `http://localhost:8501` dans ton navigateur.

---

## L'environnement virtuel n'est pas activé

**Symptôme :** Les librairies ne sont pas trouvées, ou `(venv)` n'apparaît pas dans le terminal.

**Solution :**
```powershell
venv\Scripts\activate
```
