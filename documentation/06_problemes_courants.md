# 06 — Problèmes courants et solutions

## L'IA répond "Je ne trouve pas cette information dans les documents fournis"

**Cause 1 :** La définition ou le passage demandé n'est dans aucun des 5 chunks sélectionnés.

**Solutions :**
1. Regarde les logs `[Fusion] Top 5` — est-ce que les bons chunks sont sélectionnés ?
2. Reformule ta question avec des mots présents dans les documents
3. Si BM25 trouve les bons chunks mais sémantique non (visible dans les logs), c'est normal — la fusion RRF devrait quand même les faire remonter

**Cause 2 :** `ingest.py` n'a pas été relancé après l'ajout d'un document.

**Solution :** `python src/ingest.py`

---

## `the input length exceeds the context length` (erreur Ollama embedding)

**Cause :** Un chunk dépasse la fenêtre de contexte du modèle d'embedding. `bge-m3` supporte 8192 tokens, donc cela ne devrait pas arriver avec `chunk_size=1000`. Si l'erreur apparaît, c'est probablement avec `mxbai-embed-large` (512 tokens seulement).

**Solution :** Vérifier que `EMBED_MODEL = "bge-m3"` dans `ingest.py` et `ask.py`.

---

## `ModuleNotFoundError: No module named 'pymupdf4llm'`

**Solution :**
```powershell
pip install pymupdf4llm
```

---

## `ModuleNotFoundError: No module named 'rank_bm25'`

**Solution :**
```powershell
pip install rank-bm25
```

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

**Cause probable :** Ollama plante ou sature en mémoire.

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

## La recherche sémantique retourne des résultats du mauvais cours

**Exemple :** Question sur le calcul différentiel → les scores sémantiques retournent des chunks de statistiques.

**Cause :** C'est une limitation connue des modèles d'embedding sur du texte mathématique. bge-m3 comprend le vocabulaire mathématique général mais peut se tromper de matière si les formules sont peu lisibles.

**Ce qui compense :** La recherche BM25 trouvera les bons chunks par mots-clés exacts. La fusion RRF donnera la priorité aux chunks présents dans les deux listes. Regarde les logs `[Fusion]` — le résultat final devrait être correct même si `[Sémantique]` est bruité.

**Solution durable :** Passer à Python 3.11 ou 3.12 et installer `marker-pdf` pour une extraction PDF avec reconnaissance de formules LaTeX.

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

## `pip install marker-pdf` échoue (Pillow, regex)

**Cause :** Tu utilises Python 3.14, trop récent pour marker-pdf. Pillow 10.4.0 ne supporte pas Python 3.14.

**Solution :** Utiliser `pymupdf4llm` à la place (déjà implémenté dans ce projet). Pour marker-pdf, il faut Python 3.11 ou 3.12.

---

## L'environnement virtuel n'est pas activé

**Symptôme :** Les librairies ne sont pas trouvées, ou `(venv)` n'apparaît pas dans le terminal.

**Solution :**
```powershell
venv\Scripts\activate
```
