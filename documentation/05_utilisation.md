# 05 — Guide d'utilisation

## Prérequis

Avant de commencer, vérifie que tu as :
- **Python** installé (version 3.10 ou plus)
- **Ollama** installé et lancé (visible dans la barre des tâches)
- Les deux modèles téléchargés :
  ```powershell
  ollama pull gemma:latest
  ollama pull nomic-embed-text
  ```

---

## Installation (à faire une seule fois)

### 1. Créer l'environnement virtuel

Un environnement virtuel est un dossier qui contient les librairies Python du projet, séparées du reste de ton ordinateur.

```powershell
python -m venv venv
venv\Scripts\activate
```

Quand l'environnement est actif, tu vois `(venv)` au début de chaque ligne du terminal.

### 2. Installer les dépendances

```powershell
pip install -r requirements.txt
pip install langchain-text-splitters
```

---

## Utilisation normale

### Étape 1 — Ajouter tes documents

Copie tes fichiers (PDF, Word ou TXT) dans le dossier `documents/`.

### Étape 2 — Indexer les documents

```powershell
python src/ingest.py
```

Tu verras s'afficher :
```
Chargement des documents...
210 document(s) chargé(s).
601 chunk(s) créé(s).
Lot 1 / 13 (50 chunks)...
...
Index créé dans vector_db/.
```

**Durée :** 10 à 20 minutes selon ta machine. À ne refaire que si tu ajoutes de nouveaux documents.

### Étape 3 — Poser une question

```powershell
python src/ask.py
```

Le terminal affiche `Question :`. Tape ta question et appuie sur Entrée. Les chunks récupérés s'affichent avec leur score, puis la réponse apparaît. Utilise **Ctrl+C** pour quitter.

---

## Si tu ajoutes de nouveaux documents

1. Copie les nouveaux fichiers dans `documents/`
2. Relance `python src/ingest.py` — cela recrée la base entière
3. Relance `python src/ask.py`

---

## Vérifier qu'Ollama fonctionne

```powershell
ollama list
```

Tu dois voir `nomic-embed-text:latest` et `gemma:latest` dans la liste. Si ce n'est pas le cas :
```powershell
ollama pull nomic-embed-text
ollama pull gemma:latest
```

---

