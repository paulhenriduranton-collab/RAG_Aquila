# 05 — Guide d'utilisation

## Prérequis

- **Python** 3.10 à 3.13 (pas 3.14)
- **Ollama** installé et lancé (visible dans la barre des tâches)
- **Docker Desktop** installé (pour Open WebUI)
- Les deux modèles Ollama téléchargés :
  ```powershell
  ollama pull bge-m3
  ollama pull gemma2:2b
  ```

---

## Installation (à faire une seule fois)

### 1. Activer l'environnement virtuel

```powershell
venv\Scripts\activate
```

Quand l'environnement est actif, tu vois `(venv)` au début de chaque ligne du terminal.

### 2. Installer les dépendances

```powershell
pip install -r requirements.txt
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
  ✓ ENS.pdf (1 doc(s))
  ✓ SORBONNE.pdf (1 doc(s))

2 document(s) chargé(s).
708 chunk(s) créé(s).
Lot 1 / 15 (50 chunks)...
...
Index créé dans vector_db/.
```

**Durée :** 15 à 30 minutes selon ta machine (calcul des embeddings bge-m3).
**À ne refaire que si tu ajoutes ou modifies des documents.**

### Étape 3a — Lancer l'interface Open WebUI (recommandé)

**Terminal 1 — démarrer le serveur RAG :**
```powershell
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
Attends le message `Application startup complete.`

**Terminal 2 — démarrer Open WebUI :**
```powershell
docker compose up -d
```

Ouvre **http://localhost:3000** dans ton navigateur. Sélectionne le modèle `aquila-rag` et pose ta question.

### Étape 3b — Mode terminal (pour déboguer)

```powershell
python src/ask.py
```

Ce mode affiche les logs détaillés de la recherche à chaque question :
```
[DB] 708 chunks dans la base

[Sémantique] Top 5 :
  #1  score=0.721  ENS.pdf  p.5
      ↳ Calcul Différentiel I. Différentielles d'ordre supérieur...

[BM25] Top 5 :
  #1  bm25=16.03  ENS.pdf  p.5
      ↳ Calcul Différentiel I. Différentielles d'ordre supérieur...

[Fusion] Top 5 après fusion :
  score=1.2210  ENS.pdf  p.5
  ...

Réponse :
La différentielle d'ordre 2 est définie comme...
```

Utilise **Ctrl+C** pour quitter.

---

## Évaluation de la qualité du retrieval

### Générer le dataset de test (une fois)

```powershell
python evaluation/generate_dataset.py
```

Génère ~60 paires (question, chunk de référence) dans `evaluation/dataset.json`.

**Ouvre ensuite le fichier** et supprime les questions de mauvaise qualité. Conserve 25 à 40 questions.

### Mesurer les métriques

```powershell
# Avec les paramètres actuels
python evaluation/eval_retrieval.py

# Tester un changement de paramètre
python evaluation/eval_retrieval.py --k-retrieve 30 --bm25-weight 0.3
```

Affiche Recall@1, @3, @5, @10 et MRR. Compare avant/après chaque modification du pipeline.

---

## Si tu ajoutes de nouveaux documents

```powershell
# 1. Supprimer l'ancienne base (obligatoire)
Remove-Item -Recurse -Force vector_db

# 2. Réindexer
python src/ingest.py

# 3. Régénérer le dataset d'évaluation (optionnel)
python evaluation/generate_dataset.py
```

**Pourquoi supprimer `vector_db/` ?** Si tu gardes l'ancienne base et ajoutes de nouveaux chunks, les vecteurs coexistent mais le dataset d'évaluation devient incohérent (les chunk_ids ne correspondent plus).

---

## Vérifier qu'Ollama fonctionne

```powershell
ollama list
```

Tu dois voir `bge-m3:latest` et `gemma2:2b` dans la liste. Sinon :
```powershell
ollama pull bge-m3
ollama pull gemma2:2b
```

---

## Arrêter les services

```powershell
# Arrêter Open WebUI
docker compose down

# Arrêter le serveur RAG
# Ctrl+C dans le terminal uvicorn
```
