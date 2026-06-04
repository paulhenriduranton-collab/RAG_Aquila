# 05 — Guide d'utilisation

## Prérequis

- **Python 3.14** (version utilisée dans ce projet)
- **Ollama** installé et lancé (visible dans la barre des tâches)
- Les deux modèles Ollama téléchargés :
  ```powershell
  ollama pull bge-m3
  ollama pull gemma2:2b
  ```

---

## Installation (à faire une seule fois)

### Installer les dépendances

```powershell
pip install -r requirements.txt
```

Le re-ranker CrossEncoder (~471 Mo) sera téléchargé automatiquement depuis HuggingFace au premier lancement.

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
  ✓ Brochure-2024-2025.pdf (1 doc(s))
  ✓ Brochure Master2526_1.pdf (1 doc(s))

2 document(s) chargé(s).
718 chunk(s) créé(s).
Lot 1 / 15 (50 chunks)...
...
Index créé dans vector_db/.
```

**Durée :** 15 à 30 minutes selon ta machine (calcul des embeddings bge-m3).
**À ne refaire que si tu ajoutes ou modifies des documents.**

### Étape 3 — Lancer l'interface Streamlit

```powershell
python -m streamlit run src/app.py --server.headless true --server.fileWatcherType none
```

Ouvre **http://localhost:8501** dans ton navigateur. Tape ta question et clique sur Envoyer.

L'option `--server.headless true` est nécessaire car Python 3.14 a un comportement différent avec Streamlit.
L'option `--server.fileWatcherType none` supprime les warnings de `torchvision`.

### Mode terminal (pour déboguer)

```powershell
python src/ask.py
```

Ce mode affiche les logs détaillés de la recherche à chaque question :
```
[DB] 718 chunks dans la base

[Sémantique] Recherche des 20 plus proches voisins...
[Sémantique] Top 5 :
  #1  score=0.721  Brochure-2024-2025.pdf  p.?
      ↳ Les quatre cours communs obligatoires sont...

[BM25] Top 5 résultats lexicaux :
  #1  bm25=16.03  Brochure-2024-2025.pdf  p.?
      ↳ Les quatre cours communs obligatoires sont...

[RRF] Top 10 après fusion sémantique + BM25 :
  rrf=0.0300  Brochure-2024-2025.pdf  p.?
  ...

[Reranker] Notation des 10 candidats...

[Top 5 final] :
  #1  Brochure-2024-2025.pdf  p.?
      ↳ Les quatre cours communs obligatoires sont...

Réponse :
Les quatre cours obligatoires sont Algèbre 1, Analyse complexe...
```

Utilise **Ctrl+C** pour quitter.

---

## Évaluation de la qualité

### Lancer l'évaluation complète (40 questions)

```powershell
python src/evaluate.py
```

Durée estimée : **30 à 60 minutes** (200 appels LLM au total).

Les résultats sont sauvegardés dans `data/results.json` après **chaque question** — tu peux faire Ctrl+C à tout moment sans perdre les résultats déjà calculés.

### Lancer sur quelques questions (test rapide)

Ouvre `src/evaluate.py` ligne 168 et ajoute `[:4]` :

```python
dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))[:4]
```

Retire le `[:4]` pour relancer sur les 40 questions.

### Interpréter les résultats

```
Faithfulness    → proche de 1.0 = pas d'hallucination
Answer Relevancy → proche de 1.0 = réponse pertinente
Context Quality  → proche de 1.0 = bons chunks récupérés
Context Recall   → proche de 1.0 = toutes les infos nécessaires récupérées
Answer Correct.  → proche de 1.0 = réponse factuellement correcte
```

---

## Si tu ajoutes de nouveaux documents

```powershell
# 1. Supprimer l'ancienne base (obligatoire car les vecteurs seraient incohérents)
Remove-Item -Recurse -Force vector_db

# 2. Réindexer avec les nouveaux documents
python src/ingest.py
```

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

## Récapitulatif des commandes

| Action | Commande |
|---|---|
| Indexer les documents | `python src/ingest.py` |
| Lancer l'interface web | `python -m streamlit run src/app.py --server.headless true --server.fileWatcherType none` |
| Mode terminal (debug) | `python src/ask.py` |
| Évaluer le pipeline | `python src/evaluate.py` |
