# 05 — Guide d'utilisation

## Prérequis

- **Python 3.10 à 3.13** (pas 3.14+ — incompatibilité Pillow)
- **Ollama** installé et lancé (visible dans la barre des tâches ou via `ollama serve`)
- Les deux modèles Ollama téléchargés :
  ```powershell
  ollama pull bge-m3       # modèle d'embedding multilingue
  ollama pull gemma4:12b   # LLM de génération et d'évaluation
  ```

---

## Installation (à faire une seule fois)

### Créer l'environnement virtuel

```powershell
python -m venv venv
venv\Scripts\activate
```

### Installer les dépendances

```powershell
pip install -r requirements.txt
```

Le re-ranker CrossEncoder (~471 Mo) sera téléchargé automatiquement depuis HuggingFace au premier lancement. Ollama lui reste nécessaire pour les modèles LLM.

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
  ✓ Brochure-2024-2025.pdf (42 doc(s))
  ✓ Brochure Master2526_1.pdf (38 doc(s))

80 document(s) chargé(s).
  [1/80] Brochure-2024-2025.pdf p.1 → table des matières ignorée
  [2/80] Brochure-2024-2025.pdf p.2 → 3 chunk(s) contextualisé(s)
  ...
718 chunk(s) créé(s).
Lot 1 / 15 (50 chunks)...
...
Index créé dans vector_db/.
```

**Durée :** La contextualisation LLM (étape 5) prend 1 à 3 secondes par chunk via Ollama, soit **20 à 60 minutes** selon ta machine. Sur GPU Colab : 5 à 15 minutes.

**À ne refaire que si tu ajoutes ou modifies des documents.**

Si le run s'interrompt (crash llama-server), relance simplement `python src/ingest.py` — la reprise repart du dernier checkpoint automatiquement.

### Étape 3 — Lancer l'interface chat (Open WebUI)

**Fenêtre PowerShell 1 — Serveur RAG agentique :**
```powershell
cd src
uvicorn api_server:app --host 0.0.0.0 --port 8001
```

**Fenêtre PowerShell 2 — Open WebUI :**
```powershell
# Si Open WebUI est installé dans un venv séparé
& "$env:USERPROFILE\open-webui-venv\Scripts\Activate.ps1"
open-webui serve --port 3000
```

Ouvre **http://localhost:3000**, sélectionne le modèle `rag-aquila-agentic` et pose ta question.

⚠️ **Pas de streaming** : la réponse n'apparaît qu'à la toute fin, après plusieurs dizaines de secondes à plusieurs minutes selon la complexité de la question.

---

## Mode terminal (pour déboguer)

### RAG classique (ask.py)

```powershell
python src/ask.py
```

Affiche les logs détaillés de chaque étape :
```
[DB] 718 chunks dans la base

[HyDE] Génération de la réponse hypothétique...
[HyDE] Réponse fictive : Les quatre cours communs obligatoires de L3 sont...

[Sémantique] Recherche MMR des 20 plus proches voisins diversifiés...
[Sémantique] Top 5 :
  #1  Brochure-2024-2025.pdf  p.12
      ↳ [Brochure-2024-2025.pdf | p.12 | DMA > L3 | cours, ECTS, obligatoires]...

[BM25] Top 5 résultats lexicaux :
  #1  bm25=16.03  Brochure-2024-2025.pdf  p.12

[RRF] Top 10 après fusion sémantique + BM25 :
  rrf=0.0300  Brochure-2024-2025.pdf  p.12
  ...

[Re-ranking] Scores (seuil = 0.0) :
  score=8.412  Brochure-2024-2025.pdf  p.12
  score=-1.203  Brochure Master2526_1.pdf  p.7  ← écarté (hors-sujet)

[Top 5 final] (3 chunk(s) au-dessus du seuil) :
  #1  Brochure-2024-2025.pdf  p.12

Réponse :
Les quatre cours obligatoires sont Algèbre 1, Analyse complexe...
```

### RAG agentique (agent.py)

```powershell
python src/agent.py
```

Affiche les décisions de l'agent en plus des logs de retrieval :
```
[Agent] Source(s) ciblée(s) : Brochure-2024-2025.pdf
[Agent] Tentative(s) de retrieval : 1
[Agent] Chunks jugés suffisants : oui
```

⚠️ Chaque question peut prendre **plusieurs minutes** — plusieurs appels LLM s'enchaînent.

Utilise **Ctrl+C** pour quitter les deux modes.

---

## Évaluation de la qualité

### Étape 1 — Passer le dataset au pipeline agentique

```powershell
python src/run_agentic_all.py
```

Lance les 40 questions séquentiellement. Durée estimée : **2 à 4 heures** (selon GPU/CPU). Les résultats sont sauvegardés dans `data/agentic_results.json` après **chaque question** — tu peux faire Ctrl+C à tout moment sans perdre la progression.

### Étape 2 — Calculer les métriques RAGAS

```powershell
python src/evaluate_ragas.py
```

Durée estimée : **30 à 60 minutes** (5 appels LLM par question × 40 questions). Les scores sont sauvegardés dans `data/ragas_evaluation.csv`.

### Interpréter les résultats

```
faithfulness        → proche de 1.0 = pas d'hallucination
answer_relevancy    → proche de 1.0 = réponse pertinente
context_precision   → proche de 1.0 = bons chunks récupérés
context_recall      → proche de 1.0 = toutes les infos nécessaires récupérées
answer_correctness  → proche de 1.0 = réponse factuellement correcte
```

---

## Sur Google Colab (GPU recommandé)

Ouvre le notebook `colab_run.ipynb` depuis :

**https://colab.research.google.com/github/paulhenriduranton-collab/RAG_Aquila/blob/main/colab_run.ipynb**

Avant de lancer : **Exécution → Modifier le type d'exécution → GPU (T4 ou A100)**.

---

## Si tu ajoutes de nouveaux documents

```powershell
# 1. Supprimer l'ancienne base (les nouveaux vecteurs seraient incohérents avec les anciens)
Remove-Item -Recurse -Force "C:\vector_db_aquila"

# 2. Réindexer avec les nouveaux documents
python src/ingest.py
```

---

## Vérifier qu'Ollama fonctionne

```powershell
ollama list
```

Tu dois voir `bge-m3:latest` et `gemma4:12b` dans la liste. Sinon :
```powershell
ollama pull bge-m3
ollama pull gemma4:12b
```

---

## Récapitulatif des commandes

| Action | Commande |
|---|---|
| Indexer les documents | `python src/ingest.py` |
| Interface chat (serveur) | `uvicorn api_server:app --host 0.0.0.0 --port 8001` (depuis `src/`) |
| Interface chat (UI) | `open-webui serve --port 3000` |
| Mode terminal RAG classique | `python src/ask.py` |
| Mode terminal agentique | `python src/agent.py` |
| Batch évaluation | `python src/run_agentic_all.py` |
| Métriques RAGAS | `python src/evaluate_ragas.py` |
