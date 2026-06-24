# 04 — Lancer le projet sur Google Colab

## Pourquoi Colab ?

gemma4:12b (12 milliards de paramètres) est lent sur CPU. Sur un GPU Colab (L4 ou A100), la génération est **5 à 10x plus rapide**, ce qui rend le passage des 50 questions viable en quelques heures au lieu d'une journée.

## Lien direct

**https://colab.research.google.com/github/paulhenriduranton-collab/RAG_Aquila/blob/main/colab_run.ipynb**

---

## Avant de commencer

1. **GPU** : Exécution → Modifier le type d'exécution → **GPU** (L4 ou A100 de préférence, T4 si les autres sont indisponibles)
2. **Token GitHub** : le notebook utilise un secret `AH` (token GitHub) stocké dans les secrets Colab (icône clé dans la barre latérale)

---

## Les étapes du notebook

### Étape 1 — Clone du repo

Clone le repo GitHub dans `/content/RAG_Aquila` en utilisant le token `AH` pour l'authentification.

### Étape 2 — Dépendances Python

Installe les librairies nécessaires (langchain, chromadb, etc.) directement dans l'environnement Colab.

### Étape 3 — Ollama + modèles

Installe Ollama sur Colab et télécharge les modèles :
- `bge-m3` (~570 Mo) — embeddings
- `gemma4:12b` (~7 Go) — split agentique des chunks (ingestion), HyDE, grading, génération finale
- `gemma2:2b` (~1.6 Go) — optionnel, utilisé uniquement comme juge RAGAS plus rapide à l'étape 5

Le téléchargement prend ~5-10 minutes la première fois.

### Étape 4 — Ingestion (si nécessaire)

Lance `ingest.py` avec un override du chemin de la base vectorielle :

```python
# Le notebook override VECTOR_DB_DIR pour écrire dans le repo cloné
ingest.VECTOR_DB_DIR = Path("/content/RAG_Aquila/vector_db")
```

**Important :** Le notebook supprime l'ancienne `vector_db/` avant d'ingérer, puis la recrée. `ingest.py` crée automatiquement le dossier (`VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)`) pour éviter l'erreur `SQLITE_CANTOPEN` de ChromaDB.

**Si la `vector_db/` est déjà versionnée et à jour dans le repo**, tu peux sauter cette étape — le clone l'a déjà récupérée.

### Étape 4 — Batch (run_agentic_all.py)

Passe les 50 questions au pipeline agentique. Le notebook :
- Configure git (user, email, token) pour pouvoir pusher automatiquement
- Override `ask.VECTOR_DB_DIR` pour pointer sur la base clonée
- Sauvegarde les résultats après chaque question dans `data/agentic_results.json`
- Push sur GitHub toutes les 5 questions (`PUSH_EVERY = 5`)

**Durée estimée :** 2 à 4 heures sur GPU L4/A100, selon la complexité des questions.

### Étape 4b — Interface Gradio (alternative interactive)

Au lieu du batch, cette cellule lance une **interface web interactive** avec :

1. **Zone de question** — tape ta question manuellement
2. **Réponse** — texte généré par le pipeline agentique
3. **Chunks récupérés** — HTML détaillé avec source, page, score de re-ranking
4. **Pages PDF surlignées** — galerie d'images des pages sources avec le chunk **surligné en bleu**

Le surlignage utilise PyMuPDF pour localiser le texte du chunk dans la page et le marquer en bleu.

Gradio génère un **lien public** `*.gradio.live` (valable 72h) que tu peux partager avec quelqu'un sans qu'il ait besoin de Colab.

### Étape 5 — Évaluation RAGAS

Calcule les 5 métriques RAGAS sur les résultats de l'étape 4. Le notebook :
- Installe `ragas`, `datasets`, `openai`
- Patche l'import `ChatVertexAI` supprimé dans langchain-community 0.4.x
- Override les chemins de `evaluate_ragas.py` pour Colab
- Utilise `gemma2:2b` comme juge RAGAS (3-5x plus rapide que gemma4:12b)

---

## Pousser les résultats sur GitHub depuis Colab

La cellule batch (étape 4) push automatiquement les résultats toutes les 5 questions. Si tu as besoin de pusher manuellement (par exemple après une ingestion) :

```python
import subprocess
from google.colab import userdata

# Récupère le token GitHub depuis les secrets Colab
token = userdata.get('AH')

# Configure git
subprocess.run(["git", "-C", "/content/RAG_Aquila", "config", "user.email", "Benjamin.plrd@icloud.com"], check=True)
subprocess.run(["git", "-C", "/content/RAG_Aquila", "config", "user.name", "Benjamin1234323"], check=True)
subprocess.run(["git", "-C", "/content/RAG_Aquila", "remote", "set-url", "origin",
                f"https://{token}@github.com/paulhenriduranton-collab/RAG_Aquila.git"], check=True)

# Pull + rebase pour intégrer les commits locaux (ex: fix ingest.py poussé depuis ta machine)
subprocess.run(["git", "-C", "/content/RAG_Aquila", "pull", "--rebase"], check=True)

# Add, commit, push
subprocess.run(["git", "-C", "/content/RAG_Aquila", "add", "."], check=True)
subprocess.run(["git", "-C", "/content/RAG_Aquila", "commit", "-m", "feat: résultats Colab"], check=True)
subprocess.run(["git", "-C", "/content/RAG_Aquila", "push"], check=True)
print("Pushé sur GitHub.")
```

**Erreur fréquente :** `rejected ... fetch first` → le remote a des commits que Colab n'a pas. Ajoute `git pull --rebase` avant le push (déjà inclus dans le script ci-dessus).

---

## Contournements spécifiques à Colab

### llama-server "cannot get current path"

Ollama est lancé avec `cwd="/tmp"` au lieu du répertoire courant — contourne un bug de llama-server spécifique à Colab.

### Crash llama-server sur les longs runs

La contextualisation (~700 appels LLM) peut faire crasher llama-server. Le code gère ça :
- `_invoke_with_retry` : retente 3 fois en redémarrant Ollama entre chaque tentative
- Checkpoint pickle : sauvegarde la progression après chaque page
- Relance `ingest.py` : repart du dernier checkpoint automatiquement

### Redémarrage préventif d'Ollama

`ask.py` redémarre Ollama toutes les 2 questions (`QUESTION_RESTART_INTERVAL = 2`) pour purger le KV-cache qui dégrade la latence au fil du temps.

---

## Choix du GPU

| GPU | VRAM | Vitesse relative | Disponibilité |
|---|---|---|---|
| H100 | 80 GB | ⚡⚡⚡⚡ | Rare (Pro+) |
| A100 | 40/80 GB | ⚡⚡⚡ | Limitée (Pro/Pro+) |
| L4 | 24 GB | ⚡⚡ | Bonne |
| T4 | 16 GB | ⚡ | Toujours disponible |

Pour ce projet (gemma4:12b ~8 GB), le **L4** est le meilleur rapport disponibilité/performance. Le T4 fonctionne mais est plus lent.
