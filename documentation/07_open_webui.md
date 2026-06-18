# 07 — Lancer l'interface Open WebUI (RAG agentique)

Ce guide explique comment brancher le RAG agentique (`src/agent.py`) sur **Open WebUI**,
une interface de chat type ChatGPT, en local.

Le serveur `src/api_server.py` expose `ask_question_agentic()` via une API compatible OpenAI.
Open WebUI se connecte dessus comme s'il parlait à un serveur OpenAI — aucune modification
des fichiers existants n'est nécessaire.

## Prérequis (à faire une seule fois)

- Avoir installé Open WebUI dans un venv séparé :
  ```powershell
  python -m venv "$env:USERPROFILE\open-webui-venv"
  & "$env:USERPROFILE\open-webui-venv\Scripts\Activate.ps1"
  pip install open-webui
  ```
- Avoir installé les dépendances du projet (inclut `fastapi` et `uvicorn`) :
  ```powershell
  pip install -r requirements.txt
  ```
- Avoir indexé les documents (`python src/ingest.py`) et lancé Ollama
- Avoir les 3 modèles Ollama (`bge-m3`, `gemma2:2b`, `gemma4:12b`)

---

## À chaque session : 2 fenêtres PowerShell à laisser ouvertes

### Fenêtre 1 — Serveur RAG agentique

```powershell
# Depuis le dossier src/ du projet
cd src
uvicorn api_server:app --host 0.0.0.0 --port 8001
```

Ce serveur expose `agent.py` (pipeline agentique : identification de source + difficulté →
retrieval hybride → HyDE → BM25 → RRF → déduplication → re-ranking → évaluation de suffisance →
reformulation si besoin → génération) via une API compatible OpenAI sur le port 8001.

Tu dois voir au lancement :
```
INFO:     Started server process [...]
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
```

### Fenêtre 2 — Open WebUI

```powershell
& "$env:USERPROFILE\open-webui-venv\Scripts\Activate.ps1"
open-webui serve --port 3000
```

Ouvre ensuite **http://localhost:3000** dans ton navigateur.

---

## Connexion (à faire une seule fois)

Dans Open WebUI : **⚙️ Réglages → Connexions → Ajouter une connexion** :

| Champ | Valeur |
|---|---|
| Type | OpenAI |
| URL de base | `http://localhost:8001/v1` |
| Clé API | n'importe quelle valeur, ex : `sk-local` (non vérifiée) |

Le modèle **`rag-aquila-agentic`** apparaît alors dans le sélecteur en haut du chat.

---

## Utilisation

1. Sélectionne le modèle `rag-aquila-agentic`.
2. Pose ta question dans le chat.

⚠️ **Pas de streaming** : la réponse n'apparaît qu'à la toute fin. Compte :
- 30 à 90 secondes pour une question simple (1 retrieval suffisant)
- 2 à 5 minutes pour une question qui déclenche une reformulation de requête

C'est normal — l'agent enchaîne plusieurs appels LLM (identification de source + difficulté,
HyDE, grading, éventuelle reformulation, génération finale).

---

## Tester l'API sans Open WebUI

Tu peux vérifier que le serveur répond correctement depuis le terminal :

```powershell
# Vérifie la liste des modèles disponibles
curl http://localhost:8001/v1/models

# Envoie une question directement (remplace la question par la tienne)
curl -X POST http://localhost:8001/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"rag-aquila-agentic","messages":[{"role":"user","content":"Qui dirige le DMA ?"}]}'
```

---

## Récapitulatif des commandes

| Action | Commande |
|---|---|
| Lancer le serveur RAG | `uvicorn api_server:app --host 0.0.0.0 --port 8001` (depuis `src/`) |
| Lancer Open WebUI | `open-webui serve --port 3000` |
| Interface | http://localhost:3000 |
| Tester l'API | `curl http://localhost:8001/v1/models` |
