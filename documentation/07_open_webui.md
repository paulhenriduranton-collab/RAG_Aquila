# 07 — Lancer l'interface Open WebUI (RAG agentique)

Ce guide explique comment brancher le RAG agentique (`src/agent.py`) sur **Open WebUI**,
une interface de chat type ChatGPT, en local.

## Prérequis (à faire une seule fois)

- Avoir installé Open WebUI :
  ```powershell
  python -m venv "$env:USERPROFILE\open-webui-venv"
  & "$env:USERPROFILE\open-webui-venv\Scripts\Activate.ps1"
  pip install open-webui
  ```
- Avoir installé les dépendances du projet (inclut `fastapi` et `uvicorn`) :
  ```powershell
  pip install -r requirements.txt
  ```

---

## À chaque session : 2 fenêtres PowerShell à laisser ouvertes

### Fenêtre 1 — Serveur RAG agentique

```powershell
cd "Projet Aquila"
venv\Scripts\Activate.ps1
cd src
uvicorn api_server:app --host 0.0.0.0 --port 8001
```

Ce serveur expose `src/agent.py` (pipeline agentique : identification de source →
retrieval hybride → vérification → reformulation si besoin → génération) via une
API compatible OpenAI, sans rien modifier au projet existant.

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
| Clé API | n'importe quelle valeur, ex : `sk-local` |

Le modèle **`rag-aquila-agentic`** apparaît alors dans le sélecteur en haut du chat.

---

## Utilisation

1. Sélectionne le modèle `rag-aquila-agentic`.
2. Pose ta question dans le chat.

⚠️ **Pas de streaming** : la réponse n'apparaît qu'à la toute fin, et peut prendre
**plusieurs minutes** car l'agent enchaîne plusieurs appels LLM (identification de
source, retrieval, vérification, éventuelle reformulation, génération).

---

## Récapitulatif des commandes

| Action | Commande |
|---|---|
| Lancer le serveur RAG | `uvicorn api_server:app --host 0.0.0.0 --port 8001` (depuis `src/`) |
| Lancer Open WebUI | `open-webui serve --port 3000` |
| Interface | http://localhost:3000 |
