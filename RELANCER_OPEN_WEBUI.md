# Relancer Open WebUI + RAG agentique

À faire à chaque fois que tu redémarres ton PC ou fermes les fenêtres PowerShell.
La connexion Open WebUI ↔ RAG est déjà configurée, pas besoin de la refaire.

## 1. Vérifier qu'Ollama tourne

Ollama doit être lancé (icône dans la barre des tâches). Sinon, lance-le depuis le menu Démarrer.

## 2. Fenêtre PowerShell n°1 — Serveur RAG agentique

```powershell
cd "c:\Users\paulh\OneDrive\Bureau\Projet Aquila"
venv\Scripts\Activate.ps1
cd src
uvicorn api_server:app --host 0.0.0.0 --port 8001
```

Laisse cette fenêtre ouverte.

## 3. Fenêtre PowerShell n°2 — Open WebUI

```powershell
& "$env:USERPROFILE\open-webui-venv\Scripts\Activate.ps1"
open-webui serve --port 3000
```

Laisse cette fenêtre ouverte aussi.

⚠️ Si tu obtiens une erreur `[Errno 10048] ... bind on address ('0.0.0.0', 3000)`, c'est qu'une instance précédente tourne déjà en arrière-plan — c'est normal, ouvre directement http://localhost:3000.

## 4. Utiliser

Ouvre **http://localhost:3000**, connecte-toi avec ton compte, sélectionne le modèle **`rag-aquila-agentic`** et pose ta question.

⚠️ Pas de streaming — la réponse peut prendre **plusieurs minutes**.
