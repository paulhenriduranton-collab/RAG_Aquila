"""
Serveur API compatible OpenAI pour brancher le RAG agentique sur Open WebUI.

Ne modifie aucun fichier existant : réutilise simplement ask_question_agentic()
de agent.py et l'expose via les routes que Open WebUI attend
(/v1/models, /v1/chat/completions).

Lancement (depuis le dossier src/) :
    uvicorn api_server:app --host 0.0.0.0 --port 8001

Dans Open WebUI : Réglages > Connexions > Ajouter une connexion OpenAI
    URL de base : http://host.docker.internal:8001/v1
    Clé API     : n'importe quelle valeur (non vérifiée)
"""

import time

from fastapi import FastAPI
from pydantic import BaseModel

from agent import ask_question_agentic

app = FastAPI()

MODEL_NAME = "rag-aquila-agentic"


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "rag-aquila"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    question = req.messages[-1].content  # dernier message envoyé par l'utilisateur
    answer, _ = ask_question_agentic(question, verbose=False)

    return {
        "id": "chatcmpl-rag-aquila",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
    }
