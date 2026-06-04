"""
Serveur API compatible OpenAI — pipeline RAG Aquila.
Permet à Open WebUI (ou tout client OpenAI) d'interroger le RAG local.

Démarrage :
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

import json
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_ollama import OllamaLLM
from pydantic import BaseModel

# Import du pipeline RAG depuis ask.py
sys.path.insert(0, str(Path(__file__).parent))
from ask import GEN_MODEL, PROMPT_PATH, retrieve  # noqa: E402

app = FastAPI(title="Aquila RAG API")

# CORS ouvert : Open WebUI tourne sur un port différent
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm      = OllamaLLM(model=GEN_MODEL, num_ctx=4096, temperature=0)
MODEL_ID = "aquila-rag"  # nom affiché dans Open WebUI

NO_ANSWER = "Je ne trouve pas cette information dans les documents fournis."


# ── Schémas Pydantic (format OpenAI) ─────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[Message]
    stream: bool = True  # Open WebUI demande le streaming par défaut


# ── Helpers SSE (Server-Sent Events) ─────────────────────────────────────────

def _sse(chunk_id: str, created: int, delta: dict, finish_reason: str | None = None) -> str:
    # Format attendu par tous les clients compatibles OpenAI
    payload = {
        "id":      chunk_id,
        "object":  "chat.completion.chunk",
        "created": created,
        "model":   MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_response(text_or_generator) -> StreamingResponse:
    """Enveloppe un texte fixe ou un générateur de tokens dans une réponse SSE."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created  = int(time.time())

    def _generate():
        # Delta initial avec le rôle (requis par le protocole OpenAI streaming)
        yield _sse(chunk_id, created, {"role": "assistant", "content": ""})
        if isinstance(text_or_generator, str):
            yield _sse(chunk_id, created, {"content": text_or_generator})
        else:
            for token in text_or_generator:
                yield _sse(chunk_id, created, {"content": token})
        yield _sse(chunk_id, created, {}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


def _non_stream_response(content: str) -> dict:
    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage":   {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/v1/models")
def list_models():
    # Open WebUI appelle cet endpoint pour peupler sa liste de modèles
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "aquila"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    # Extraire la dernière question de l'utilisateur (le reste est l'historique)
    user_msgs = [m for m in req.messages if m.role == "user"]
    if not user_msgs:
        return _non_stream_response("Aucune question détectée.")

    question = user_msgs[-1].content

    # Retrieval sans logs verbeux (les logs sont utiles en terminal, pas en API)
    docs = retrieve(question, verbose=False)

    if not docs:
        return _stream_response(NO_ANSWER) if req.stream else _non_stream_response(NO_ANSWER)

    # Assemblage du contexte : chunks + leur fichier source
    context = "\n\n---\n\n".join(
        f"Source : {doc.metadata.get('source', '?')}\n{doc.page_content}"
        for doc in docs
    )
    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(question=question, context=context)

    if req.stream:
        return _stream_response(llm.stream(prompt))

    return _non_stream_response(llm.invoke(prompt))
