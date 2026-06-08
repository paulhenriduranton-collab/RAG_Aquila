import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))
from agent import ask_question_agentic
data = json.loads((BASE_DIR / "data" / "questions.json").read_text(encoding="utf-8"))

# Choix de la question : par ID en argument (ex: `python test_agent_q1.py L1_ENS_003`),
# sinon la première question du dataset par défaut.
if len(sys.argv) > 1:
    target_id = sys.argv[1]
    matches = [e for e in data if e["id"] == target_id]
    if not matches:
        raise SystemExit(f"Aucune question avec l'ID '{target_id}' dans data/questions.json")
    q = matches[0]
else:
    q = data[0]

print(f"=== Question {q['id']} ===")
print(f"Question : {q['question']}")
print(f"Réponse attendue : {q['reponse_attendue']}")
print(f"Source attendue : {q['source']} ({q['doc']}, {q['pages']})")
print()
print("--- Lancement du RAG agentique ---", flush=True)

answer, docs = ask_question_agentic(q["question"], verbose=True)

print()
print("=== RÉPONSE GÉNÉRÉE ===")
print(answer)
print()
print(f"=== CHUNKS UTILISÉS ({len(docs)}) ===")
for i, doc in enumerate(docs):
    print(f"\n#{i+1}  source={doc.metadata.get('source','?')}  page={doc.metadata.get('page','?')}")
    print(doc.page_content)
