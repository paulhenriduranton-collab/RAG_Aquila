import contextlib
import io
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import ask_question_agentic

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "questions.json"
OUTPUT_PATH = DATASET_PATH.parent / "agentic_results.json"


class _Tee(io.TextIOBase):
    """Écrit en même temps sur la console et dans un buffer, pour capturer les logs [Agent]
    affichés par ask_question_agentic(verbose=True) tout en les gardant visibles à l'écran."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for stream in self._streams:
            stream.write(s)
        return len(s)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def run_question(entry: dict) -> dict:
    """Lance le RAG agentique sur une question et regroupe tout ce qu'on veut garder :
    question, réponse attendue/générée, chunks récupérés et logs de l'agent."""
    question = entry["question"]
    print(f"\n{'='*60}")
    print(f"[{entry['id']}] {question}")
    print("=" * 60)

    buffer = io.StringIO()
    start = time.time()
    with contextlib.redirect_stdout(_Tee(sys.stdout, buffer)):
        answer, docs = ask_question_agentic(question, verbose=True)
    duration = time.time() - start

    return {
        "id": entry["id"],
        "question": question,
        "reponse_attendue": entry["reponse_attendue"],
        "reponse_llm": answer,
        "duree_secondes": round(duration, 1),
        "logs": buffer.getvalue().strip(),
        "chunks": [
            {
                "source": doc.metadata.get("source", "?"),
                "page": doc.metadata.get("page", "?"),
                "content": doc.page_content,
            }
            for doc in docs
        ],
    }


def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    # Reprend là où on s'était arrêté (utile si le run de nuit est interrompu)
    if OUTPUT_PATH.exists():
        results = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        already_done = {r["id"] for r in results}
    else:
        results = []
        already_done = set()

    to_run = [e for e in dataset if e["id"] not in already_done]

    print("=== RAG agentique — passage complet sur le dataset ===")
    print(f"Dataset      : {DATASET_PATH.name} ({len(dataset)} questions au total)")
    print(f"Déjà traités : {len(already_done)} — À traiter : {len(to_run)}")
    print(f"Résultats    : {OUTPUT_PATH}")

    try:
        for entry in to_run:
            result = run_question(entry)
            results.append(result)
            # Sauvegarde après chaque question : un crash ou un Ctrl+C en pleine nuit ne perd rien
            OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  → sauvegardé ({len(results)}/{len(dataset)})")
    except KeyboardInterrupt:
        print("\n\nInterrompu — résultats partiels sauvegardés.")

    print(f"\nTerminé. {len(results)} question(s) traitée(s). Résultats dans {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
