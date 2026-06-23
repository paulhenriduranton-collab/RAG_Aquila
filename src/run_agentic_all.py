import contextlib
import io
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # nécessaire sur Windows, pas disponible dans Jupyter
except AttributeError:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import run_agent

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "questions.json"
OUTPUT_PATH = DATASET_PATH.parent / "agentic_results.json"


def _doc_to_dict(doc) -> dict:
    return {
        "source": doc.metadata.get("source", "?"),
        "page": doc.metadata.get("page", "?"),
        "rerank_score": doc.metadata.get("rerank_score"),
        "content": doc.page_content,
    }
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
    question, réponse attendue/générée, chunks récupérés (à chaque étape) et logs de l'agent.
    Capture aussi tous les états intermédiaires (router, grading, rewrite) nécessaires à
    evaluate_components.py, pour ne plus jamais avoir à refaire tourner le pipeline."""
    question = entry["question"]
    print(f"\n{'='*60}")
    print(f"[{entry['id']}] {question}")
    print("=" * 60)

    buffer = io.StringIO()
    start = time.time()
    with contextlib.redirect_stdout(_Tee(sys.stdout, buffer)):
        final_state = run_agent(question, verbose=True)
    duration = time.time() - start

    docs = final_state["docs"]
    rewrite_triggered = final_state["attempts"] >= 2

    # Affiche les chunks sélectionnés avec leur score de re-ranking
    print(f"\n[Chunks sélectionnés] {len(docs)} chunk(s) :")
    for i, doc in enumerate(docs):
        score = doc.metadata.get("rerank_score")
        score_str = f"{score:.4f}" if score is not None else "n/a"
        print(f"  #{i+1}  score={score_str}  {doc.metadata.get('source','?')}  p.{doc.metadata.get('page','?')}")

    return {
        "id": entry["id"],
        "question": question,
        "reponse_attendue": entry["reponse_attendue"],
        "reponse_llm": final_state["answer"],
        "duree_secondes": round(duration, 1),
        "logs": buffer.getvalue().strip(),
        "chunks": [_doc_to_dict(d) for d in docs],
        # --- états intermédiaires, pour evaluate_components.py ---
        "router": {
            "sources": final_state["sources"],
            "difficulty": final_state["initial_difficulty"],
        },
        "pre_rerank_docs": [_doc_to_dict(d) for d in final_state["pre_rerank_docs"]],
        "post_rerank_docs": [_doc_to_dict(d) for d in (
            final_state["docs_before_rewrite"] if rewrite_triggered else docs
        )],
        "grading": {
            # Le grading n'a réellement lieu que pour difficulté > 1 (cf. _route_after_retrieve) ;
            # pour difficulté 1, sufficient/grade_verdict gardent leur valeur initiale (non significative).
            "performed": final_state["initial_difficulty"] > 1,
            "sufficient": final_state["sufficient"],
            "verdict": final_state["grade_verdict"],
        },
        "rewrite": {
            "triggered": rewrite_triggered,
            "new_query": final_state["current_query"] if rewrite_triggered else "",
        },
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
