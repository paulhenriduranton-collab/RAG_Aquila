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
# Fichier séparé, dédié à evaluate_components.py — contient les états intermédiaires complets
# (router, grading, rewrite, chunks avant/après re-ranking) pour ne jamais avoir à relancer
# le pipeline. Tenu à part car bien plus volumineux (contenu texte des chunks dupliqué 2-3 fois).
DEBUG_OUTPUT_PATH = DATASET_PATH.parent / "agentic_results_debug.json"


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


def _print_summary(final_state: dict) -> None:
    """Affiche le récapitulatif [Agent] (difficulté, source(s), tentatives, suffisance) — repris
    de ask_question_agentic(), perdu depuis que run_question() appelle run_agent() directement."""
    diff_labels = {1: "factuel", 2: "synthèse", 3: "complexe"}
    init_d = final_state["initial_difficulty"]
    final_d = final_state["difficulty"]
    upgrade_str = " → remontée à 3 (pipeline complet)" if final_d > init_d else ""
    print(f"[Agent] Difficulté : {init_d} — {diff_labels.get(init_d, '?')}{upgrade_str}")
    if final_state["sub_queries"]:
        for i, sq in enumerate(final_state["sub_queries"], 1):
            print(f"[Agent] Sous-requête {i} : {sq}")
    print(f"[Agent] Source(s) ciblée(s) : {', '.join(final_state['sources']) if final_state['sources'] else 'toutes (pas de filtre)'}")
    print(f"[Agent] Tentative(s) de retrieval : {final_state['attempts']}")
    if final_d > 1 or final_state["attempts"] > 1:
        print(f"[Agent] Chunks jugés suffisants : {'oui' if final_state['sufficient'] else 'non'}")


def run_question(entry: dict) -> tuple[dict, dict]:
    """Lance le RAG agentique sur une question et retourne deux dicts :
    - le résultat léger (question, réponse, chunks finaux, logs) destiné à agentic_results.json
    - le résultat complet (+ router, grading, rewrite, chunks avant/après re-ranking) destiné à
      agentic_results_debug.json, utilisé par evaluate_components.py pour ne jamais avoir à
      refaire tourner le pipeline."""
    question = entry["question"]
    print(f"\n{'='*60}")
    print(f"[{entry['id']}] {question}")
    print("=" * 60)

    buffer = io.StringIO()
    start = time.time()
    with contextlib.redirect_stdout(_Tee(sys.stdout, buffer)):
        final_state = run_agent(question, verbose=True)
        _print_summary(final_state)
    duration = time.time() - start

    docs = final_state["docs"]
    rewrite_triggered = final_state["attempts"] >= 2

    # Affiche les chunks sélectionnés avec leur score de re-ranking
    print(f"\n[Chunks sélectionnés] {len(docs)} chunk(s) :")
    for i, doc in enumerate(docs):
        score = doc.metadata.get("rerank_score")
        score_str = f"{score:.4f}" if score is not None else "n/a"
        print(f"  #{i+1}  score={score_str}  {doc.metadata.get('source','?')}  p.{doc.metadata.get('page','?')}")

    light = {
        "id": entry["id"],
        "question": question,
        "reponse_attendue": entry["reponse_attendue"],
        "reponse_llm": final_state["answer"],
        "duree_secondes": round(duration, 1),
        "logs": buffer.getvalue().strip(),
        "chunks": [_doc_to_dict(d) for d in docs],
    }

    debug = {
        **light,
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

    return light, debug


def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    # Reprend là où on s'était arrêté (utile si le run de nuit est interrompu) — la reprise se
    # base sur le fichier léger, qui existe toujours dès qu'une question a été traitée.
    if OUTPUT_PATH.exists():
        results = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        already_done = {r["id"] for r in results}
    else:
        results = []
        already_done = set()

    debug_results = json.loads(DEBUG_OUTPUT_PATH.read_text(encoding="utf-8")) if DEBUG_OUTPUT_PATH.exists() else []

    to_run = [e for e in dataset if e["id"] not in already_done]

    print("=== RAG agentique — passage complet sur le dataset ===")
    print(f"Dataset      : {DATASET_PATH.name} ({len(dataset)} questions au total)")
    print(f"Déjà traités : {len(already_done)} — À traiter : {len(to_run)}")
    print(f"Résultats    : {OUTPUT_PATH}")
    print(f"Debug (eval) : {DEBUG_OUTPUT_PATH}")

    try:
        for entry in to_run:
            light, debug = run_question(entry)
            results.append(light)
            debug_results.append(debug)
            # Sauvegarde après chaque question : un crash ou un Ctrl+C en pleine nuit ne perd rien
            OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            DEBUG_OUTPUT_PATH.write_text(json.dumps(debug_results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  → sauvegardé ({len(results)}/{len(dataset)})")
    except KeyboardInterrupt:
        print("\n\nInterrompu — résultats partiels sauvegardés.")

    print(f"\nTerminé. {len(results)} question(s) traitée(s). Résultats dans {OUTPUT_PATH} et {DEBUG_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
