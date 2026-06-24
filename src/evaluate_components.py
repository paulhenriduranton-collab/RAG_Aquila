# evaluate_components.py
# Évaluation par composant (component-level) du RAG agentique.
# 6 briques testées indépendamment :
#   1. Router (identify_sources)       — comparaison exacte avec ground truth
#   2. Retrieval (pipeline complet)    — RAGAS Context Precision + Context Recall
#   3. Re-ranking (cross-encoder)      — RAGAS Context Precision avant vs après
#   4. Grading (jugement suffisance)   — verdict OUI/NON enregistré
#   5. Query Rewriting (reformulation) — RAGAS Context Precision après rewrite
#   6. Generation (réponse finale)     — RAGAS Faithfulness
#
# Ne ré-exécute jamais le pipeline (retrieval/génération) : tout est lu depuis
# data/agentic_results_debug.json, généré une seule fois par run_agentic_all.py (qui fait
# tourner le vrai graph agent.py et sauvegarde tous les états intermédiaires : router,
# chunks avant/après re-ranking, verdict de grading, requête reformulée). Seuls les
# calculs propres au scoring (RAGAS + juge externe du grading) font des appels LLM ici
# — ce qui permet de relancer/ajuster l'évaluation sans tout refaire tourner.
# Tourne sur Colab avec Ollama — même setup que evaluate_ragas.py.

import asyncio
import inspect
import json
import math
import sys
import time
import pandas as pd
from pathlib import Path

# Ajoute src/ au path pour les imports locaux
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ask import _invoke_with_retry, _maybe_restart_ollama
from eval_common import init_source_map, expected_sources, expected_difficulty, chunks_to_contexts

# --- Imports RAGAS ---
from ragas import SingleTurnSample
from ragas.metrics.collections import (
    Faithfulness,        # la réponse est-elle fidèle au contexte ?
    ContextPrecision,    # les chunks récupérés sont-ils pertinents ?
    ContextRecall,        # le contexte couvre-t-il la réponse attendue ?
)
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from openai import AsyncOpenAI

# =============================================================================
# CONFIGURATION — modifier ici selon l'environnement
# =============================================================================
QUESTIONS_PATH       = Path(__file__).resolve().parent.parent / "data" / "questions.json"
AGENTIC_RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "agentic_results_debug.json"
OUTPUT_CSV           = Path(__file__).resolve().parent.parent / "data" / "component_evaluation.csv"

EVAL_LLM    = "gemma2:2b"   # modèle Ollama pour les jugements RAGAS (plus rapide que gemma4:12b)
EVAL_EMBED  = "bge-m3"      # modèle d'embedding pour RAGAS

# =============================================================================
# RAGAS SETUP
# =============================================================================

def _init_ragas():
    """Initialise le LLM et les métriques RAGAS via Ollama (API compatible OpenAI)."""
    # Client async OpenAI pointant vers Ollama local
    client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    ragas_llm = llm_factory(EVAL_LLM, client=client)
    ragas_emb = embedding_factory("openai", model=EVAL_EMBED, client=client, interface="modern")

    # 3 métriques utilisées par l'évaluation par composant
    metrics = {
        "ctx_precision": ContextPrecision(llm=ragas_llm),
        "ctx_recall": ContextRecall(llm=ragas_llm),
        "faithfulness": Faithfulness(llm=ragas_llm),
    }
    return metrics


def _build_kwargs(method, sample: SingleTurnSample) -> dict:
    """Mappe les champs de SingleTurnSample aux paramètres attendus par ascore()."""
    field_map = {
        "user_input":          sample.user_input,
        "response":            sample.response,
        "retrieved_contexts":  sample.retrieved_contexts,
        "reference":           sample.reference,
    }
    sig = inspect.signature(method)
    return {
        name: field_map[name]
        for name, param in sig.parameters.items()
        if name in field_map and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    }


async def _ragas_score(metric, sample: SingleTurnSample) -> float:
    """Score un sample avec une métrique RAGAS. Retourne un float entre 0 et 1."""
    try:
        kwargs = _build_kwargs(metric.ascore, sample)
        result = metric.ascore(**kwargs)
        score = await result if asyncio.iscoroutine(result) else result
        val = score.value if hasattr(score, "value") else score
        return round(float(val), 3)
    except Exception as e:
        print(f"    [WARN] {metric.name}: {e}")
        return float("nan")


# =============================================================================
# ÉVALUATION — 1. ROUTER
# =============================================================================

def eval_router(entry: dict, meta: dict) -> dict:
    """Compare la prédiction du router (capturée par run_agentic_all.py) à la ground truth.
    Aucun appel LLM ici — identify_sources a déjà tourné dans le vrai graph agent.py."""
    pred_src = entry["router"]["sources"]
    pred_diff = entry["router"]["difficulty"]
    exp_src = expected_sources(meta)
    exp_diff = expected_difficulty(meta)

    if exp_src is None and pred_src is None:
        src_ok = True
    elif exp_src is not None and pred_src is not None:
        src_ok = set(pred_src) == set(exp_src)
    else:
        src_ok = False

    return {
        "router_source_ok": int(src_ok),
        "router_diff_ok": int(pred_diff == exp_diff),
        "router_pred_src": ",".join(pred_src) if pred_src else "TOUS",
        "router_exp_src": ",".join(exp_src) if exp_src else "TOUS",
        "router_pred_diff": pred_diff,
        "router_exp_diff": exp_diff,
    }


# =============================================================================
# ÉVALUATION — 2+3. RETRIEVAL + RE-RANKING (RAGAS Context Precision/Recall)
# =============================================================================

async def eval_retrieval_and_reranking(entry: dict, meta: dict, ragas_metrics: dict) -> dict:
    """Évalue retrieval (Context Precision + Recall) et re-ranking (delta Context Precision)
    à partir des chunks pre/post re-ranking déjà capturés par run_agentic_all.py."""
    question = meta["question"]
    reference = meta["reponse_attendue"]
    pre_ctx = chunks_to_contexts(entry["pre_rerank_docs"])
    post_ctx = chunks_to_contexts(entry["post_rerank_docs"])

    # --- ② Retrieval : Context Precision + Recall APRÈS re-ranking (avant tout rewrite) ---
    post_sample = SingleTurnSample(
        user_input=question, response="",
        retrieved_contexts=post_ctx, reference=reference,
    )

    if pre_ctx == post_ctx:
        # Pool <= K_FINAL : _rerank() n'a jamais été appelé (cf. agent.py::retrieve_node), donc
        # pre == post exactement. Inutile de scorer deux fois le même contenu — le delta sera 0.
        post_precision, post_recall = await asyncio.gather(
            _ragas_score(ragas_metrics["ctx_precision"], post_sample),
            _ragas_score(ragas_metrics["ctx_recall"], post_sample),
        )
        pre_precision, pre_recall = post_precision, post_recall
    else:
        # --- ③ Re-ranking : Context Precision + Recall AVANT re-ranking ---
        pre_sample = SingleTurnSample(
            user_input=question, response="",
            retrieved_contexts=pre_ctx, reference=reference,
        )
        pre_precision, pre_recall, post_precision, post_recall = await asyncio.gather(
            _ragas_score(ragas_metrics["ctx_precision"], pre_sample),
            _ragas_score(ragas_metrics["ctx_recall"], pre_sample),
            _ragas_score(ragas_metrics["ctx_precision"], post_sample),
            _ragas_score(ragas_metrics["ctx_recall"], post_sample),
        )

    # Deltas re-ranking (positif = le re-ranker améliore)
    def _delta(a: float, b: float) -> float:
        return round(a - b, 3) if not (math.isnan(a) or math.isnan(b)) else float("nan")

    return {
        # ② Retrieval (résultat après re-ranking, avant rewrite éventuel)
        "retrieval_ctx_precision": post_precision,
        "retrieval_ctx_recall": post_recall,
        "retrieval_n_chunks": len(entry["post_rerank_docs"]),
        # ③ Re-ranking (amélioration apportée par le cross-encoder)
        "rerank_pre_precision": pre_precision,
        "rerank_post_precision": post_precision,
        "rerank_precision_delta": _delta(post_precision, pre_precision),
        "rerank_pre_recall": pre_recall,
        "rerank_post_recall": post_recall,
        "rerank_recall_delta": _delta(post_recall, pre_recall),
    }


# =============================================================================
# ÉVALUATION — 4. GRADING
# =============================================================================

# Prompt du juge externe — il a accès à la réponse attendue, contrairement au grader du pipeline
GRADING_JUDGE = """Voici une question, des extraits de documents, et la réponse attendue.

Question : {question}

Extraits récupérés :
{context}

Réponse attendue (ground truth) :
{reference}

En comparant les extraits avec la réponse attendue, est-ce que les extraits contiennent
suffisamment d'information pour produire cette réponse attendue ?

- Si oui, réponds uniquement : OUI
- Si non, réponds : NON — [explique en 1 phrase ce qui manque]"""


async def eval_grading(entry: dict, meta: dict) -> dict:
    """Évalue grade_documents en comparant son verdict (déjà capturé par run_agentic_all.py)
    avec celui d'un juge externe qui, lui, voit la réponse attendue. Seul ce juge externe
    fait un appel LLM ici. Non applicable pour les questions de difficulté 1 (le vrai pipeline
    ne grade jamais ces questions — cf. agent.py::_route_after_retrieve)."""
    grading = entry["grading"]
    if not grading.get("performed"):
        return {
            "grading_performed": 0,
            "grading_sufficient": float("nan"),
            "grading_judge_sufficient": float("nan"),
            "grading_correct": float("nan"),
            "grading_label": "non_applicable",
            "grading_verdict": "",
        }

    grader_sufficient = grading["sufficient"]
    grader_verdict = grading["verdict"]
    post_docs = entry["post_rerank_docs"]

    # Pas de troncature — le juge doit voir les mêmes chunks complets que le grader du pipeline
    context = "\n\n---\n\n".join(c["content"] for c in post_docs)
    judge_prompt = GRADING_JUDGE.format(
        question=meta["question"], context=context, reference=meta["reponse_attendue"],
    )
    judge_raw = _invoke_with_retry(judge_prompt).strip()
    judge_sufficient = judge_raw.lower().startswith("oui")

    if grader_sufficient and judge_sufficient:
        label = "vrai_positif"       # les deux disent OUI
    elif not grader_sufficient and not judge_sufficient:
        label = "vrai_negatif"       # les deux disent NON
    elif not grader_sufficient and judge_sufficient:
        label = "faux_negatif"       # grader dit NON mais le juge dit OUI → trop sévère
    else:
        label = "faux_positif"       # grader dit OUI mais le juge dit NON → trop laxiste

    return {
        "grading_performed": 1,
        "grading_sufficient": int(grader_sufficient),
        "grading_judge_sufficient": int(judge_sufficient),
        "grading_correct": int(grader_sufficient == judge_sufficient),
        "grading_label": label,
        "grading_verdict": grader_verdict[:200],
    }


# =============================================================================
# ÉVALUATION — 5. QUERY REWRITING
# =============================================================================

async def eval_rewriting(
    entry: dict, meta: dict,
    baseline_precision: float, baseline_recall: float,
    ragas_metrics: dict,
) -> dict:
    """Évalue le query rewriting : la reformulation améliore-t-elle le retrieval ?
    Le déclenchement et la fusion des chunks ont déjà eu lieu dans le vrai pipeline
    (run_agentic_all.py) — ici on ne fait que le scoring RAGAS sur les chunks finaux."""
    rewrite = entry["rewrite"]
    if not rewrite.get("triggered"):
        return {
            "rewrite_triggered": 0,
            "rewrite_new_query": "",
            "rewrite_ctx_precision": float("nan"),
            "rewrite_precision_gain": float("nan"),
            "rewrite_ctx_recall": float("nan"),
            "rewrite_recall_gain": float("nan"),
        }

    # entry["chunks"] = pool final du graph, déjà fusionné post-rewrite (3 anciens + 2 nouveaux)
    merged_ctx = chunks_to_contexts(entry["chunks"])
    sample = SingleTurnSample(
        user_input=meta["question"], response="",
        retrieved_contexts=merged_ctx, reference=meta["reponse_attendue"],
    )
    new_precision, new_recall = await asyncio.gather(
        _ragas_score(ragas_metrics["ctx_precision"], sample),
        _ragas_score(ragas_metrics["ctx_recall"], sample),
    )

    def _gain(new: float, old: float) -> float:
        return round(new - old, 3) if not (math.isnan(new) or math.isnan(old)) else float("nan")

    return {
        "rewrite_triggered": 1,
        "rewrite_new_query": rewrite["new_query"][:150],
        "rewrite_ctx_precision": new_precision,
        "rewrite_precision_gain": _gain(new_precision, baseline_precision),
        "rewrite_ctx_recall": new_recall,
        "rewrite_recall_gain": _gain(new_recall, baseline_recall),
    }


# =============================================================================
# ÉVALUATION — 6. GENERATION (RAGAS Faithfulness)
# =============================================================================

async def eval_generation(entry: dict, meta: dict, ragas_metrics: dict) -> dict:
    """Évalue la génération : la réponse réellement produite par le pipeline est-elle
    fidèle aux chunks qui ont vraiment servi à la générer (entry["chunks"], le pool final —
    post-rewrite si applicable) ? Seul le scoring RAGAS Faithfulness fait un appel LLM ici."""
    answer = entry["reponse_llm"]
    ctx = chunks_to_contexts(entry["chunks"])

    sample = SingleTurnSample(
        user_input=meta["question"], response=answer,
        retrieved_contexts=ctx, reference=meta["reponse_attendue"],
    )
    faithfulness = await _ragas_score(ragas_metrics["faithfulness"], sample)

    return {
        "gen_answer": answer[:300],
        "gen_faithfulness": faithfulness,
    }


# =============================================================================
# BOUCLE D'ÉVALUATION PRINCIPALE (async)
# =============================================================================

async def _run_all(questions: list[dict], results_by_id: dict, ragas_metrics: dict) -> pd.DataFrame:
    """Évalue toutes les questions, sauvegarde après chaque question."""
    # Reprise si le CSV existe déjà
    if OUTPUT_CSV.exists():
        df_existing = pd.read_csv(OUTPUT_CSV, encoding="utf-8")
        already_done = set(df_existing["id"].astype(str))
        rows = df_existing.to_dict("records")
        print(f"      Reprise : {len(already_done)} questions déjà évaluées")
    else:
        already_done = set()
        rows = []

    total = len(questions)

    for i, meta in enumerate(questions, 1):
        qid = meta["id"]
        question = meta["question"]

        if qid in already_done:
            print(f"  [SKIP] {qid}")
            continue

        entry = results_by_id.get(qid)
        if entry is None:
            print(f"  [MANQUANT] {qid} — pas de résultat agentic (lancez run_agentic_all.py)")
            continue

        _maybe_restart_ollama(verbose=True)

        q_preview = question[:80] + "…" if len(question) > 80 else question
        print(f"\n{'─'*65}")
        print(f"  [{i}/{total}] {qid} — {q_preview}")
        print(f"{'─'*65}")

        row: dict = {
            "id": qid, "question": question,
            "niveau": meta.get("niveau"), "source": meta.get("source"),
        }
        t0 = time.time()

        try:
            # ① Router
            router_row = eval_router(entry, meta)
            row.update(router_row)
            print(f"  ① Router : src={'OK' if router_row['router_source_ok'] else 'MISS'}  "
                  f"diff={'OK' if router_row['router_diff_ok'] else 'MISS'} "
                  f"(pred={router_row['router_pred_diff']}, exp={router_row['router_exp_diff']})")

            # ②③ Retrieval + Re-ranking
            ret_row = await eval_retrieval_and_reranking(entry, meta, ragas_metrics)
            row.update(ret_row)
            print(f"  ②③ ctx_prec={ret_row['retrieval_ctx_precision']}  "
                  f"ctx_rec={ret_row['retrieval_ctx_recall']}  "
                  f"rerank Δprec={ret_row['rerank_precision_delta']}  "
                  f"Δrec={ret_row['rerank_recall_delta']}")

            # ④ Grading
            grade_row = await eval_grading(entry, meta)
            row.update(grade_row)
            if grade_row["grading_performed"]:
                grader = "OUI" if grade_row["grading_sufficient"] else "NON"
                judge = "OUI" if grade_row["grading_judge_sufficient"] else "NON"
                print(f"  ④ Grading : grader={grader}  juge={judge}  → {grade_row['grading_label']}")
            else:
                print("  ④ Grading : non applicable (difficulté 1)")

            # ⑤ Query Rewriting
            rewrite_row = await eval_rewriting(
                entry, meta,
                baseline_precision=ret_row["retrieval_ctx_precision"],
                baseline_recall=ret_row["retrieval_ctx_recall"],
                ragas_metrics=ragas_metrics,
            )
            row.update(rewrite_row)
            if rewrite_row["rewrite_triggered"]:
                print(f"  ⑤ Rewrite : Δprec={rewrite_row['rewrite_precision_gain']}  "
                      f"Δrec={rewrite_row['rewrite_recall_gain']}")
            else:
                print("  ⑤ Rewrite : non déclenché")

            # ⑥ Generation
            gen_row = await eval_generation(entry, meta, ragas_metrics)
            row.update(gen_row)
            print(f"  ⑥ Generation : faithfulness={gen_row['gen_faithfulness']}")

        except Exception as e:
            print(f"\n  [ERREUR] {type(e).__name__}: {e}")
            row["error"] = str(e)[:200]

        row["duree_secondes"] = round(time.time() - t0, 1)

        # Sauvegarde incrémentale après chaque question
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"  → sauvegardé ({len(rows)}/{total}) en {row['duree_secondes']}s")

    return pd.DataFrame(rows)


# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================

def print_summary(df: pd.DataFrame):
    """Affiche un résumé des scores par composant et par niveau."""
    n = len(df)
    print(f"\n{'='*65}")
    print("  RÉSUMÉ — ÉVALUATION PAR COMPOSANT")
    print(f"{'='*65}")

    # ① Router
    if "router_source_ok" in df.columns:
        src_acc = df["router_source_ok"].mean()
        diff_acc = df["router_diff_ok"].mean()
        print(f"\n  ① ROUTER ({n} questions)")
        print(f"     Source correcte     : {src_acc:.1%}")
        print(f"     Difficulté correcte : {diff_acc:.1%}")

    # ② Retrieval
    if "retrieval_ctx_precision" in df.columns:
        prec = df["retrieval_ctx_precision"].mean()
        rec = df["retrieval_ctx_recall"].mean()
        print(f"\n  ② RETRIEVAL")
        print(f"     Context Precision : {prec:.3f}")
        print(f"     Context Recall    : {rec:.3f}")

    # ③ Re-ranking
    if "rerank_precision_delta" in df.columns:
        pre_p = df["rerank_pre_precision"].mean()
        post_p = df["rerank_post_precision"].mean()
        delta_p = df["rerank_precision_delta"].mean()
        pre_r = df["rerank_pre_recall"].mean()
        post_r = df["rerank_post_recall"].mean()
        delta_r = df["rerank_recall_delta"].mean()
        print(f"\n  ③ RE-RANKING")
        print(f"     Context Precision avant  : {pre_p:.3f}")
        print(f"     Context Precision après  : {post_p:.3f}")
        print(f"     Δ Precision              : {delta_p:+.3f}")
        print(f"     Context Recall avant     : {pre_r:.3f}")
        print(f"     Context Recall après     : {post_r:.3f}")
        print(f"     Δ Recall                 : {delta_r:+.3f}")

    # ④ Grading — uniquement sur les questions où le grading a réellement eu lieu (difficulté > 1)
    if "grading_performed" in df.columns:
        graded = df[df["grading_performed"] == 1]
        print(f"\n  ④ GRADING ({len(graded)}/{n} questions concernées — difficulté > 1)")
        if len(graded) > 0:
            suf = graded["grading_sufficient"].mean()
            acc = graded["grading_correct"].mean()
            labels = graded["grading_label"].value_counts()
            tp = labels.get("vrai_positif", 0)
            tn = labels.get("vrai_negatif", 0)
            fn = labels.get("faux_negatif", 0)
            fp = labels.get("faux_positif", 0)
            precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            f1 = (2 * precision * recall / (precision + recall)
                  if not (math.isnan(precision) or math.isnan(recall)) and (precision + recall) > 0
                  else float("nan"))
            print(f"     Grader dit OUI          : {suf:.1%}")
            print(f"     Accuracy vs juge externe: {acc:.1%}")
            for label in ["vrai_positif", "vrai_negatif", "faux_negatif", "faux_positif"]:
                count = labels.get(label, 0)
                print(f"     {label:<20} : {count}")

            # Matrice de confusion (grader réel vs juge externe = référence)
            print(f"\n     {'Matrice de confusion':<25}{'Juge: OUI':>12}{'Juge: NON':>12}")
            print(f"     {'Grader: OUI':<25}{tp:>12}{fp:>12}")
            print(f"     {'Grader: NON':<25}{fn:>12}{tn:>12}")

            print(f"\n     Precision (classe OUI)  : {precision:.1%}" if not math.isnan(precision) else "\n     Precision (classe OUI)  : N/A")
            print(f"     Recall (classe OUI)     : {recall:.1%}" if not math.isnan(recall) else "     Recall (classe OUI)     : N/A")
            print(f"     F1-score (classe OUI)   : {f1:.3f}" if not math.isnan(f1) else "     F1-score (classe OUI)   : N/A")
        else:
            print("     Aucune question concernée")

    # ⑤ Query Rewriting
    if "rewrite_triggered" in df.columns:
        triggered = df[df["rewrite_triggered"] == 1]
        print(f"\n  ⑤ QUERY REWRITING")
        if len(triggered) > 0:
            gain_p = triggered["rewrite_precision_gain"].mean()
            gain_r = triggered["rewrite_recall_gain"].mean()
            print(f"     Déclenchements       : {len(triggered)}/{n}")
            print(f"     Δ Context Precision   : {gain_p:+.3f}")
            print(f"     Δ Context Recall      : {gain_r:+.3f}")
        else:
            print(f"     Jamais déclenché")

    # ⑥ Generation
    if "gen_faithfulness" in df.columns:
        faith = df["gen_faithfulness"].mean()
        print(f"\n  ⑥ GENERATION")
        print(f"     Faithfulness : {faith:.3f}")

    # Ventilation par niveau
    if "niveau" in df.columns and "retrieval_ctx_precision" in df.columns:
        print(f"\n  {'─'*50}")
        print("  PAR NIVEAU")
        for niv, grp in df.groupby("niveau"):
            prec = grp["retrieval_ctx_precision"].mean()
            rec = grp["retrieval_ctx_recall"].mean()
            faith = grp["gen_faithfulness"].mean() if "gen_faithfulness" in grp.columns else float("nan")
            r_src = grp["router_source_ok"].mean() if "router_source_ok" in grp.columns else float("nan")
            print(f"     Niveau {niv} ({len(grp):>2} qs) : "
                  f"prec={prec:.3f}  rec={rec:.3f}  faith={faith:.3f}  router={r_src:.0%}")

    print(f"\n{'='*65}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("[1/4] Chargement du dataset et des résultats agentic...")
    if not QUESTIONS_PATH.exists():
        sys.exit(f"[ERREUR] Fichier introuvable : {QUESTIONS_PATH}")
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    if not AGENTIC_RESULTS_PATH.exists():
        sys.exit(
            f"[ERREUR] Fichier introuvable : {AGENTIC_RESULTS_PATH}\n"
            "Lancez d'abord `python run_agentic_all.py` pour générer les résultats du pipeline "
            "(génère à la fois agentic_results.json et agentic_results_debug.json)."
        )
    results = json.loads(AGENTIC_RESULTS_PATH.read_text(encoding="utf-8"))
    results_by_id = {r["id"]: r for r in results}
    print(f"      {len(questions)} questions chargées — {len(results_by_id)} résultats agentic disponibles")

    # Détecte les sources disponibles dans documents/ (pour la ground truth du router)
    init_source_map()

    # Initialisation RAGAS avec Ollama
    print(f"[2/4] Initialisation RAGAS ({EVAL_LLM}, {EVAL_EMBED})...")
    ragas_metrics = _init_ragas()
    print(f"      3 métriques : {list(ragas_metrics.keys())}")

    # Lancement de l'évaluation
    print(f"[3/4] Évaluation par composant ({len(questions)} questions)...")
    print(f"      Résultats → {OUTPUT_CSV}\n")

    # Gestion de l'event loop (script Python vs Jupyter/Colab)
    try:
        df = asyncio.run(_run_all(questions, results_by_id, ragas_metrics))
    except RuntimeError:
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        df = loop.run_until_complete(_run_all(questions, results_by_id, ragas_metrics))

    # Résumé final
    print("\n[4/4] Résumé :")
    print_summary(df)
    print(f"\nTerminé. Résultats dans {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
