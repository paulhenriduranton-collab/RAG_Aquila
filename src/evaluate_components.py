# evaluate_components.py
# Évaluation par composant (component-level) du RAG agentique.
# 6 briques testées indépendamment :
#   1. Router (identify_sources)       — comparaison exacte avec ground truth
#   2. Retrieval (pipeline complet)    — RAGAS Context Precision + Context Recall
#   3. Re-ranking (cross-encoder)      — RAGAS Context Precision avant vs après
#   4. Grading (jugement suffisance)   — verdict OUI/NON enregistré
#   5. Query Rewriting (reformulation) — RAGAS Context Precision après rewrite
#   6. Generation (réponse finale)     — RAGAS Faithfulness
# Tourne sur Colab avec Ollama — même setup que evaluate_ragas.py.

import asyncio
import inspect
import json
import math
import re
import sys
import time
import pandas as pd
from pathlib import Path

# Ajoute src/ au path pour les imports locaux
sys.path.insert(0, str(Path(__file__).resolve().parent))

# --- Imports du pipeline de retrieval (ask.py) ---
from ask import (
    retrieve, _rerank, _merge, _dedup, _build_bm25_index, _tokenize, _hyde,
    _invoke_with_retry, _maybe_restart_ollama, llm,
    K_RETRIEVE, K_RERANK, K_FINAL, MMR_FETCH_K, MMR_LAMBDA,
    VECTOR_DB_DIR, EMBED_MODEL, PROMPT_PATH,
)
# --- Imports du pipeline agentique (agent.py) ---
from agent import (
    identify_sources, grade_documents, rewrite_query, generate_node,
    AgentState, _available_sources,
)
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# --- Imports RAGAS ---
from ragas import SingleTurnSample
from ragas.metrics.collections import (
    Faithfulness,        # la réponse est-elle fidèle au contexte ?
    ContextPrecision,    # les chunks récupérés sont-ils pertinents ?
    ContextRecall,       # le contexte couvre-t-il la réponse attendue ?
)
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from openai import AsyncOpenAI

# =============================================================================
# CONFIGURATION — modifier ici selon l'environnement
# =============================================================================
QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "questions.json"
OUTPUT_CSV     = Path(__file__).resolve().parent.parent / "data" / "component_evaluation.csv"

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


def _docs_to_contexts(docs: list[Document]) -> list[str]:
    """Convertit des Documents LangChain en liste de strings pour RAGAS."""
    return [
        f"[Source : {d.metadata.get('source', '?')} — p.{d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
        if d.page_content.strip()
    ]


# =============================================================================
# GROUND TRUTH — PARSING
# =============================================================================

# Mapping source (questions.json) → fichiers PDF — rempli au démarrage par init_source_map()
SOURCE_MAP: dict[str, list[str]] = {}


def init_source_map():
    """Détecte les PDF dans documents/ et construit le mapping source → fichier(s)."""
    global SOURCE_MAP
    available = _available_sources()
    SOURCE_MAP.clear()
    for name in available:
        upper = name.upper()
        if "ENS" in upper:
            SOURCE_MAP.setdefault("ENS", []).append(name)
        else:
            SOURCE_MAP.setdefault("Sorbonne", []).append(name)
    # ENS+Sorbonne = pas de filtre, toutes les sources
    SOURCE_MAP["ENS+Sorbonne"] = available
    print(f"[Config] Sources détectées : {SOURCE_MAP}")


def expected_sources(meta: dict) -> list[str] | None:
    """Fichiers PDF attendus pour cette question. None = toutes les sources."""
    src = meta.get("source", "")
    if "+" in src:
        return None
    return SOURCE_MAP.get(src)


def expected_difficulty(meta: dict) -> int:
    """Difficulté attendue du router (niveau 1→factuel, 2→synthèse, 3→complexe)."""
    return meta.get("niveau", 2)


# =============================================================================
# PIPELINE AVEC ÉTATS INTERMÉDIAIRES
# =============================================================================

def run_pipeline_detailed(question: str, sources: list[str] | None, difficulty: int) -> dict:
    """Exécute le retrieval pas à pas et retourne les docs à chaque étape.
    Permet de comparer les chunks avant et après re-ranking."""
    use_hyde = difficulty > 1

    # Connexion à la base vectorielle Chroma
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_db = Chroma(persist_directory=str(VECTOR_DB_DIR), embedding_function=embeddings)
    chroma_filter = {"source": {"$in": sources}} if sources else None

    # HyDE (réponse fictive pour améliorer la recherche sémantique)
    hyde_query = _hyde(question) if use_hyde else question

    # Recherche sémantique (MMR pour diversité)
    mmr_docs = vector_db.max_marginal_relevance_search(
        hyde_query, k=K_RETRIEVE, fetch_k=MMR_FETCH_K,
        lambda_mult=MMR_LAMBDA, filter=chroma_filter,
    )
    raw_semantic = [(doc, None) for doc in mmr_docs]

    # Recherche BM25 (lexicale)
    bm25, bm25_chunks = _build_bm25_index(vector_db, verbose=False)
    bm25_scores = bm25.get_scores(_tokenize(question))
    candidates = list(range(len(bm25_chunks)))
    if sources:
        candidates = [i for i in candidates if bm25_chunks[i][1].get("source") in sources]
    top_bm25 = sorted(candidates, key=lambda i: bm25_scores[i], reverse=True)[:K_RETRIEVE]

    # Fusion RRF (combine sémantique + BM25)
    rrf_docs, _ = _merge(
        raw_semantic, top_bm25, bm25_chunks,
        max_per_source=K_RERANK if sources else 3,
    )

    # Déduplication (supprime les quasi-doublons)
    deduped = _dedup(rrf_docs, verbose=False)

    # Re-ranking (cross-encoder)
    reranked = _rerank(question, deduped, verbose=False) if deduped else []

    return {
        "pre_rerank": deduped[:K_FINAL],  # top-K AVANT re-ranking (ordre RRF)
        "reranked": reranked,              # top-K APRÈS re-ranking (ordre cross-encoder)
    }


# =============================================================================
# ÉVALUATION — 1. ROUTER
# =============================================================================

def eval_router(question: str, meta: dict) -> tuple[dict, dict]:
    """Évalue identify_sources : source(s) et difficulté correctes ?"""
    state: AgentState = {
        "question": question, "current_query": question,
        "sources": None, "difficulty": 2, "initial_difficulty": 2,
        "sub_queries": [], "docs": [], "sufficient": False,
        "grade_verdict": "", "attempts": 0, "answer": "",
    }
    result = identify_sources(state)
    pred_src = result.get("sources")
    pred_diff = result.get("difficulty", 2)
    exp_src = expected_sources(meta)
    exp_diff = expected_difficulty(meta)

    # Source correcte ? (None==None = "toutes sources" des deux côtés)
    if exp_src is None and pred_src is None:
        src_ok = True
    elif exp_src is not None and pred_src is not None:
        src_ok = set(pred_src) == set(exp_src)
    else:
        src_ok = False

    row = {
        "router_source_ok": int(src_ok),
        "router_diff_ok": int(pred_diff == exp_diff),
        "router_pred_src": ",".join(pred_src) if pred_src else "TOUS",
        "router_exp_src": ",".join(exp_src) if exp_src else "TOUS",
        "router_pred_diff": pred_diff,
        "router_exp_diff": exp_diff,
    }
    return row, result


# =============================================================================
# ÉVALUATION — 2+3. RETRIEVAL + RE-RANKING (RAGAS Context Precision/Recall)
# =============================================================================

async def eval_retrieval_and_reranking(
    question: str, meta: dict, sources: list[str] | None,
    difficulty: int, ragas_metrics: dict,
) -> tuple[dict, list[Document]]:
    """Évalue retrieval (Context Precision + Recall) et re-ranking (delta Context Precision).
    Un seul passage pipeline, deux scorings RAGAS."""
    reference = meta["reponse_attendue"]

    # Exécute le pipeline et capture avant/après re-ranking
    pipeline = run_pipeline_detailed(question, sources, difficulty)
    pre_docs = pipeline["pre_rerank"]
    post_docs = pipeline["reranked"]

    # --- ③ Re-ranking : Context Precision + Recall AVANT re-ranking ---
    pre_sample = SingleTurnSample(
        user_input=question,
        response="",
        retrieved_contexts=_docs_to_contexts(pre_docs),
        reference=reference,
    )
    pre_precision = await _ragas_score(ragas_metrics["ctx_precision"], pre_sample)
    pre_recall = await _ragas_score(ragas_metrics["ctx_recall"], pre_sample)

    # --- ② Retrieval : Context Precision + Recall APRÈS re-ranking (résultat final) ---
    post_sample = SingleTurnSample(
        user_input=question,
        response="",
        retrieved_contexts=_docs_to_contexts(post_docs),
        reference=reference,
    )
    post_precision = await _ragas_score(ragas_metrics["ctx_precision"], post_sample)
    post_recall = await _ragas_score(ragas_metrics["ctx_recall"], post_sample)

    # Deltas re-ranking (positif = le re-ranker améliore)
    def _delta(a: float, b: float) -> float:
        return round(a - b, 3) if not (math.isnan(a) or math.isnan(b)) else float("nan")

    row = {
        # ② Retrieval (résultat final = après re-ranking)
        "retrieval_ctx_precision": post_precision,
        "retrieval_ctx_recall": post_recall,
        "retrieval_n_chunks": len(post_docs),
        # ③ Re-ranking (amélioration apportée par le cross-encoder)
        "rerank_pre_precision": pre_precision,
        "rerank_post_precision": post_precision,
        "rerank_precision_delta": _delta(post_precision, pre_precision),
        "rerank_pre_recall": pre_recall,
        "rerank_post_recall": post_recall,
        "rerank_recall_delta": _delta(post_recall, pre_recall),
    }
    return row, post_docs


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


async def eval_grading(question: str, docs: list[Document], meta: dict, ragas_metrics: dict) -> dict:
    """Évalue grade_documents en comparant son verdict avec celui d'un juge externe.
    Le grader du pipeline juge à l'aveugle (sans la réponse attendue).
    Le juge externe juge avec la réponse attendue — son verdict sert de référence."""
    # 1. Verdict du grader (le vrai, celui du pipeline, sans ground truth)
    state: AgentState = {
        "question": question, "current_query": question,
        "sources": None, "difficulty": 2, "initial_difficulty": 2,
        "sub_queries": [], "docs": docs, "sufficient": False,
        "grade_verdict": "", "attempts": 0, "answer": "",
    }
    result = grade_documents(state)
    grader_sufficient = result.get("sufficient", False)
    grader_verdict = result.get("grade_verdict", "")

    # 2. Verdict du juge externe (avec la réponse attendue comme référence)
    context = "\n\n---\n\n".join(d.page_content[:500] for d in docs)
    judge_prompt = GRADING_JUDGE.format(
        question=question, context=context, reference=meta["reponse_attendue"],
    )
    judge_raw = _invoke_with_retry(judge_prompt).strip()
    judge_sufficient = judge_raw.lower().startswith("oui")

    # 3. Comparaison : le grader est-il d'accord avec le juge ?
    if grader_sufficient and judge_sufficient:
        label = "vrai_positif"       # les deux disent OUI
    elif not grader_sufficient and not judge_sufficient:
        label = "vrai_negatif"       # les deux disent NON
    elif not grader_sufficient and judge_sufficient:
        label = "faux_negatif"       # grader dit NON mais le juge dit OUI → trop sévère
    else:
        label = "faux_positif"       # grader dit OUI mais le juge dit NON → trop laxiste

    return {
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
    question: str, docs: list[Document], sources: list[str] | None,
    difficulty: int, grading_sufficient: bool, grade_verdict: str,
    meta: dict, baseline_precision: float, baseline_recall: float,
    ragas_metrics: dict,
) -> dict:
    """Évalue le query rewriting : la reformulation améliore-t-elle le retrieval ?
    Compare Context Precision et Context Recall avant/après reformulation."""
    # Le rewriting n'est déclenché que si grade=NON et difficulté >= 2
    if grading_sufficient or difficulty < 2:
        return {
            "rewrite_triggered": 0,
            "rewrite_new_query": "",
            "rewrite_ctx_precision": float("nan"),
            "rewrite_precision_gain": float("nan"),
            "rewrite_ctx_recall": float("nan"),
            "rewrite_recall_gain": float("nan"),
            "rewrite_grade_flip": 0,
        }

    # Reformulation de la requête
    state: AgentState = {
        "question": question, "current_query": question,
        "sources": sources, "difficulty": max(difficulty, 3),
        "initial_difficulty": difficulty, "sub_queries": [],
        "docs": docs, "sufficient": False,
        "grade_verdict": grade_verdict, "attempts": 1, "answer": "",
    }
    rewrite_result = rewrite_query(state)
    new_query = rewrite_result.get("current_query", question)

    # Nouveau retrieval avec la requête reformulée
    new_docs = retrieve(new_query, sources=sources, verbose=False, use_hyde=False)

    # Fusion comme dans retrieve_node : 3 anciens + 2 nouveaux
    old_top = docs[:K_FINAL - 2]
    old_contents = {d.page_content for d in old_top}
    new_filtered = [d for d in new_docs if d.page_content not in old_contents]
    new_top = _rerank(new_query, new_filtered, n=2, verbose=False) if new_filtered else []
    merged = old_top + new_top

    # Context Precision + Recall sur les chunks fusionnés
    sample = SingleTurnSample(
        user_input=question,
        response="",
        retrieved_contexts=_docs_to_contexts(merged),
        reference=meta["reponse_attendue"],
    )
    new_precision = await _ragas_score(ragas_metrics["ctx_precision"], sample)
    new_recall = await _ragas_score(ragas_metrics["ctx_recall"], sample)

    # Re-grading sur les chunks fusionnés
    regrade_state: AgentState = {
        "question": question, "current_query": new_query,
        "sources": sources, "difficulty": 3, "initial_difficulty": difficulty,
        "sub_queries": [], "docs": merged, "sufficient": False,
        "grade_verdict": "", "attempts": 2, "answer": "",
    }
    regrade = grade_documents(regrade_state)

    # Gains par rapport au retrieval initial
    def _gain(new: float, old: float) -> float:
        return round(new - old, 3) if not (math.isnan(new) or math.isnan(old)) else float("nan")

    return {
        "rewrite_triggered": 1,
        "rewrite_new_query": new_query[:150],
        "rewrite_ctx_precision": new_precision,
        "rewrite_precision_gain": _gain(new_precision, baseline_precision),
        "rewrite_ctx_recall": new_recall,
        "rewrite_recall_gain": _gain(new_recall, baseline_recall),
        "rewrite_grade_flip": int(regrade.get("sufficient", False)),
    }


# =============================================================================
# ÉVALUATION — 6. GENERATION (RAGAS Faithfulness)
# =============================================================================

async def eval_generation(
    question: str, docs: list[Document], meta: dict, ragas_metrics: dict,
) -> dict:
    """Évalue la génération : produit une réponse puis mesure sa fidélité via RAGAS."""
    # Génère la réponse avec les chunks récupérés
    state: AgentState = {
        "question": question, "current_query": question,
        "sources": None, "difficulty": 2, "initial_difficulty": 2,
        "sub_queries": [], "docs": docs, "sufficient": True,
        "grade_verdict": "", "attempts": 1, "answer": "",
    }
    gen_result = generate_node(state)
    answer = gen_result.get("answer", "")

    # RAGAS Faithfulness : la réponse est-elle fidèle au contexte ?
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=_docs_to_contexts(docs),
        reference=meta["reponse_attendue"],
    )
    faithfulness = await _ragas_score(ragas_metrics["faithfulness"], sample)

    return {
        "gen_answer": answer[:300],
        "gen_faithfulness": faithfulness,
    }


# =============================================================================
# BOUCLE D'ÉVALUATION PRINCIPALE (async)
# =============================================================================

async def _run_all(questions: list[dict], ragas_metrics: dict) -> pd.DataFrame:
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
            print("  ① Router...", end=" ", flush=True)
            router_row, router_result = eval_router(question, meta)
            row.update(router_row)
            src = router_result["sources"]
            diff = router_result["difficulty"]
            print(f"src={'OK' if router_row['router_source_ok'] else 'MISS'}  "
                  f"diff={'OK' if router_row['router_diff_ok'] else 'MISS'} "
                  f"(pred={diff}, exp={router_row['router_exp_diff']})")

            # ②③ Retrieval + Re-ranking
            print("  ②③ Retrieval + Re-ranking...", end=" ", flush=True)
            ret_row, final_docs = await eval_retrieval_and_reranking(
                question, meta, src, diff, ragas_metrics,
            )
            row.update(ret_row)
            print(f"ctx_prec={ret_row['retrieval_ctx_precision']}  "
                  f"ctx_rec={ret_row['retrieval_ctx_recall']}  "
                  f"rerank Δprec={ret_row['rerank_precision_delta']}  "
                  f"Δrec={ret_row['rerank_recall_delta']}")

            # ④ Grading
            print("  ④ Grading...", end=" ", flush=True)
            grade_row = await eval_grading(question, final_docs, meta, ragas_metrics)
            row.update(grade_row)
            grader = "OUI" if grade_row["grading_sufficient"] else "NON"
            judge = "OUI" if grade_row["grading_judge_sufficient"] else "NON"
            print(f"grader={grader}  juge={judge}  → {grade_row['grading_label']}")

            # ⑤ Query Rewriting
            print("  ⑤ Query Rewriting...", end=" ", flush=True)
            rewrite_row = await eval_rewriting(
                question, final_docs, src, diff,
                bool(grade_row["grading_sufficient"]),
                grade_row["grading_verdict"], meta,
                baseline_precision=ret_row["retrieval_ctx_precision"],
                baseline_recall=ret_row["retrieval_ctx_recall"],
                ragas_metrics=ragas_metrics,
            )
            row.update(rewrite_row)
            if rewrite_row["rewrite_triggered"]:
                print(f"Δprec={rewrite_row['rewrite_precision_gain']}  "
                      f"Δrec={rewrite_row['rewrite_recall_gain']}  "
                      f"flip={'OUI' if rewrite_row['rewrite_grade_flip'] else 'NON'}")
            else:
                print("non déclenché")

            # ⑥ Generation
            print("  ⑥ Generation...", end=" ", flush=True)
            gen_row = await eval_generation(question, final_docs, meta, ragas_metrics)
            row.update(gen_row)
            print(f"faithfulness={gen_row['gen_faithfulness']}")

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

    # ④ Grading
    if "grading_correct" in df.columns:
        suf = df["grading_sufficient"].mean()
        acc = df["grading_correct"].mean()
        labels = df["grading_label"].value_counts()
        print(f"\n  ④ GRADING")
        print(f"     Grader dit OUI          : {suf:.1%}")
        print(f"     Accuracy vs juge externe: {acc:.1%}")
        for label in ["vrai_positif", "vrai_negatif", "faux_negatif", "faux_positif"]:
            count = labels.get(label, 0)
            print(f"     {label:<20} : {count}")

    # ⑤ Query Rewriting
    if "rewrite_triggered" in df.columns:
        triggered = df[df["rewrite_triggered"] == 1]
        print(f"\n  ⑤ QUERY REWRITING")
        if len(triggered) > 0:
            gain_p = triggered["rewrite_precision_gain"].mean()
            gain_r = triggered["rewrite_recall_gain"].mean()
            flip = triggered["rewrite_grade_flip"].mean()
            print(f"     Déclenchements       : {len(triggered)}/{n}")
            print(f"     Δ Context Precision   : {gain_p:+.3f}")
            print(f"     Δ Context Recall      : {gain_r:+.3f}")
            print(f"     Grade flip NON→OUI   : {flip:.1%}")
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
    print("[1/4] Chargement du dataset...")
    if not QUESTIONS_PATH.exists():
        sys.exit(f"[ERREUR] Fichier introuvable : {QUESTIONS_PATH}")
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    print(f"      {len(questions)} questions chargées depuis {QUESTIONS_PATH.name}")

    # Détecte les sources disponibles dans documents/
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
        df = asyncio.run(_run_all(questions, ragas_metrics))
    except RuntimeError:
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        df = loop.run_until_complete(_run_all(questions, ragas_metrics))

    # Résumé final
    print("\n[4/4] Résumé :")
    print_summary(df)
    print(f"\nTerminé. Résultats dans {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
