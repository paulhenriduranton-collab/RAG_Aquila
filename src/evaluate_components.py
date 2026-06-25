# evaluate_components.py
# Évaluation par composant (component-level) du RAG agentique.
# 8 briques testées indépendamment :
#   1. Router (identify_sources)       — comparaison exacte avec ground truth
#   2. Retrieval (pipeline complet)    — RAGAS Context Precision + Context Recall
#   3. Re-ranking (cross-encoder)      — RAGAS Context Precision + MRR (LLM-juge)
#   4. Grading (jugement suffisance)   — verdict OUI/NON enregistré
#   5. Query Rewriting (reformulation) — RAGAS + taux de réussite + nouveaux chunks + ciblage (LLM-juge)
#   6. Generation (réponse finale)     — RAGAS Faithfulness
#   7. Déduplication                   — chunk écarté vs chunk conservé (perte d'info réelle)
#   8. Fusion RRF                      — sémantique vs BM25 vs fusion (LLM-juge)
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
import re
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
# ÉVALUATION — 2+3. RETRIEVAL + RE-RANKING (RAGAS Context Precision/Recall + MRR)
# =============================================================================

# Prompt LLM-juge pour labelliser la pertinence d'un lot de chunks (0 ou 1 chacun)
RELEVANCE_JUDGE_BATCH = """Voici une question, la réponse attendue, et {n} extraits de documents numérotés.

Question : {question}

Réponse attendue : {reference}

Extraits :
{chunks}

Pour chaque extrait, indique s'il contient de l'information directement utile pour produire
la réponse attendue. Réponds avec exactement {n} lignes, au format suivant (rien d'autre) :
1. OUI ou NON
2. OUI ou NON
..."""

_BATCH_SIZE = 5
_BATCH_LINE_RE = re.compile(r"^\s*(\d+)\D+(OUI|NON)", re.IGNORECASE)


def _judge_relevance_one(question: str, reference: str, chunk: str) -> int:
    """Juge un seul chunk isolément (fallback si le parsing du batch échoue)."""
    prompt = RELEVANCE_JUDGE_BATCH.format(
        n=1, question=question, reference=reference, chunks=f"1. {chunk}",
    )
    raw = _invoke_with_retry(prompt).strip()
    return 1 if raw.lower().lstrip("1.").strip().startswith("oui") else 0


def _judge_relevance(question: str, reference: str, chunks: list[dict]) -> list[int]:
    """Labellise chaque chunk comme pertinent (1) ou non (0) via un appel LLM-juge,
    par lots de _BATCH_SIZE chunks pour limiter le nombre d'appels LLM.
    Retourne une liste de labels dans le même ordre que les chunks."""
    labels: list[int] = []
    for i in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[i:i + _BATCH_SIZE]
        contents = [c["content"][:1500] for c in batch]
        numbered = "\n\n".join(f"{j+1}. {content}" for j, content in enumerate(contents))
        prompt = RELEVANCE_JUDGE_BATCH.format(
            n=len(batch), question=question, reference=reference, chunks=numbered,
        )
        raw = _invoke_with_retry(prompt).strip()

        parsed: dict[int, int] = {}
        for line in raw.splitlines():
            m = _BATCH_LINE_RE.match(line)
            if m:
                idx, verdict = int(m.group(1)), m.group(2)
                parsed[idx] = 1 if verdict.upper() == "OUI" else 0

        if len(parsed) == len(batch):
            labels.extend(parsed[j + 1] for j in range(len(batch)))
        else:
            # Parsing raté (réponse mal formée) : on retombe sur un appel par chunk pour ce lot
            labels.extend(_judge_relevance_one(question, reference, content) for content in contents)
    return labels


def _mrr(relevances: list[int]) -> float:
    """Calcule le MRR (Mean Reciprocal Rank) : 1/position du premier chunk pertinent.
    MRR = 1.0 si le premier chunk est pertinent, 0.5 si c'est le deuxième, etc."""
    for i, rel in enumerate(relevances):
        if rel:
            return round(1.0 / (i + 1), 3)
    return 0.0

async def eval_retrieval_and_reranking(entry: dict, meta: dict, ragas_metrics: dict) -> dict:
    """Évalue retrieval (Context Precision + Recall) et re-ranking (delta Context Precision
    + MRR via LLM-juge) à partir des chunks pre/post re-ranking déjà capturés."""
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

    # --- ③bis MRR : le LLM-juge labellise chaque chunk, puis on calcule ---
    # On labellise les chunks POST re-ranking (dans l'ordre produit par le cross-encoder)
    post_docs = entry["post_rerank_docs"]
    if post_docs:
        post_labels = _judge_relevance(question, reference, post_docs)
        rerank_mrr = _mrr(post_labels)
        rerank_n_pertinents = sum(post_labels)
    else:
        post_labels = []
        rerank_mrr = float("nan")
        rerank_n_pertinents = 0

    # MRR AVANT re-ranking pour mesurer l'amélioration apportée par le cross-encoder
    pre_docs = entry["pre_rerank_docs"]
    if pre_docs and pre_ctx != post_ctx:
        pre_labels = _judge_relevance(question, reference, pre_docs)
        pre_mrr = _mrr(pre_labels)
    else:
        pre_labels = post_labels
        pre_mrr = rerank_mrr

    # Deltas re-ranking (positif = le re-ranker améliore)
    def _delta(a: float, b: float) -> float:
        return round(a - b, 3) if not (math.isnan(a) or math.isnan(b)) else float("nan")

    return {
        # ② Retrieval (résultat après re-ranking, avant rewrite éventuel)
        "retrieval_ctx_precision": post_precision,
        "retrieval_ctx_recall": post_recall,
        "retrieval_n_chunks": len(post_docs),
        # ③ Re-ranking — RAGAS (amélioration apportée par le cross-encoder)
        "rerank_pre_precision": pre_precision,
        "rerank_post_precision": post_precision,
        "rerank_precision_delta": _delta(post_precision, pre_precision),
        "rerank_pre_recall": pre_recall,
        "rerank_post_recall": post_recall,
        "rerank_recall_delta": _delta(post_recall, pre_recall),
        # ③ Re-ranking — MRR (qualité de l'ordre des chunks)
        "rerank_pre_mrr": pre_mrr,
        "rerank_post_mrr": rerank_mrr,
        "rerank_mrr_delta": _delta(rerank_mrr, pre_mrr),
        "rerank_n_pertinents": rerank_n_pertinents,
        "rerank_n_total": len(post_docs),
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

# Prompt LLM-juge : la nouvelle requête cible-t-elle bien l'information manquante
# identifiée par le grading (celle qui a déclenché la reformulation) ? Évalue la
# qualité de la reformulation elle-même, indépendamment du résultat du retrieval qui suit.
REWRITE_TARGETING_JUDGE = """Voici une question initiale, l'information jugée manquante dans
les documents récupérés (verdict du grading qui a déclenché la reformulation), et la nouvelle
requête de recherche reformulée pour combler ce manque.

Question initiale : {question}

Information manquante (verdict du grading) : {verdict}

Nouvelle requête reformulée : {new_query}

Cette nouvelle requête cible-t-elle bien et spécifiquement l'information manquante identifiée
ci-dessus ?

- Si oui, réponds uniquement : OUI
- Si non, réponds : NON — [explique en 1 phrase ce qui ne colle pas]"""


def _judge_rewrite_targeting(question: str, verdict: str, new_query: str) -> tuple[int, str]:
    """Juge si la requête reformulée cible bien l'information manquante du verdict de grading.
    Retourne (1 si bien ciblée sinon 0, le verdict brut du juge)."""
    prompt = REWRITE_TARGETING_JUDGE.format(question=question, verdict=verdict, new_query=new_query)
    raw = _invoke_with_retry(prompt).strip()
    return int(raw.lower().startswith("oui")), raw


async def eval_rewriting(
    entry: dict, meta: dict,
    baseline_precision: float, baseline_recall: float,
    ragas_metrics: dict,
) -> dict:
    """Évalue le query rewriting : la reformulation améliore-t-elle le retrieval ?
    Métriques : RAGAS precision/recall + taux de réussite + nouveaux chunks utiles +
    ciblage de la reformulation (la nouvelle requête vise-t-elle bien le manque identifié
    par le grading qui a déclenché le rewrite ?). Le déclenchement et la fusion des chunks
    ont déjà eu lieu dans le vrai pipeline (run_agentic_all.py) — ici on ne fait que le
    scoring RAGAS + les jugements LLM sur les chunks/requêtes finaux."""
    rewrite = entry["rewrite"]
    if not rewrite.get("triggered"):
        return {
            "rewrite_triggered": 0,
            "rewrite_new_query": "",
            "rewrite_ctx_precision": float("nan"),
            "rewrite_precision_gain": float("nan"),
            "rewrite_ctx_recall": float("nan"),
            "rewrite_recall_gain": float("nan"),
            "rewrite_success": float("nan"),
            "rewrite_n_new_chunks": float("nan"),
            "rewrite_n_new_pertinents": float("nan"),
            "rewrite_new_ratio": float("nan"),
            "rewrite_targeted": float("nan"),
            "rewrite_targeting_verdict": "",
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

    # --- Taux de réussite : le grading final (après rewrite, sur le pool post-rewrite) dit-il
    # OUI ? rewrite["grading_after"] est distinct de entry["grading"] (qui porte lui sur le
    # grading AVANT rewrite, cohérent avec post_rerank_docs pour l'éval ④). Absent sur les
    # anciens résultats générés avant l'ajout du grading final post-rewrite → NaN.
    grading = entry["grading"]
    grading_after = rewrite.get("grading_after") or {}
    rewrite_success = int(grading_after["sufficient"]) if "sufficient" in grading_after else float("nan")

    # --- Ciblage de la reformulation : la nouvelle requête vise-t-elle bien le manque
    # identifié par le grading qui a déclenché le rewrite ? (entry["grading"]["verdict"] est
    # le verdict du grading initial, le seul à avoir lieu avant la reformulation)
    rewrite_targeted, targeting_verdict = _judge_rewrite_targeting(
        meta["question"], grading.get("verdict", ""), rewrite["new_query"],
    )

    # --- Nouveaux chunks utiles : combien de chunks du 2ème retrieval sont pertinents ? ---
    chunks_loop_1 = rewrite["chunks_loop_1"]
    chunks_loop_2 = rewrite["chunks_loop_2"]
    # Les contenus du pool avant rewrite (boucle 1)
    old_contents = {c["content"] for c in chunks_loop_1}
    # Les chunks qui n'étaient PAS dans le pool avant rewrite = apportés par la reformulation
    new_chunks = [c for c in chunks_loop_2 if c["content"] not in old_contents]
    n_new = len(new_chunks)

    # Labelliser la pertinence des nouveaux chunks via LLM-juge
    if new_chunks:
        new_labels = _judge_relevance(meta["question"], meta["reponse_attendue"], new_chunks)
        n_new_pertinents = sum(new_labels)
    else:
        n_new_pertinents = 0

    # Ratio : proportion de nouveaux chunks qui sont effectivement pertinents
    new_ratio = round(n_new_pertinents / n_new, 3) if n_new > 0 else float("nan")

    return {
        "rewrite_triggered": 1,
        "rewrite_new_query": rewrite["new_query"][:150],
        "rewrite_ctx_precision": new_precision,
        "rewrite_precision_gain": _gain(new_precision, baseline_precision),
        "rewrite_ctx_recall": new_recall,
        "rewrite_recall_gain": _gain(new_recall, baseline_recall),
        "rewrite_success": rewrite_success,
        "rewrite_n_new_chunks": n_new,
        "rewrite_n_new_pertinents": n_new_pertinents,
        "rewrite_new_ratio": new_ratio,
        "rewrite_targeted": rewrite_targeted,
        "rewrite_targeting_verdict": targeting_verdict[:200],
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
# ÉVALUATION — 7. DÉDUPLICATION
# =============================================================================

# Prompt LLM-juge pour la dédup : compare directement les deux chunks entre eux,
# sans passer par la question — au lieu de juger le chunk écarté isolément (ancienne
# méthode, trop pessimiste : elle comptait comme "perte" tout chunk écarté pertinent,
# même si son quasi-doublon conservé couvrait déjà la même information).
DEDUP_LOSS_JUDGE = """Voici deux extraits de documents quasi-identiques détectés comme doublons.
Le premier a été ÉCARTÉ, le second a été CONSERVÉ à sa place.

Extrait ÉCARTÉ :
{removed}

Extrait CONSERVÉ (l'a remplacé) :
{kept}

L'extrait ÉCARTÉ contient-il une information qui N'EST PAS déjà présente dans l'extrait CONSERVÉ ?

- Si oui (information perdue), réponds uniquement : OUI
- Si non (l'extrait conservé couvre déjà toute l'information de l'extrait écarté), réponds : NON"""


def _judge_loss(removed: dict, kept: dict) -> int:
    """Compare un chunk écarté par la dédup à celui qui l'a remplacé. Retourne 1 si le LLM
    juge que le chunk écarté contient une information absente du chunk conservé
    (= perte réelle), 0 sinon."""
    prompt = DEDUP_LOSS_JUDGE.format(
        removed=removed["content"][:1500], kept=kept["content"][:1500],
    )
    raw = _invoke_with_retry(prompt).strip()
    return 1 if raw.lower().startswith("oui") else 0


def eval_dedup(entry: dict, meta: dict) -> dict:
    """Évalue la déduplication par comparaison directe de chaque chunk écarté avec le chunk
    conservé qui l'a remplacé (dedup_removed_docs / dedup_kept_docs, même index, ajoutés par
    run_agentic_all.py). Mesure une vraie perte d'information — pas juste la pertinence du
    chunk écarté pris isolément. Si ces champs sont absents (anciens résultats), retourne NaN."""
    removed = entry.get("dedup_removed_docs")
    kept = entry.get("dedup_kept_docs")

    if removed is None:
        # Données générées avant l'ajout de ces champs — on ne peut pas évaluer
        return {
            "dedup_available": 0,
            "dedup_n_retires": float("nan"),
            "dedup_n_pertes": float("nan"),
            "dedup_perte_ratio": float("nan"),
        }

    n_retires = len(removed)
    if n_retires == 0:
        return {
            "dedup_available": 1,
            "dedup_n_retires": 0,
            "dedup_n_pertes": 0,
            "dedup_perte_ratio": 0.0,
        }

    pertes = [_judge_loss(r, k) for r, k in zip(removed, kept)]
    n_pertes = sum(pertes)

    return {
        "dedup_available": 1,
        "dedup_n_retires": n_retires,
        "dedup_n_pertes": n_pertes,
        "dedup_perte_ratio": round(n_pertes / n_retires, 3),
    }


# =============================================================================
# ÉVALUATION — 8. FUSION RRF
# =============================================================================

def eval_rrf(entry: dict, meta: dict) -> dict:
    """Évalue la fusion RRF : la combinaison sémantique + BM25 est-elle meilleure que chacune
    seule ? Compare le taux de chunks pertinents dans les résultats sémantiques seuls, BM25 seuls,
    et la fusion RRF. Nécessite les champs semantic_docs, bm25_docs et pre_dedup_docs."""
    semantic = entry.get("semantic_docs")
    bm25 = entry.get("bm25_docs")
    rrf = entry.get("pre_dedup_docs")  # résultat de la fusion RRF (avant dédup)

    if semantic is None or bm25 is None or rrf is None:
        # Données générées avant l'ajout de ces champs — on ne peut pas évaluer
        return {
            "rrf_available": 0,
            "rrf_n_semantic": float("nan"),
            "rrf_n_bm25": float("nan"),
            "rrf_n_fusion": float("nan"),
            "rrf_semantic_pertinents": float("nan"),
            "rrf_bm25_pertinents": float("nan"),
            "rrf_fusion_pertinents": float("nan"),
            "rrf_semantic_ratio": float("nan"),
            "rrf_bm25_ratio": float("nan"),
            "rrf_fusion_ratio": float("nan"),
            "rrf_exclusifs_semantic": float("nan"),
            "rrf_exclusifs_bm25": float("nan"),
            "rrf_communs": float("nan"),
        }

    question = meta["question"]
    reference = meta["reponse_attendue"]

    # Labelliser la pertinence pour chaque source (on prend le top N de chaque, aligné sur le RRF)
    n_rrf = len(rrf)
    sem_top = semantic[:n_rrf] if n_rrf > 0 else semantic
    bm25_top = bm25[:n_rrf] if n_rrf > 0 else bm25

    # Labellisation LLM-juge pour chaque ensemble
    sem_labels = _judge_relevance(question, reference, sem_top) if sem_top else []
    bm25_labels = _judge_relevance(question, reference, bm25_top) if bm25_top else []
    rrf_labels = _judge_relevance(question, reference, rrf) if rrf else []

    sem_pertinents = sum(sem_labels)
    bm25_pertinents = sum(bm25_labels)
    rrf_pertinents = sum(rrf_labels)

    # Ratios de pertinence (% de chunks pertinents dans chaque ensemble)
    sem_ratio = round(sem_pertinents / len(sem_top), 3) if sem_top else float("nan")
    bm25_ratio = round(bm25_pertinents / len(bm25_top), 3) if bm25_top else float("nan")
    rrf_ratio = round(rrf_pertinents / len(rrf), 3) if rrf else float("nan")

    # Analyse de la complémentarité : chunks exclusifs à chaque méthode dans le top RRF
    sem_contents = {c["content"] for c in sem_top}
    bm25_contents = {c["content"] for c in bm25_top}
    rrf_contents = {c["content"] for c in rrf}
    # Chunks du RRF qui viennent uniquement du sémantique (pas dans BM25 top)
    exclusifs_sem = len(rrf_contents & sem_contents - bm25_contents)
    # Chunks du RRF qui viennent uniquement du BM25 (pas dans sémantique top)
    exclusifs_bm25 = len(rrf_contents & bm25_contents - sem_contents)
    # Chunks du RRF présents dans les deux méthodes
    communs = len(rrf_contents & sem_contents & bm25_contents)

    return {
        "rrf_available": 1,
        "rrf_n_semantic": len(sem_top),
        "rrf_n_bm25": len(bm25_top),
        "rrf_n_fusion": len(rrf),
        "rrf_semantic_pertinents": sem_pertinents,
        "rrf_bm25_pertinents": bm25_pertinents,
        "rrf_fusion_pertinents": rrf_pertinents,
        "rrf_semantic_ratio": sem_ratio,
        "rrf_bm25_ratio": bm25_ratio,
        "rrf_fusion_ratio": rrf_ratio,
        "rrf_exclusifs_semantic": exclusifs_sem,
        "rrf_exclusifs_bm25": exclusifs_bm25,
        "rrf_communs": communs,
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

            # ②③ Retrieval + Re-ranking (RAGAS + MRR)
            ret_row = await eval_retrieval_and_reranking(entry, meta, ragas_metrics)
            row.update(ret_row)
            print(f"  ②③ ctx_prec={ret_row['retrieval_ctx_precision']}  "
                  f"ctx_rec={ret_row['retrieval_ctx_recall']}  "
                  f"rerank Δprec={ret_row['rerank_precision_delta']}  "
                  f"Δrec={ret_row['rerank_recall_delta']}")
            print(f"      MRR={ret_row['rerank_post_mrr']}  "
                  f"ΔMRR={ret_row['rerank_mrr_delta']}  "
                  f"pertinents={ret_row['rerank_n_pertinents']}/{ret_row['rerank_n_total']}")

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
                succes_str = "N/A" if math.isnan(rewrite_row["rewrite_success"]) else (
                    "OUI" if rewrite_row["rewrite_success"] else "NON"
                )
                print(f"  ⑤ Rewrite : Δprec={rewrite_row['rewrite_precision_gain']}  "
                      f"Δrec={rewrite_row['rewrite_recall_gain']}  "
                      f"succès={succes_str}  "
                      f"nouveaux={rewrite_row['rewrite_n_new_pertinents']}/{rewrite_row['rewrite_n_new_chunks']} pertinents  "
                      f"ciblage={'OUI' if rewrite_row['rewrite_targeted'] else 'NON'}")
            else:
                print("  ⑤ Rewrite : non déclenché")

            # ⑥ Generation
            gen_row = await eval_generation(entry, meta, ragas_metrics)
            row.update(gen_row)
            print(f"  ⑥ Generation : faithfulness={gen_row['gen_faithfulness']}")

            # ⑦ Déduplication
            dedup_row = eval_dedup(entry, meta)
            row.update(dedup_row)
            if dedup_row["dedup_available"]:
                print(f"  ⑦ Dédup : {dedup_row['dedup_n_retires']} chunk(s) écarté(s), "
                      f"{dedup_row['dedup_n_pertes']} avec perte d'info réelle "
                      f"(ratio={dedup_row['dedup_perte_ratio']})")
            else:
                print("  ⑦ Dédup : données non disponibles (relancez run_agentic_all.py)")

            # ⑧ Fusion RRF
            rrf_row = eval_rrf(entry, meta)
            row.update(rrf_row)
            if rrf_row["rrf_available"]:
                print(f"  ⑧ RRF : sém={rrf_row['rrf_semantic_pertinents']}/{rrf_row['rrf_n_semantic']}  "
                      f"BM25={rrf_row['rrf_bm25_pertinents']}/{rrf_row['rrf_n_bm25']}  "
                      f"fusion={rrf_row['rrf_fusion_pertinents']}/{rrf_row['rrf_n_fusion']}  "
                      f"(excl_sém={rrf_row['rrf_exclusifs_semantic']} excl_bm25={rrf_row['rrf_exclusifs_bm25']} "
                      f"communs={rrf_row['rrf_communs']})")
            else:
                print("  ⑧ RRF : données non disponibles (relancez run_agentic_all.py)")

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
        print(f"\n  ③ RE-RANKING — RAGAS")
        print(f"     Context Precision avant  : {pre_p:.3f}")
        print(f"     Context Precision après  : {post_p:.3f}")
        print(f"     Δ Precision              : {delta_p:+.3f}")
        print(f"     Context Recall avant     : {pre_r:.3f}")
        print(f"     Context Recall après     : {post_r:.3f}")
        print(f"     Δ Recall                 : {delta_r:+.3f}")

    if "rerank_post_mrr" in df.columns:
        pre_mrr = df["rerank_pre_mrr"].mean()
        post_mrr = df["rerank_post_mrr"].mean()
        delta_mrr = df["rerank_mrr_delta"].mean()
        n_pert = df["rerank_n_pertinents"].sum()
        n_tot = df["rerank_n_total"].sum()
        print(f"\n  ③ RE-RANKING — MRR (LLM-juge)")
        print(f"     MRR avant                : {pre_mrr:.3f}")
        print(f"     MRR après                : {post_mrr:.3f}")
        print(f"     Δ MRR                    : {delta_mrr:+.3f}")
        print(f"     Chunks pertinents        : {n_pert}/{n_tot} ({n_pert/n_tot:.1%})" if n_tot > 0 else "")

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
            # Taux de réussite et nouveaux chunks utiles
            if "rewrite_success" in triggered.columns:
                success_rate = triggered["rewrite_success"].mean()
                print(f"     Taux de réussite      : {success_rate:.1%} (grading OUI après rewrite)")
            if "rewrite_n_new_chunks" in triggered.columns:
                total_new = triggered["rewrite_n_new_chunks"].sum()
                total_new_pert = triggered["rewrite_n_new_pertinents"].sum()
                ratio = total_new_pert / total_new if total_new > 0 else float("nan")
                print(f"     Nouveaux chunks       : {int(total_new)} au total")
                print(f"     Nouveaux pertinents   : {int(total_new_pert)} ({ratio:.1%})")
            # Qualité du ciblage de la reformulation (vise-t-elle bien le manque identifié ?)
            if "rewrite_targeted" in triggered.columns:
                targeting_rate = triggered["rewrite_targeted"].mean()
                print(f"     Reformulation ciblée  : {targeting_rate:.1%} (vise bien le manque identifié)")
        else:
            print(f"     Jamais déclenché")

    # ⑥ Generation
    if "gen_faithfulness" in df.columns:
        faith = df["gen_faithfulness"].mean()
        print(f"\n  ⑥ GENERATION")
        print(f"     Faithfulness : {faith:.3f}")

    # ⑦ Déduplication
    if "dedup_available" in df.columns:
        dedup_df = df[df["dedup_available"] == 1]
        print(f"\n  ⑦ DÉDUPLICATION ({len(dedup_df)}/{n} questions avec données)")
        if len(dedup_df) > 0:
            avg_retires = dedup_df["dedup_n_retires"].mean()
            total_retires = dedup_df["dedup_n_retires"].sum()
            total_pertes = dedup_df["dedup_n_pertes"].sum()
            perte_glob = total_pertes / total_retires if total_retires > 0 else 0.0
            print(f"     Retirés par dédup     : {int(total_retires)} au total ({avg_retires:.1f}/question)")
            print(f"     Perte d'info réelle   : {int(total_pertes)} ({perte_glob:.1%} des retirés)")

    # ⑧ Fusion RRF
    if "rrf_available" in df.columns:
        rrf_df = df[df["rrf_available"] == 1]
        print(f"\n  ⑧ FUSION RRF ({len(rrf_df)}/{n} questions avec données)")
        if len(rrf_df) > 0:
            sem_r = rrf_df["rrf_semantic_ratio"].mean()
            bm25_r = rrf_df["rrf_bm25_ratio"].mean()
            fus_r = rrf_df["rrf_fusion_ratio"].mean()
            total_excl_sem = rrf_df["rrf_exclusifs_semantic"].sum()
            total_excl_bm25 = rrf_df["rrf_exclusifs_bm25"].sum()
            total_communs = rrf_df["rrf_communs"].sum()
            print(f"     Taux pertinence sémantique : {sem_r:.1%}")
            print(f"     Taux pertinence BM25       : {bm25_r:.1%}")
            print(f"     Taux pertinence fusion RRF : {fus_r:.1%}")
            print(f"     Chunks exclusifs sémantique: {int(total_excl_sem)}")
            print(f"     Chunks exclusifs BM25      : {int(total_excl_bm25)}")
            print(f"     Chunks communs             : {int(total_communs)}")
            # Verdict sur la complémentarité
            if fus_r > sem_r and fus_r > bm25_r:
                print(f"     → La fusion RRF améliore le taux de pertinence")
            elif fus_r >= max(sem_r, bm25_r):
                print(f"     → La fusion RRF maintient le meilleur taux")
            else:
                best = "sémantique" if sem_r > bm25_r else "BM25"
                print(f"     → Le {best} seul fait mieux que la fusion")

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
