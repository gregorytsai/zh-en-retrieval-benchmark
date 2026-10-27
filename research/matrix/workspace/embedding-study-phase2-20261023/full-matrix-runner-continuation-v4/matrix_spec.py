"""Offline specification and invariants for the seven-embedder matrix.

This module performs no I/O at import time and never starts model processes.
Paths in MODEL_REGISTRY are relative to the user-selected workspace root.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable


DATASET_COUNTS = {
    "eng-zho": {"corpus": 430, "queries": 504, "qrels": 504, "grade_mode": "binary"},
    "zho-zho": {"corpus": 443, "queries": 504, "qrels": 504, "grade_mode": "binary"},
    "eng-eng": {"corpus": 978, "queries": 1148, "qrels": 1148, "grade_mode": "binary"},
    "acord-eng-eng": {"corpus": 3931, "queries": 57, "qrels": 61988, "grade_mode": "graded"},
    "miracl-zh-zh": {"corpus": 3786, "queries": 200, "qrels": 1998, "grade_mode": "binary"},
    "miracl-en-en": {"corpus": 7921, "queries": 200, "qrels": 2091, "grade_mode": "binary"},
}

QUERY_ROWS = sum(row["queries"] for row in DATASET_COUNTS.values())
RERANKERS = ("bge-reranker", "qwen-reranker-0.6b", "qwen-reranker-4b")
CONFIGS_PER_EMBEDDER = 1 + len(RERANKERS)
DATASET_COUNT = len(DATASET_COUNTS)
SUMMARIES_PER_EMBEDDER = DATASET_COUNT * CONFIGS_PER_EMBEDDER
ROWS_PER_EMBEDDER = QUERY_ROWS * CONFIGS_PER_EMBEDDER
TOTAL_EMBEDDERS = 7
TOTAL_SUMMARIES = TOTAL_EMBEDDERS * SUMMARIES_PER_EMBEDDER
TOTAL_ROWS = TOTAL_EMBEDDERS * ROWS_PER_EMBEDDER

INSTRUCTION = "Instruct: Given a web search query, retrieve relevant passages\nQuery: "


@dataclass(frozen=True)
class ModelSpec:
    kind: str
    path: str
    bytes: int
    sha256: str
    pooling: str | None = None
    dimension: int | None = None
    query_prefix: str = ""
    document_prefix: str = ""
    prompt: str = ""
    context_tokens: int | None = None  # populated only from independently verified GGUF metadata


MODEL_REGISTRY = {
    "qwen-0.6b": ModelSpec("embedding", "embedding-study-20261022/models/Qwen3-Embedding-0.6B-Q8_0.gguf", 639150592,
        "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439", "last", 1024, prompt=INSTRUCTION),
    "qwen-4b": ModelSpec("embedding", "Qwen3-Embedding-4B.Q8_0.gguf", 4279660672,
        "d520a7537a5b1d516363ec58c31296af1b1377eb011aa4a54af508f2e9d92d7e", "last", 2560, prompt=INSTRUCTION),
    "qwen-8b": ModelSpec("embedding", "embedding-study-20261022/models/Qwen3-Embedding-8B-Q8_0.gguf", 8047105824,
        "d20ddc71e8a5c4344f2343481e242233a997dc5eaff442427a945836c97b4deb", "last", 4096, prompt=INSTRUCTION),
    "bge-m3": ModelSpec("embedding", "embedding-study-20261022/models/bge-m3-q8_0.gguf", 634553760,
        "aa473d51f451a22f0fcf39ba3330c14bed38a385712b1113440f69df4047a173", "cls", 1024),
    "embeddinggemma-2": ModelSpec("embedding", "embedding-study-20261022/models/embeddinggemma-2-Q8_0.gguf", 309855456,
        "2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135", "mean", 768,
        query_prefix="task: search result | query: ", document_prefix="title: none | text: "),
    "f2llm-v2-0.6b": ModelSpec("embedding", "embedding-study-20261022/models/F2LLM-v2-0.6B.Q8_0.gguf", 639448832,
        "6854d9fa3e411f8a93d0f2e691d947bffdbc4bffb601bfe35e3b34c4de4fdbe7", "last", 1024, prompt=INSTRUCTION),
    "granite-97m-r2": ModelSpec("embedding", "embedding-study-20261022/models/granite-embedding-97M-multilingual-r2-Q8_0.gguf", 115061088,
        "25155b89638e501ac33495fa278d551d7545e1e2f62722a499bba1f064c080f2", "cls", 384),
    "bge-reranker": ModelSpec("reranker", "bge-reranker-v2-m3-q8_0.gguf", 635676416,
        "a43c7c9b11a4c1517e5bf95151960e1621d1b72f7a493364b01e386cf1aaa1d3", prompt="bge-fallback-pair-v1"),
    "qwen-reranker-0.6b": ModelSpec("reranker", "qwen-reranker-rebuilt/Qwen3-Reranker-0.6B-Q8_0.gguf", 639153344,
        "21c265fd5e3695f998983645dab7e59eddecde1e61349a68534e96afdc3f8821", prompt="qwen-rerank-chat-v1"),
    "qwen-reranker-4b": ModelSpec("reranker", "qwen-reranker-rebuilt/Qwen3-Reranker-4B-Q8_0.gguf", 4279678912,
        "4178dde89ef52ba4c986269466c2f1150bf4d94ed1b7ec91cb78ed87361794d1", prompt="qwen-rerank-chat-v1"),
}

EMBEDDERS = tuple(name for name, spec in MODEL_REGISTRY.items() if spec.kind == "embedding")
BATCHES = (
    ("B01", "qwen-4b"),
    ("B02", "qwen-8b"),
    ("B03", "embeddinggemma-2"),
    ("B04", "qwen-0.6b"),
    ("B05", "bge-m3"),
    ("B06", "f2llm-v2-0.6b"),
    ("B07", "granite-97m-r2"),
)

# Continuation schedule: B02 is split by measured work (not by model/config)
# so that every configuration still has the same six-dataset scope in the
# combined result. The original BATCHES registry remains unchanged for
# global-matrix enumeration and for audit compatibility with B01.
RUN_BATCHES = (
    ("B02-A", "qwen-8b"),
    ("B02-B", "qwen-8b"),
    ("B03", "embeddinggemma-2"),
    ("B04", "qwen-0.6b"),
    ("B05", "bge-m3"),
    ("B06", "f2llm-v2-0.6b"),
    ("B07", "granite-97m-r2"),
)
BATCH_DATASETS = {
    "B02-A": ("eng-eng",),
    "B02-B": ("eng-zho", "zho-zho", "acord-eng-eng", "miracl-zh-zh", "miracl-en-en"),
    **{batch: tuple(DATASET_COUNTS) for batch, _embedder in BATCHES if batch != "B02"},
}
MODEL_PATHS = {
    "llama-server": "llama.cpp-qwen38/build/bin/llama-server",
}


class MatrixValidationError(ValueError):
    """Raised when frozen matrix inputs violate the declared schema."""


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def enumerate_configs(embedder_ids: Iterable[str] | None = None) -> list[dict[str, str]]:
    selected = list(embedder_ids) if embedder_ids is not None else list(EMBEDDERS)
    unknown = sorted(set(selected) - set(EMBEDDERS))
    if unknown:
        raise MatrixValidationError(f"Unknown embedder(s): {unknown}")
    if len(selected) != len(set(selected)):
        raise MatrixValidationError("Embedder selection must not contain duplicates")
    result = []
    for embedder in selected:
        result.append({"id": f"{embedder}__standalone", "embedder": embedder, "reranker": "standalone"})
        result.extend({"id": f"{embedder}__{reranker}", "embedder": embedder, "reranker": reranker} for reranker in RERANKERS)
    return result


def validate_dataset_mapping(datasets: dict[str, Any], require_full: bool = True) -> dict[str, dict[str, Any]]:
    """Validate the shared six-dataset JSON schema without writing sidecars."""
    if not isinstance(datasets, dict):
        raise MatrixValidationError("datasets must be an object mapping names to records")
    if require_full and set(datasets) != set(DATASET_COUNTS):
        raise MatrixValidationError(f"Expected datasets {sorted(DATASET_COUNTS)}, got {sorted(datasets)}")
    normalized: dict[str, dict[str, Any]] = {}
    for name, raw in datasets.items():
        if name not in DATASET_COUNTS:
            raise MatrixValidationError(f"Unexpected dataset {name}")
        if not isinstance(raw, dict):
            raise MatrixValidationError(f"{name}: dataset record must be an object")
        mode = raw.get("grade_mode")
        if mode not in ("binary", "graded"):
            raise MatrixValidationError(f"{name}: grade_mode must be binary or graded")
        corpus, queries, qrels = raw.get("corpus"), raw.get("queries"), raw.get("qrels")
        if not isinstance(corpus, list) or not isinstance(queries, list) or not isinstance(qrels, dict):
            raise MatrixValidationError(f"{name}: expected corpus/queries lists and qrels mapping")
        doc_ids = [str(row.get("id")) for row in corpus if isinstance(row, dict) and row.get("id") is not None]
        query_ids = [str(row.get("id")) for row in queries if isinstance(row, dict) and row.get("id") is not None]
        if len(doc_ids) != len(corpus) or len(set(doc_ids)) != len(doc_ids):
            raise MatrixValidationError(f"{name}: corpus IDs missing or duplicated")
        if len(query_ids) != len(queries) or len(set(query_ids)) != len(query_ids):
            raise MatrixValidationError(f"{name}: query IDs missing or duplicated")
        if any(not isinstance(row.get("text"), str) or not row["text"].strip() for row in corpus + queries):
            raise MatrixValidationError(f"{name}: every corpus/query record needs nonempty text")
        doc_id_set, query_id_set = set(doc_ids), set(query_ids)
        if set(map(str, qrels)) != query_id_set:
            raise MatrixValidationError(f"{name}: qrels query IDs differ from query IDs")
        qrel_count = 0
        for raw_qid, judgments in qrels.items():
            qid = str(raw_qid)
            if not isinstance(judgments, dict) or not judgments:
                raise MatrixValidationError(f"{name}: qrels for {qid} must be nonempty")
            has_positive = False
            for raw_did, grade in judgments.items():
                did = str(raw_did)
                if did not in doc_id_set or type(grade) is not int:
                    raise MatrixValidationError(f"{name}: invalid qrel {qid}/{did}")
                allowed = (0, 1) if mode == "binary" else (0, 1, 2, 3, 4)
                if grade not in allowed:
                    raise MatrixValidationError(f"{name}: grade {grade} outside {mode} scale")
                has_positive |= grade > 0
                qrel_count += 1
            if not has_positive:
                raise MatrixValidationError(f"{name}: qrels for {qid} have no positive judgment")
        actual = {"corpus": len(corpus), "queries": len(queries), "qrels": qrel_count, "grade_mode": mode}
        expected = DATASET_COUNTS[name]
        if require_full and actual != expected:
            raise MatrixValidationError(f"{name}: count/schema mismatch, expected {expected}, got {actual}")
        normalized[name] = {**raw, "grade_mode": mode}
    return normalized


def validate_and_merge_manifests(first: dict[str, Any], second: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Merge the immutable phase-2 and MIRACL manifests, rejecting overlap."""
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise MatrixValidationError("Each manifest must be a JSON object")
    a, b = first.get("datasets"), second.get("datasets")
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise MatrixValidationError("Each manifest must have top-level datasets mapping")
    overlap = sorted(set(a) & set(b))
    if overlap:
        raise MatrixValidationError(f"Dataset manifests overlap: {overlap}")
    return validate_dataset_mapping({**a, **b})


def rank_candidates(score_by_doc: dict[str, float], k: int = 50) -> list[str]:
    """Stable descending similarity order with document ID as fixed tie-break."""
    if k < 1:
        raise MatrixValidationError("candidate k must be positive")
    if any(not isinstance(score, (int, float)) or not math.isfinite(float(score)) for score in score_by_doc.values()):
        raise MatrixValidationError("candidate scores must be finite numeric values")
    return sorted(score_by_doc, key=lambda doc_id: (-float(score_by_doc[doc_id]), str(doc_id)))[:k]


def candidate_pool_hash(query_ids: list[str], candidates: list[dict[str, Any]]) -> str:
    if len(query_ids) != len(candidates):
        raise MatrixValidationError("candidate count does not match query IDs")
    if [str(row.get("query_id")) for row in candidates] != list(map(str, query_ids)):
        raise MatrixValidationError("candidate rows are missing or out of query order")
    normalized = []
    for row in candidates:
        ids = row.get("ids")
        if not isinstance(ids, list) or len(ids) != 50 or len(set(map(str, ids))) != 50:
            raise MatrixValidationError("each query must have exactly 50 unique candidate IDs")
        normalized.append({"query_id": str(row["query_id"]), "ids": list(map(str, ids))})
    return sha256_json(normalized)


def build_cache_manifest(*, dataset: str, embedder: str, data_sha256: str, model_sha256: str,
                         binary_sha256: str, code_sha256: str, query_ids: list[str],
                         candidates: list[dict[str, Any]], prompt_spec: dict[str, Any]) -> dict[str, Any]:
    if dataset not in DATASET_COUNTS or embedder not in EMBEDDERS:
        raise MatrixValidationError("Unknown dataset or embedder for candidate cache")
    pool_hash = candidate_pool_hash(query_ids, candidates)
    manifest = {
        "schema_version": 1, "dataset": dataset, "embedder": embedder,
        "dataset_sha256": data_sha256, "model_sha256": model_sha256,
        "binary_sha256": binary_sha256, "code_sha256": code_sha256,
        "query_ids": list(map(str, query_ids)), "candidate_k": 50,
        "candidate_pool_sha256": pool_hash, "tie_break": "score_desc_then_doc_id_asc",
        "prompt_spec": prompt_spec,
    }
    manifest["cache_key_sha256"] = sha256_json(manifest)
    return manifest


def verify_cache_manifest(manifest: dict[str, Any], expected: dict[str, Any],
                          candidates: list[dict[str, Any]]) -> None:
    """Fail closed unless all cache provenance and candidate-order hashes match."""
    for key in ("dataset", "embedder", "dataset_sha256", "model_sha256", "binary_sha256",
                "code_sha256", "query_ids", "candidate_k", "tie_break", "prompt_spec"):
        if manifest.get(key) != expected.get(key):
            raise MatrixValidationError(f"candidate cache provenance mismatch: {key}")
    observed_hash = candidate_pool_hash(expected["query_ids"], candidates)
    if manifest.get("candidate_pool_sha256") != observed_hash:
        raise MatrixValidationError("candidate pool IDs/order do not match their manifest hash")
    unsigned = {key: value for key, value in manifest.items() if key != "cache_key_sha256"}
    if manifest.get("cache_key_sha256") != sha256_json(unsigned):
        raise MatrixValidationError("candidate cache key hash is invalid")


def evaluate_ranking(order: list[str], candidate_ids: list[str], qrels: dict[str, int], grade_mode: str) -> dict[str, float | int]:
    """Binary MLQA/MIRACL or graded exp2-gain ACORD metrics with judged coverage."""
    if grade_mode not in ("binary", "graded"):
        raise MatrixValidationError(f"Unknown grade mode: {grade_mode}")
    judgments = {str(doc): int(grade) for doc, grade in qrels.items()}
    if not judgments or not any(value > 0 for value in judgments.values()):
        raise MatrixValidationError("Every query needs at least one judged positive")
    final = list(map(str, order[:50]))
    candidates = list(map(str, candidate_ids[:50]))
    relevant = {doc for doc, grade in judgments.items() if grade > 0}
    gain = (lambda grade: float(grade > 0)) if grade_mode == "binary" else (lambda grade: float(2 ** grade - 1))
    dcg = sum(gain(judgments.get(doc, 0)) / math.log2(rank + 1)
              for rank, doc in enumerate(final[:10], 1))
    ideal = sorted(judgments.values(), reverse=True)[:10]
    idcg = sum(gain(grade) / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    hits = [rank for rank, doc in enumerate(final[:10], 1) if doc in relevant]

    def coverage(ids: list[str], k: int) -> float:
        top = ids[:k]
        return len(set(top) & judgments.keys()) / len(top) if top else 0.0

    result: dict[str, float | int] = {
        "top1": float(bool(final) and final[0] in relevant),
        "mrr10": 1.0 / hits[0] if hits else 0.0,
        "ndcg10": dcg / idcg if idcg else 0.0,
        "recall10": len(set(final[:10]) & relevant) / len(relevant),
        "recall50": len(set(final[:50]) & relevant) / len(relevant),
        "hit10": float(bool(hits)),
        "candidate_recall50": len(set(candidates) & relevant) / len(relevant),
        "judged_coverage10": coverage(final, 10),
        "judged_coverage50": coverage(final, 50),
        "candidate_judged_coverage50": coverage(candidates, 50),
        "relevant_count": len(relevant),
        "judged_count": len(judgments),
    }
    if grade_mode == "graded":
        for threshold in range(1, 5):
            result[f"p5_grade{threshold}"] = sum(judgments.get(doc, 0) >= threshold for doc in final[:5]) / 5
    return result


def build_run_manifest(*, workspace_label: str, datasets_sha256: dict[str, str],
                       model_sha256: dict[str, str], binary_sha256: str,
                       code_sha256: dict[str, str], runtime: dict[str, Any]) -> dict[str, Any]:
    """Create the immutable hash envelope used to validate resume/reuse."""
    manifest = {
        "schema_version": 1,
        "scope": "full-matrix-seven-embedder-four-reranker-types",
        "workspace_label": workspace_label,
        "datasets_sha256": dict(sorted(datasets_sha256.items())),
        "model_sha256": dict(sorted(model_sha256.items())),
        "binary_sha256": binary_sha256,
        "code_sha256": dict(sorted(code_sha256.items())),
        "runtime": runtime,
        "embedder_ids": list(EMBEDDERS),
        "reranker_ids": ["standalone", *RERANKERS],
        "dataset_ids": list(DATASET_COUNTS),
        "candidate_k": 50,
        "standalone_final_k": 50,
        "tie_break": "score_desc_then_doc_id_asc",
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def verify_resume_manifest(saved: dict[str, Any], current: dict[str, Any]) -> None:
    """Resume is allowed only for byte-for-byte compatible frozen inputs/config."""
    for label, manifest in (("saved", saved), ("current", current)):
        unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        if manifest.get("manifest_sha256") != sha256_json(unsigned):
            raise MatrixValidationError(f"{label} run manifest self-hash is invalid")
    if saved != current:
        changed = sorted(key for key in set(saved) | set(current) if saved.get(key) != current.get(key))
        raise MatrixValidationError(f"resume manifest changed: {changed}")


def equal_weight_macro(config_rows: list[dict[str, Any]], expected_query_counts: dict[str, int] | None = None) -> dict[str, Any]:
    """Compute six-dataset macro nDCG only for complete, error-free configurations."""
    counts = expected_query_counts or {name: spec["queries"] for name, spec in DATASET_COUNTS.items()}
    by_dataset: dict[str, dict[str, Any]] = {}
    for row in config_rows:
        dataset = row.get("dataset")
        if dataset in by_dataset:
            raise MatrixValidationError(f"Duplicate summary for dataset {dataset}")
        by_dataset[dataset] = row
    missing = sorted(set(counts) - set(by_dataset))
    extras = sorted(set(by_dataset) - set(counts))
    eligible = not missing and not extras
    reasons = []
    if missing:
        eligible = False
        reasons.append("missing dataset summaries")
    if extras:
        eligible = False
        reasons.append("unexpected dataset summaries")
    values = []
    for dataset, expected_count in counts.items():
        row = by_dataset.get(dataset)
        if row is None:
            continue
        if type(row.get("queries")) is not int or type(row.get("errors")) is not int:
            raise MatrixValidationError(f"Summary for {dataset} must explicitly contain integer queries and errors")
        if row["queries"] != expected_count or row["errors"] != 0:
            eligible = False
            reasons.append(f"incomplete or errored summary: {dataset}")
        value = row.get("ndcg10")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise MatrixValidationError(f"nDCG@10 for {dataset} must be finite and within [0, 1]")
        values.append(float(value))
    return {
        "eligible": eligible,
        "dataset_count_expected": len(counts),
        "dataset_count_present": len(by_dataset),
        "macro_ndcg10_equal_weight": sum(values) / len(counts) if eligible else None,
        "coverage": {name: by_dataset[name].get("queries", 0) for name in sorted(by_dataset)},
        "reasons": reasons,
    }
