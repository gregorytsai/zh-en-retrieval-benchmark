#!/usr/bin/env python3
"""Phase 2 retrieval core. Execution is deliberately owned by run_guarded.py."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import statistics
import struct
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PHASE2_DIR = Path(__file__).resolve().parent
WORKSPACE = PHASE2_DIR.parent.parent
BINARY = WORKSPACE / "llama.cpp-qwen38/build/bin/llama-server"
LEGACY_BENCHMARK = PHASE2_DIR / "server_helper.py"
LEGACY_REQUIRED = ("LlamaServer", "json_request", "get_json", "sha256_file")
BGE_RERANKER_SHA256 = "a43c7c9b11a4c1517e5bf95151960e1621d1b72f7a493364b01e386cf1aaa1d3"
BGE_RERANKER_SIZE = 635676416


def _load_legacy_benchmark():
    """Load only the previously validated HTTP/server helper by a stable path."""
    if not LEGACY_BENCHMARK.is_file():
        raise RuntimeError(f"Required legacy server helper is missing: {LEGACY_BENCHMARK}")
    spec = importlib.util.spec_from_file_location("phase2_legacy_benchmark_core", LEGACY_BENCHMARK)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load legacy server helper: {LEGACY_BENCHMARK}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [name for name in LEGACY_REQUIRED if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"Legacy server helper is incompatible; missing {missing}")
    return module


base = _load_legacy_benchmark()

from matrix_spec import MODEL_REGISTRY, DATASET_COUNTS, enumerate_configs

def configure(workspace: Path, binary: Path | None = None):
    global WORKSPACE, BINARY, EMBEDDERS, RERANKERS, PILOT_CONFIGS
    WORKSPACE = workspace.resolve()
    BINARY = binary.resolve() if binary else WORKSPACE / "llama.cpp-qwen38/build/bin/llama-server"
    EMBEDDERS = {name: (WORKSPACE / spec.path, spec.pooling, spec.dimension,
                       spec.query_prefix or spec.prompt, spec.document_prefix, True)
                 for name, spec in MODEL_REGISTRY.items() if spec.kind == "embedding"}
    RERANKERS = {name: WORKSPACE / spec.path for name, spec in MODEL_REGISTRY.items() if spec.kind == "reranker"}
    PILOT_CONFIGS = {c["id"]: (c["embedder"], c["reranker"]) for c in enumerate_configs()}

configure(WORKSPACE)

FIXTURES = [
    ("太陽能電池如何發電？", "Solar cells convert sunlight into electricity.", "Espresso is made from ground coffee."),
    ("水的化學式是什麼？", "The chemical formula of water is H2O.", "Saturn has rings of ice and rock."),
    ("哪個行星以紅色聞名？", "Mars is known as the red planet.", "Bread is baked from flour and yeast."),
    ("光合作用需要什麼能量？", "Photosynthesis uses energy from sunlight.", "A bicycle usually has two wheels."),
    ("蝙蝠如何在黑暗中導航？", "Bats use echolocation to navigate in the dark.", "Copper is used in electrical wiring."),
    ("What do bees produce?", "Bees produce honey.", "Earthquakes are caused by movements of tectonic plates."),
    ("How many legs does a spider have?", "A spider has eight legs.", "The ocean contains salt water."),
    ("冰融化後變成什麼？", "冰融化後會變成液態水。", "火車在鐵軌上行駛。"),
    ("植物的根有什麼作用？", "植物的根會吸收土壤中的水和養分。", "電腦鍵盤用來輸入文字。"),
    ("誰寫了哈姆雷特？", "《哈姆雷特》的作者是莎士比亞。", "長頸鹿是脖子很長的動物。"),
]
METRICS = [
    "top1", "mrr10", "ndcg10", "recall10", "recall50", "hit10", "candidate_recall50",
    "p5", "judged_coverage10", "judged_coverage50", "candidate_judged_coverage50",
]


class DataValidationError(ValueError):
    pass


class TokenLimitError(RuntimeError):
    def __init__(self, issues: list[dict[str, Any]]):
        self.issues = issues
        super().__init__(f"Token preflight blocked {len(issues)} model/dataset input set(s); see token-audit.jsonl")


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def append(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def sha256_file(path: Path) -> str:
    return base.sha256_file(path)


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_bge_reranker_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate the pinned BGE model signature and derive fallback pair overhead."""
    expected = {
        "general.architecture": "bert",
        "tokenizer.ggml.model": "t5",
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 2,
        "tokenizer.ggml.add_bos_token": True,
        "tokenizer.ggml.add_eos_token": True,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items() if metadata.get(key) != value
    }
    add_sep = metadata.get("tokenizer.ggml.add_sep_token", False)
    if add_sep is not False:
        mismatches["tokenizer.ggml.add_sep_token"] = {"expected": False, "actual": add_sep}
    template_keys = sorted(key for key in metadata if "chat_template" in key or "rerank" in key)
    if template_keys:
        mismatches["template_metadata_keys"] = {"expected": [], "actual": template_keys}
    if mismatches:
        raise RuntimeError(f"BGE GGUF metadata differs from the audited fallback signature: {mismatches}")
    overhead = int(metadata["tokenizer.ggml.add_bos_token"]) + int(metadata["tokenizer.ggml.add_eos_token"]) * 2
    if overhead != 3:
        raise RuntimeError(f"BGE fallback pair overhead changed: expected 3, found {overhead}")
    return {
        "architecture": metadata["general.architecture"],
        "tokenizer_model": metadata["tokenizer.ggml.model"],
        "bos_token_id": metadata["tokenizer.ggml.bos_token_id"],
        "eos_token_id": metadata["tokenizer.ggml.eos_token_id"],
        "add_bos_token": metadata["tokenizer.ggml.add_bos_token"],
        "add_eos_token": metadata["tokenizer.ggml.add_eos_token"],
        "add_sep_token": add_sep,
        "chat_template_keys": template_keys,
        "fallback_pair_special_tokens": {"bos": 1, "query_eos": 1, "document_eos": 1, "separator": 0},
        "fallback_pair_overhead": overhead,
    }


def verify_bge_reranker_model(path: Path) -> dict[str, Any]:
    """Check pinned model bytes and GGUF metadata before exact BGE token auditing."""
    if path.stat().st_size != BGE_RERANKER_SIZE:
        raise RuntimeError(f"BGE model size changed: {path.stat().st_size} != {BGE_RERANKER_SIZE}")
    digest = sha256_file(path)
    if digest != BGE_RERANKER_SHA256:
        raise RuntimeError(f"BGE model SHA256 changed: {digest} != {BGE_RERANKER_SHA256}")
    metadata: dict[str, Any] = {}
    scalar_formats = {
        0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
        4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
        10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
    }
    selected = {
        "general.architecture", "tokenizer.ggml.model", "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id", "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token", "tokenizer.ggml.add_sep_token",
    }
    def exact_read(handle, size: int) -> bytes:
        value = handle.read(size)
        if len(value) != size:
            raise RuntimeError("truncated GGUF metadata")
        return value
    def read_u32(handle) -> int:
        return struct.unpack("<I", exact_read(handle, 4))[0]
    def read_u64(handle) -> int:
        return struct.unpack("<Q", exact_read(handle, 8))[0]
    def read_string(handle) -> str:
        size = read_u64(handle)
        if size > 1_073_741_824:
            raise RuntimeError(f"unreasonable GGUF string length {size}")
        return exact_read(handle, size).decode("utf-8")
    try:
        with path.open("rb") as handle:
            if exact_read(handle, 4) != b"GGUF":
                raise RuntimeError("invalid GGUF magic")
            version = read_u32(handle)
            if version not in (2, 3):
                raise RuntimeError(f"unsupported GGUF version {version}")
            _tensor_count = read_u64(handle)
            metadata_count = read_u64(handle)
            if metadata_count > 1_000_000:
                raise RuntimeError(f"unreasonable GGUF metadata count {metadata_count}")
            for _ in range(metadata_count):
                key = read_string(handle)
                value_type = read_u32(handle)
                keep = key in selected or "chat_template" in key or "rerank" in key
                if value_type == 8:  # STRING
                    value: Any = read_string(handle)
                elif value_type == 9:  # ARRAY
                    item_type, count = read_u32(handle), read_u64(handle)
                    if count > 1_073_741_824:
                        raise RuntimeError(f"unreasonable GGUF array length {count}")
                    if item_type == 8:
                        value = [read_string(handle) for _ in range(count)] if keep else None
                        if not keep:
                            # Strings are length-prefixed; skip without materializing them.
                            for _ in range(count):
                                size = read_u64(handle)
                                handle.seek(size, 1)
                    elif item_type in scalar_formats:
                        fmt, width = scalar_formats[item_type]
                        if keep:
                            value = [struct.unpack(fmt, exact_read(handle, width))[0] for _ in range(count)]
                        else:
                            handle.seek(width * count, 1)
                    else:
                        raise RuntimeError(f"unsupported GGUF array type {item_type} for {key}")
                elif value_type in scalar_formats:
                    fmt, width = scalar_formats[value_type]
                    value = struct.unpack(fmt, exact_read(handle, width))[0]
                else:
                    raise RuntimeError(f"unsupported GGUF value type {value_type} for {key}")
                if keep:
                    metadata[key] = value
    except Exception as exc:
        raise RuntimeError(f"Unable to read BGE GGUF metadata: {exc}") from exc
    result = validate_bge_reranker_metadata(metadata)
    return {"model_sha256": digest, "model_bytes": BGE_RERANKER_SIZE, **result}


def validate_datasets(payload: dict[str, Any], require_expected: bool = True) -> dict[str, dict[str, Any]]:
    """Validate the frozen datasets.json format and return canonical string IDs."""
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), dict):
        raise DataValidationError("datasets.json must contain a top-level 'datasets' mapping")
    supplied = payload["datasets"]
    if require_expected and set(supplied) != set(DATASET_COUNTS):
        raise DataValidationError(f"Expected datasets {sorted(DATASET_COUNTS)}, got {sorted(supplied)}")
    result: dict[str, dict[str, Any]] = {}
    for name, raw in supplied.items():
        if not isinstance(raw, dict):
            raise DataValidationError(f"{name}: dataset value must be an object")
        grade_mode = raw.get("grade_mode")
        if grade_mode not in ("binary", "graded"):
            raise DataValidationError(f"{name}: grade_mode must be binary or graded")
        corpus = raw.get("corpus")
        queries = raw.get("queries")
        qrels = raw.get("qrels")
        if not isinstance(corpus, list) or not isinstance(queries, list) or not isinstance(qrels, dict):
            raise DataValidationError(f"{name}: expected corpus/queries lists and qrels mapping")

        normalized_corpus = []
        doc_ids: set[str] = set()
        for item in corpus:
            if not isinstance(item, dict) or "id" not in item or not isinstance(item.get("text"), str) or not item["text"].strip():
                raise DataValidationError(f"{name}: corpus records need id and nonempty text")
            doc_id = str(item["id"])
            if doc_id in doc_ids:
                raise DataValidationError(f"{name}: duplicate corpus id {doc_id}")
            doc_ids.add(doc_id)
            normalized_corpus.append({**item, "id": doc_id})

        normalized_queries = []
        query_ids: set[str] = set()
        for item in queries:
            if not isinstance(item, dict) or "id" not in item or not isinstance(item.get("text"), str) or not item["text"].strip():
                raise DataValidationError(f"{name}: query records need id and nonempty text")
            query_id = str(item["id"])
            if query_id in query_ids:
                raise DataValidationError(f"{name}: duplicate query id {query_id}")
            query_ids.add(query_id)
            normalized_queries.append({**item, "id": query_id})

        normalized_qrels: dict[str, dict[str, int]] = {}
        for raw_qid, judgments in qrels.items():
            qid = str(raw_qid)
            if qid in normalized_qrels or not isinstance(judgments, dict):
                raise DataValidationError(f"{name}: duplicate query qrels or non-mapping qrels for {qid}")
            normalized_judgments: dict[str, int] = {}
            for raw_did, grade in judgments.items():
                did = str(raw_did)
                if did not in doc_ids:
                    raise DataValidationError(f"{name}: qrels document {did} is absent from corpus")
                if type(grade) is not int:
                    raise DataValidationError(f"{name}: qrels grade for {qid}/{did} must be an integer")
                allowed = (0, 1) if grade_mode == "binary" else (0, 1, 2, 3, 4)
                if grade not in allowed:
                    raise DataValidationError(f"{name}: invalid {grade_mode} grade {grade} for {qid}/{did}")
                normalized_judgments[did] = grade
            if not any(grade > 0 for grade in normalized_judgments.values()):
                raise DataValidationError(f"{name}: query {qid} has no positive qrel")
            normalized_qrels[qid] = normalized_judgments
        if set(normalized_qrels) != query_ids:
            missing = sorted(query_ids - set(normalized_qrels))[:5]
            extra = sorted(set(normalized_qrels) - query_ids)[:5]
            raise DataValidationError(f"{name}: qrels/query IDs differ (missing={missing}, extra={extra})")

        pilot_ids = raw.get("pilot_query_ids")
        if pilot_ids is not None:
            pilot_ids = [str(value) for value in pilot_ids]
            if len(set(pilot_ids)) != len(pilot_ids) or not set(pilot_ids) <= query_ids:
                raise DataValidationError(f"{name}: pilot_query_ids must be unique known query IDs")
        result[name] = {
            "corpus": normalized_corpus,
            "queries": normalized_queries,
            "qrels": normalized_qrels,
            "grade_mode": grade_mode,
            "metadata": dict(raw.get("metadata") or {}),
            "pilot_query_ids": pilot_ids,
        }
        if require_expected:
            expected = DATASET_COUNTS.get(name)
            if expected is None:
                raise DataValidationError(f"Unexpected dataset {name}")
            observed = {
                "corpus": len(normalized_corpus),
                "queries": len(normalized_queries),
                "qrels": sum(map(len, normalized_qrels.values())),
                "grade_mode": grade_mode,
            }
            if observed != expected:
                raise DataValidationError(f"{name}: count mismatch, expected {expected}, observed {observed}")
    return result


def load_datasets(path: Path, require_expected: bool = True) -> tuple[dict[str, dict[str, Any]], str]:
    if not path.is_file():
        raise DataValidationError(f"Dataset manifest does not exist: {path}")
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes)
    datasets = validate_datasets(payload, require_expected=require_expected)
    digest = hashlib.sha256(raw_bytes).hexdigest()
    validation_path = path.with_name("data-validation.json")
    if validation_path.is_file():
        evidence = json.loads(validation_path.read_text(encoding="utf-8"))
        expected_digest = evidence.get("datasets_sha256")
        if expected_digest and expected_digest != digest:
            raise DataValidationError("data-validation.json does not match datasets.json SHA-256")
    return datasets, digest


def select_pilot_ids(dataset: dict[str, Any], requested_n: int | None = None) -> list[str]:
    queries = dataset["queries"]
    configured = dataset.get("pilot_query_ids")
    if configured and (requested_n is None or requested_n == len(configured)):
        return list(configured)
    n = requested_n if requested_n is not None else 3
    if n < 1 or n > len(queries):
        raise DataValidationError(f"Pilot query count {n} is outside 1..{len(queries)}")
    ordered = sorted(queries, key=lambda q: (len(q["text"]), hashlib.sha256(str(q["id"]).encode()).hexdigest(), q["id"]))
    if n == 1:
        return [ordered[len(ordered) // 2]["id"]]
    # Evenly spaced short-to-long quantiles, rather than the first N manifest rows.
    indices = [round(i * (len(ordered) - 1) / (n - 1)) for i in range(n)]
    return [ordered[i]["id"] for i in indices]


def query_metadata(query: dict[str, Any]) -> dict[str, Any]:
    nested = query.get("metadata") if isinstance(query.get("metadata"), dict) else {}
    return {
        "cluster_id": query.get("cluster_id", nested.get("cluster_id")),
        "cluster_unit": query.get("cluster_unit", nested.get("cluster_unit")),
        "previous100": query.get("previous100", nested.get("previous100")),
    }


def metrics(order: Iterable[str], candidates: Iterable[str], qrels: dict[str, int], grade_mode: str) -> dict[str, float | int]:
    """Binary MLQA metrics and exp2-gain ACORD nDCG with judged coverage."""
    order = [str(value) for value in order]
    candidates = [str(value) for value in candidates]
    judgments = {str(doc): int(grade) for doc, grade in qrels.items()}
    relevant = {doc for doc, grade in judgments.items() if grade > 0}
    if not relevant:
        raise ValueError("Each query needs at least one judged relevant document")
    if grade_mode not in ("binary", "graded"):
        raise ValueError(f"Unknown grade mode: {grade_mode}")
    gains = (lambda grade: float(grade > 0)) if grade_mode == "binary" else (lambda grade: float((2 ** grade) - 1))
    dcg = sum(gains(judgments.get(doc, 0)) / math.log2(rank + 1) for rank, doc in enumerate(order[:10], 1))
    ideal_grades = sorted(judgments.values(), reverse=True)[:10]
    idcg = sum(gains(grade) / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades, 1))
    hits = [rank for rank, doc in enumerate(order[:10], 1) if doc in relevant]

    def coverage(items: list[str], k: int) -> float:
        top = items[:k]
        return len(set(top) & judgments.keys()) / len(top) if top else 0.0

    result: dict[str, float | int] = {
        "top1": float(bool(order) and order[0] in relevant),
        "mrr10": 1 / hits[0] if hits else 0.0,
        "ndcg10": dcg / idcg if idcg else 0.0,
        "recall10": len(set(order[:10]) & relevant) / len(relevant),
        "recall50": len(set(order[:50]) & relevant) / len(relevant),
        "hit10": float(bool(hits)),
        "candidate_recall50": len(set(candidates[:50]) & relevant) / len(relevant),
        "p5": len(set(order[:5]) & relevant) / 5,
        "judged_coverage10": coverage(order, 10),
        "judged_coverage50": coverage(order, 50),
        "candidate_judged_coverage50": coverage(candidates, 50),
        "relevant_count": len(relevant),
        "judged_count": len(judgments),
    }
    if grade_mode == "graded":
        for threshold in range(1, 5):
            result[f"p5_grade{threshold}"] = sum(judgments.get(doc, 0) >= threshold for doc in order[:5]) / 5
    return result


class Server(base.LlamaServer):
    def __init__(self, model: Path, mode: str, name: str, deadline: float, pooling: str = "last"):
        super().__init__(model, mode, BINARY, name, ctx_size=8192)
        self.pooling = pooling
        self.wall_deadline = deadline

    def _command(self, port: int) -> list[str]:
        command = super()._command(port)
        for key, value in (("--batch-size", "4096"), ("--ubatch-size", "4096"),
                           ("--log-verbosity", "4"), ("--threads", "8"), ("--threads-batch", "8")):
            command[command.index(key) + 1] = value
        if self.mode == "embedding":
            command[command.index("--pooling") + 1] = self.pooling
        if "--n-gpu-layers" not in command:
            command += ["--n-gpu-layers", "all"]
        return command

    def start(self, timeout_s: float = 1200.0) -> float:
        remaining = self.wall_deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("Original wall-clock deadline reached before model startup")
        return super().start(timeout_s=min(timeout_s, remaining))


def request(server: Server, endpoint: str, body: dict[str, Any], timeout: float = 180.0) -> tuple[dict[str, Any], float]:
    return base.json_request(server.url + endpoint, body, timeout=timeout)


def token_count(server: Server, text: str, *, raw_reranker_input: bool = False) -> int:
    body = {"content": text, "add_special": not raw_reranker_input}
    if raw_reranker_input:
        # format_prompt_rerank tokenizes query/document independently with
        # add_special=false and parse_special=false when no rerank template exists.
        body["parse_special"] = False
    response, _ = request(server, "/tokenize", body)
    return len(response["tokens"])


def embed(server: Server, texts: list[str], dim: int) -> tuple[np.ndarray, float]:
    response, elapsed = request(server, "/v1/embeddings", {"input": texts, "encoding_format": "float"})
    data = sorted(response["data"], key=lambda row: row["index"])
    if [row["index"] for row in data] != list(range(len(texts))):
        raise RuntimeError("Embedding response indices do not match request order")
    vectors = np.asarray([row["embedding"] for row in data], dtype=np.float32)
    if vectors.shape != (len(texts), dim) or not np.isfinite(vectors).all():
        raise RuntimeError(f"Invalid embedding response shape/content: {vectors.shape}, expected {(len(texts), dim)}")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.all(np.abs(norms - 1) < 0.01):
        raise RuntimeError(f"Embedding vectors are not normalized: {norms}")
    server.sample_rss()
    return vectors / norms[:, None], elapsed


def rerank(server: Server, query: str, documents: list[str]) -> tuple[list[int], float, list[float]]:
    response, elapsed = request(server, "/v1/rerank", {"query": query, "documents": documents, "top_n": len(documents)})
    results = response.get("results")
    if not isinstance(results, list) or sorted(row["index"] for row in results) != list(range(len(documents))):
        raise RuntimeError("Reranker did not return exactly one score for every candidate")
    scored = [(int(row["index"]), float(row.get("relevance_score", row.get("score")))) for row in results]
    if not all(math.isfinite(score) for _, score in scored):
        raise RuntimeError("Reranker returned non-finite scores")
    by_index = [0.0] * len(documents)
    for index, score in scored:
        by_index[index] = score
    order = [index for index, _ in sorted(scored, key=lambda item: (-item[1], item[0]))]
    server.sample_rss()
    return order, elapsed, by_index


def audit_token_limits(server: Server, dataset_name: str, data: dict[str, Any], model_name: str,
                       model_kind: str, spec: tuple[Any, ...], output: Path,
                       check_time, max_tokens: int = 4096,
                       reranker_template_reserve: int = 512,
                       bge_template: dict[str, Any] | None = None) -> dict[str, Any]:
    """Audit all source docs and queries without issuing embedding/rerank inference."""
    corpus = data["corpus"]
    queries = data["queries"]
    doc_lengths: list[tuple[str, int]] = []
    query_lengths: list[tuple[str, int]] = []
    if model_kind == "embedding":
        _, pooling, dim, query_prefix, document_prefix, _ = spec
        for doc in corpus:
            check_time()
            doc_lengths.append((str(doc["id"]), token_count(server, document_prefix + doc["text"])))
        for query in queries:
            check_time()
            query_lengths.append((str(query["id"]), token_count(server, query_prefix + query["text"])))
        doc_limit = max_tokens
        query_limit = max_tokens
        combined_limit = None
    else:
        exact_bge = model_name == "bge-reranker"
        if exact_bge and (not bge_template or bge_template.get("fallback_pair_overhead") != 3):
            raise RuntimeError("BGE exact token audit requires a verified three-token fallback template")
        for doc in corpus:
            check_time()
            doc_lengths.append((str(doc["id"]), token_count(server, doc["text"], raw_reranker_input=exact_bge)))
        for query in queries:
            check_time()
            query_lengths.append((str(query["id"]), token_count(server, query["text"], raw_reranker_input=exact_bge)))
        doc_limit = max_tokens
        query_limit = max_tokens
        combined_limit = max_tokens - (bge_template["fallback_pair_overhead"] if exact_bge else reranker_template_reserve)

    max_doc_id, max_doc = max(doc_lengths, key=lambda item: item[1])
    max_query_id, max_query = max(query_lengths, key=lambda item: item[1])
    issues = []
    for doc_id, count in doc_lengths:
        if count >= doc_limit:
            issues.append({"kind": "document", "id": doc_id, "tokens": count, "limit": doc_limit})
    for query_id, count in query_lengths:
        if count >= query_limit:
            issues.append({"kind": "query", "id": query_id, "tokens": count, "limit": query_limit})
    combined_overhead = (bge_template["fallback_pair_overhead"] if model_name == "bge-reranker"
                         else reranker_template_reserve)
    if combined_limit is not None and max_doc + max_query + combined_overhead > max_tokens:
        issues.append({
            "kind": "reranker_template_combined_exact" if model_name == "bge-reranker"
                    else "reranker_template_combined_upper_bound", "document_id": max_doc_id,
            "query_id": max_query_id, "document_tokens": max_doc, "query_tokens": max_query,
            "template_reserve": combined_overhead, "special_token_overhead": combined_overhead,
            "limit": max_tokens, "upper_bound": max_doc + max_query + combined_overhead,
        })
    record = {
        "time": stamp(), "dataset": dataset_name, "model": model_name, "model_kind": model_kind,
        "corpus_documents": len(doc_lengths), "queries": len(query_lengths),
        "max_document": {"id": max_doc_id, "tokens": max_doc},
        "max_query": {"id": max_query_id, "tokens": max_query},
        "max_tokens": max_tokens,
        "reranker_template_reserve": (reranker_template_reserve if model_kind == "reranker" and model_name != "bge-reranker" else None),
        "reranker_special_token_overhead": bge_template.get("fallback_pair_overhead") if model_name == "bge-reranker" else None,
        "bge_template_verification": bge_template if model_name == "bge-reranker" else None,
        "truncated": False, "status": "blocked" if issues else "ok", "issues": issues,
    }
    append(output, record)
    return record


def select_by_ids(data: dict[str, Any], query_ids: list[str] | None) -> list[dict[str, Any]]:
    if query_ids is None:
        return list(data["queries"])
    by_id = {query["id"]: query for query in data["queries"]}
    return [by_id[query_id] for query_id in query_ids]


class Study:
    """Single attempt evaluator; orchestration and traffic policy live in run_guarded."""
    def __init__(self, out: Path, data: dict[str, Any], dataset_name: str, deadline: float,
                 check_time, progress):
        self.out = out
        self.data = data
        self.dataset_name = dataset_name
        self.deadline = deadline
        self.check_time = check_time
        self.progress = progress

    @contextmanager
    def server(self, name: str, model: Path, mode: str, pooling: str = "last"):
        self.check_time()
        server = Server(model, mode, name, self.deadline, pooling)
        try:
            with server:
                self.check_time()
                append(self.out / "server-events.jsonl", {
                    "time": stamp(), "model": name, "load_ms": server.load_ms,
                    "command": server._command(server.port), "props": base.get_json(server.url + "/props"),
                })
                yield server
        finally:
            server.stop()
            append(self.out / "server-events.jsonl", {
                "time": stamp(), "model": name, "event": "stopped",
                "sampled_peak_rss_mb": server.peak_rss_mb,
            })

    def gate_embedding(self, server: Server, query_prefix: str, document_prefix: str, dim: int) -> None:
        gates = []
        for query, positive, negative in FIXTURES:
            self.check_time()
            vectors, _ = embed(server, [query_prefix + query, document_prefix + positive, document_prefix + negative], dim)
            good, bad = float(vectors[0] @ vectors[1]), float(vectors[0] @ vectors[2])
            gates.append({"query": query, "positive": good, "negative": bad})
            if good <= bad:
                raise RuntimeError(f"Embedding fixture gate failed: {query!r}, positive={good}, negative={bad}")
        repeat = []
        for _ in range(3):
            self.check_time()
            repeat.append(embed(server, [query_prefix + FIXTURES[0][0]], dim)[0])
        if not all(np.allclose(repeat[0], value, atol=1e-5) for value in repeat[1:]):
            raise RuntimeError("Embedding repeatability gate failed")
        dump(self.out / f"gate-embedding.json", {"fixtures": gates, "repeatability": True})

    def gate_reranker(self, server: Server, model_name: str) -> None:
        """Warm the reranker on all fixed fixtures and fail closed on bad ordering."""
        gates = []
        for query, positive, negative in FIXTURES:
            self.check_time()
            order, _, scores = rerank(server, query, [positive, negative])
            gate = {"query": query, "order": order, "positive_score": scores[0],
                    "negative_score": scores[1], "timing_included": False}
            gates.append(gate)
            if order[0] != 0 or scores[0] <= scores[1]:
                dump(self.out / "gate-reranker.json", {
                    "model": model_name, "status": "blocked", "fixtures": gates,
                    "warmup_count": len(gates), "timing_included": False,
                })
                raise RuntimeError(
                    f"Reranker fixture gate failed for {model_name}: {query!r}, "
                    f"positive={scores[0]}, negative={scores[1]}, order={order}"
                )
        dump(self.out / "gate-reranker.json", {
            "model": model_name, "status": "passed", "fixtures": gates,
            "warmup_count": len(gates), "timing_included": False,
            "policy": "10 fixed bilingual positive/negative pairs; positive must rank first and score higher",
        })

    def standalone(self, model_name: str, queries: list[dict[str, Any]], cache_root: Path) -> list[dict[str, Any]]:
        model, pooling, dim, query_prefix, document_prefix, _ = EMBEDDERS[model_name]
        self.progress("embedding", model=model_name, dataset=self.dataset_name)
        with self.server("embed-" + model_name, model, "embedding", pooling) as server:
            self.gate_embedding(server, query_prefix, document_prefix, dim)
            corpus = self.data["corpus"]
            doc_texts = [document_prefix + doc["text"] for doc in corpus]
            query_texts = [query_prefix + query["text"] for query in queries]
            doc_lengths = []
            for text in doc_texts:
                self.check_time()
                doc_lengths.append(token_count(server, text))
            query_lengths = []
            for text in query_texts:
                self.check_time()
                query_lengths.append(token_count(server, text))
            if max(doc_lengths + query_lengths) >= 4096:
                raise RuntimeError("A document/query exceeded the token preflight despite prior audit")
            append(self.out / "input-token-audit.jsonl", {
                "model": model_name, "dataset": self.dataset_name,
                "max_doc": max(doc_lengths), "max_query": max(query_lengths), "truncated": False,
            })
            corpus_vectors = []
            corpus_ms = 0.0
            start = 0
            while start < len(doc_texts):
                self.check_time()
                end, token_sum = start, 0
                while end < len(doc_texts) and end - start < 8 and token_sum + doc_lengths[end] < 3800:
                    token_sum += doc_lengths[end]
                    end += 1
                if end == start:
                    end += 1
                vectors, elapsed = embed(server, doc_texts[start:end], dim)
                corpus_vectors.append(vectors)
                corpus_ms += elapsed
                start = end
                if start % 100 < 8:
                    self.progress("embedding", model=model_name, dataset=self.dataset_name,
                                  documents_done=start, documents_total=len(doc_texts))
            corpus_matrix = np.concatenate(corpus_vectors)
            for text in query_texts[:10]:
                self.check_time()
                embed(server, [text], dim)  # warmup excluded from per-query timings
            query_vectors, query_ms = [], []
            for text in query_texts:
                self.check_time()
                vector, elapsed = embed(server, [text], dim)
                query_vectors.append(vector[0])
                query_ms.append(elapsed)
            query_matrix = np.asarray(query_vectors, dtype=np.float32)
            ids = [doc["id"] for doc in corpus]
            candidates = []
            output_rows = []
            for query, vector, elapsed in zip(queries, query_matrix, query_ms):
                self.check_time()
                tick = time.perf_counter()
                similarities = corpus_matrix @ vector
                indices = np.lexsort((np.asarray(ids, dtype=str), -similarities))
                ranked = [ids[index] for index in indices]
                retrieval_ms = (time.perf_counter() - tick) * 1000
                final_ids = ranked[:50]
                candidate = {
                    "query_id": query["id"], "ids": final_ids,
                    "scores": [float(similarities[index]) for index in indices[:50]],
                    "embedding_ms": elapsed, "retrieval_ms": retrieval_ms,
                }
                candidates.append(candidate)
                row = {
                    "dataset": self.dataset_name, "embedding": model_name, "reranker": "standalone",
                    "query_id": query["id"], "grade_mode": self.data["grade_mode"], "k": 50,
                    "status": "ok", "embedding_ms": elapsed, "retrieval_ms": retrieval_ms,
                    "rerank_ms": None, "total_ms": elapsed + retrieval_ms,
                    "final_ids": final_ids, **query_metadata(query),
                    **metrics(final_ids, ranked[:50], self.data["qrels"][query["id"]], self.data["grade_mode"]),
                }
                output_rows.append(row)
                append(self.out / "query-results.jsonl", row)
            cache_dir = cache_root / model_name / self.dataset_name
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_dir / "corpus.npy", corpus_matrix)
            np.save(cache_dir / "queries.npy", query_matrix)
            dump(cache_dir / "candidates.json", candidates)
            cache_files = ("corpus.npy", "queries.npy", "candidates.json")
            file_hashes = {name: sha256_file(cache_dir / name) for name in cache_files}
            cache_spec = {
                "dataset": self.dataset_name, "embedding": model_name,
                "dataset_sha256": self.data["_dataset_sha256"], "model_sha256": self.data["_model_hashes"][model_name],
                "query_ids": [query["id"] for query in queries], "candidate_k": 50,
                "stored_final_ids": 50, "pooling": pooling, "dimension": dim,
                "query_prefix": query_prefix, "document_prefix": document_prefix,
                "normalized": True, "corpus_documents": len(corpus),
                "binary_sha256": self.data["_binary_sha256"],
                "code_sha256": self.data["_code_sha256"],
                "runtime": {"ctx_size": 8192, "batch_size": 4096, "ubatch_size": 4096,
                            "threads": 8, "threads_batch": 8, "parallel": 1,
                            "embedding_batch_max_items": 8, "embedding_token_batch_limit": 3800,
                            "gpu_layers": "all", "log_verbosity": 4},
            }
            cache_spec["cache_key_sha256"] = sha256_json(cache_spec)
            cache_spec["files_sha256"] = file_hashes
            dump(cache_dir / "cache-manifest.json", cache_spec)
            append(self.out / "index-stats.jsonl", {
                "model": model_name, "dataset": self.dataset_name, "corpus_docs": len(corpus),
                "corpus_ms": corpus_ms, "docs_per_second": len(corpus) / (corpus_ms / 1000),
                "dimension": dim, "query_prefix": query_prefix, "document_prefix": document_prefix,
                "pooling": pooling, "normalized": True,
            })
            self.progress("embedding_config_complete", model=model_name, dataset=self.dataset_name)
            return output_rows

    def rerank(self, model_name: str, queries: list[dict[str, Any]], candidate_path: Path, embedding_name: str) -> list[dict[str, Any]]:
        cache_started = time.perf_counter()
        model = RERANKERS[model_name]
        candidate_file = candidate_path / embedding_name / self.dataset_name / "candidates.json"
        cache_manifest = json.loads((candidate_path / embedding_name / self.dataset_name / "cache-manifest.json").read_text(encoding="utf-8"))
        expected_ids = [query["id"] for query in queries]
        if cache_manifest.get("query_ids") != expected_ids:
            raise RuntimeError("embedder candidate cache does not match the current query selection")
        if cache_manifest.get("dataset_sha256") != self.data["_dataset_sha256"]:
            raise RuntimeError("embedder candidate cache data hash mismatch")
        if cache_manifest.get("model_sha256") != self.data["_model_hashes"][embedding_name]:
            raise RuntimeError("embedder candidate cache model hash mismatch")
        for filename, digest in cache_manifest["files_sha256"].items():
            if sha256_file(candidate_file.parent / filename) != digest:
                raise RuntimeError("Candidate cache file hash changed")
        candidates = json.loads(candidate_file.read_text(encoding="utf-8"))
        if len(candidates) != len(queries) or [row["query_id"] for row in candidates] != expected_ids:
            raise RuntimeError("embedder candidate file is incomplete or out of order")
        by_id = {doc["id"]: doc["text"] for doc in self.data["corpus"]}
        dump(self.out / "cache-load-stats.json", {
            "embedding": embedding_name, "dataset": self.dataset_name,
            "cache_load_and_verify_ms": (time.perf_counter() - cache_started) * 1000,
            "candidate_k": 50, "included_in_per_query_latency": False})
        output_rows = []
        self.progress("reranking", model=model_name, dataset=self.dataset_name)
        with self.server("main-" + model_name, model, "rerank") as server:
            self.gate_reranker(server, model_name)
            for query, candidate in zip(queries, candidates):
                self.check_time()
                ids = candidate["ids"]
                if len(ids) != 50 or len(set(ids)) != 50 or not set(ids) <= set(by_id):
                    raise RuntimeError("Rerank requires 50 unique known candidate IDs")
                documents = [by_id[doc_id] for doc_id in ids]
                order, elapsed, scores = rerank(server, query["text"], documents)
                final_ids = [ids[index] for index in order]
                row = {
                    "dataset": self.dataset_name, "embedding": embedding_name, "reranker": model_name,
                    "query_id": query["id"], "grade_mode": self.data["grade_mode"], "k": 50,
                    "status": "ok", "embedding_ms": candidate["embedding_ms"],
                    "retrieval_ms": candidate["retrieval_ms"], "rerank_ms": elapsed,
                    "total_ms": candidate["embedding_ms"] + candidate["retrieval_ms"] + elapsed,
                    "candidate_ids_sha256": sha256_json(ids),
                    "latency_mode": "shared_embedding_search_plus_fresh_rerank",
                    "final_ids": final_ids, "scores": [scores[index] for index in order],
                    **query_metadata(query),
                    **metrics(final_ids, ids, self.data["qrels"][query["id"]], self.data["grade_mode"]),
                }
                append(self.out / "query-results.jsonl", row)
                output_rows.append(row)
                if len(output_rows) % 10 == 0:
                    self.progress("reranking", model=model_name, dataset=self.dataset_name,
                                  queries_done=len(output_rows), queries_total=len(queries))
        return output_rows


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        groups[(row["dataset"], row["embedding"], row["reranker"])].append(row)
    summary = []
    for (dataset, embedding, reranker_name), part in sorted(groups.items()):
        item: dict[str, Any] = {
            "dataset": dataset, "embedding": embedding, "reranker": reranker_name,
            "queries": len(part), "errors": sum(row["status"] != "ok" for row in part),
        }
        metrics_in_group = sorted({key for row in part for key in row if key in METRICS or key.startswith("p5_grade")})
        for metric in metrics_in_group:
            item[metric] = statistics.mean(float(row.get(metric, 0.0)) for row in part)
        for key in ("embedding_ms", "retrieval_ms", "rerank_ms", "total_ms"):
            values = [float(row[key]) for row in part if row.get(key) is not None]
            for label, quantile in (("p50", 0.50), ("p95", 0.95)):
                item[f"{key}_{label}"] = float(np.quantile(values, quantile)) if values else None
        summary.append(item)
    return summary


def write_report(out: Path, summary: list[dict[str, Any]], run_kind: str) -> None:
    lines = [
        "# Phase 2 guarded retrieval run", "",
        f"Run type: {run_kind}. Dataset-specific scores remain separate; no cross-dataset pooled ranking.",
        "MLQA uses binary relevance. ACORD retains grades 0–4 and nDCG gain 2^grade−1.",
        "Judged coverage reports the fraction of top-ranked documents with a qrel; unjudged items are not called irrelevant.",
        "Standalone rows retain the embedding model's top 100 IDs. Reranker rows use the fixed top 50 candidate pool.",
        "Latency sums separately measured embedding, exact search, and reranking; it is not production end-to-end latency.", "",
        "| Dataset | Embedder | Reranker | Queries | Errors | Top1 | MRR@10 | nDCG@10 | Recall@50 | Candidate recall@50 | Judged coverage@10/@50 | Rerank p50/p95 ms |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        rerank_ms = "—" if row["rerank_ms_p50"] is None else f"{row['rerank_ms_p50']:.0f}/{row['rerank_ms_p95']:.0f}"
        lines.append(
            f"| {row['dataset']} | {row['embedding']} | {row['reranker']} | {row['queries']} | {row['errors']} | "
            f"{row.get('top1', 0):.3f} | {row.get('mrr10', 0):.3f} | {row.get('ndcg10', 0):.3f} | "
            f"{row.get('recall50', 0):.3f} | {row.get('candidate_recall50', 0):.3f} | "
            f"{row.get('judged_coverage10', 0):.3f}/{row.get('judged_coverage50', 0):.3f} | {rerank_ms} |"
        )
        if row["dataset"] == "acord-eng-eng":
            lines.append(f"| └ graded P@5 | — | — | — | — | — | — | — | — | — | — | grade thresholds: " +
                         ", ".join(f"≥{g}: {row.get(f'p5_grade{g}', 0):.3f}" for g in range(1, 5)) + " |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def paired_cluster_bootstrap(results: list[dict[str, Any]], replicates: int = 2000,
                             seed: int = 20261023) -> list[dict[str, Any]]:
    """Paired query comparisons with cluster resampling for repeated source groups."""
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in results:
        groups[(row["dataset"], row["embedding"], row["reranker"])][row["query_id"]] = row
    comparisons = (
        ("qwen-4b", "standalone", "qwen-8b", "standalone", "A_vs_B"),
        ("qwen-4b", "standalone", "qwen-4b", "qwen-reranker-0.6b", "A_vs_E"),
        ("qwen-4b", "qwen-reranker-0.6b", "qwen-4b", "qwen-reranker-4b", "E_vs_F"),
    )
    rng = np.random.default_rng(seed)
    output = []
    datasets = sorted({row["dataset"] for row in results})
    for dataset in datasets:
        for left_embed, left_rerank, right_embed, right_rerank, comparison in comparisons:
            left = groups.get((dataset, left_embed, left_rerank), {})
            right = groups.get((dataset, right_embed, right_rerank), {})
            ids = sorted(set(left) & set(right))
            if not ids:
                continue
            clusters: dict[str, list[str]] = defaultdict(list)
            for query_id in ids:
                cluster_id = left[query_id].get("cluster_id") or query_id
                clusters[str(cluster_id)].append(query_id)
            cluster_ids = sorted(clusters)
            for metric_name in ("top1", "ndcg10"):
                deltas = {query_id: float(right[query_id][metric_name]) - float(left[query_id][metric_name])
                          for query_id in ids}
                estimates = np.empty(replicates, dtype=np.float64)
                for replicate in range(replicates):
                    sampled = rng.integers(0, len(cluster_ids), size=len(cluster_ids))
                    sample_values = [deltas[qid] for index in sampled for qid in clusters[cluster_ids[int(index)]]]
                    estimates[replicate] = float(np.mean(sample_values))
                output.append({
                    "dataset": dataset, "comparison": comparison,
                    "left": f"{left_embed}+{left_rerank}", "right": f"{right_embed}+{right_rerank}",
                    "metric": metric_name, "paired_delta_right_minus_left": float(np.mean(list(deltas.values()))),
                    "cluster_count": len(cluster_ids), "query_count": len(ids),
                    "bootstrap95": np.quantile(estimates, [0.025, 0.975]).tolist(),
                    "bootstrap_replicates": replicates,
                    "cluster_unit": "query metadata cluster_id; query_id fallback",
                })
    return output

