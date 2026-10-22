#!/usr/bin/env python3
"""Run the existing synthetic benchmark into an isolated repaired output tree."""

from __future__ import annotations

import sys
from pathlib import Path


BENCHMARK = (Path(__file__).resolve().parents[1] / 'bilingual-retrieval-benchmark')
OUTPUT = (Path(__file__).resolve().parents[1] / 'qwen-reranker-rebuilt/benchmark')
sys.path.insert(0, str(BENCHMARK))
import run_benchmark as benchmark  # noqa: E402


WORKSPACE = (Path(__file__).resolve().parents[1] / '')
REBUILT = (Path(__file__).resolve().parents[1] / 'qwen-reranker-rebuilt')
benchmark.ROOT = OUTPUT
benchmark.EMBED_MODELS = {
    "qwen3-embedding-8b-q8_0": WORKSPACE / "Qwen3-Embedding-8B-Q8_0.gguf",
    "qwen3-embedding-4b-q8_0": WORKSPACE / "Qwen3-Embedding-4B.Q8_0.gguf",
}
benchmark.RERANK_MODELS = {
    "qwen3-reranker-0.6b-rebuilt-q8_0": REBUILT / "Qwen3-Reranker-0.6B-Q8_0.gguf",
    "qwen3-reranker-4b-rebuilt-q5_k_m": REBUILT / "Qwen3-Reranker-4B-Q5_K_M.gguf",
    "qwen3-reranker-4b-rebuilt-q8_0": REBUILT / "Qwen3-Reranker-4B-Q8_0.gguf",
}


if __name__ == "__main__":
    raise SystemExit(benchmark.run(benchmark.parse_args()))
