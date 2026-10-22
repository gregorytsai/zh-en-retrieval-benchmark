#!/usr/bin/env python3
"""Run the repaired Qwen reranker matrix with explicit Metal offload.

This writes to a new output tree so the earlier interrupted/default-device
run remains intact for audit.  The v0.4.1 server is forced to offload all
layers; the resulting report is annotated with the verified MTL0 backend.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


BENCHMARK = (Path(__file__).resolve().parents[1] / 'bilingual-retrieval-benchmark')
OUTPUT = (Path(__file__).resolve().parents[1] / 'qwen-reranker-rebuilt/benchmark-metal')
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

_original_command = benchmark.LlamaServer._command


def _metal_command(self: benchmark.LlamaServer, port: int) -> list[str]:
    command = _original_command(self, port)
    return command + ["--gpu-layers", "99"]


benchmark.LlamaServer._command = _metal_command


def main() -> int:
    result = benchmark.run(benchmark.parse_args())
    metadata_path = OUTPUT / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["hardware_note"] = (
            "Apple M3 Ultra Metal backend (MTL0), verified with v0.4.1 --gpu-layers 99; "
            "all 29/29 Qwen 0.6B layers offloaded in the diagnostic."
        )
        metadata["gpu_layers"] = 99
        metadata["device"] = "MTL0"
        metadata["offload_diagnostic"] = {
            "results": str(REBUILT / "offload-diagnostics/results.json"),
            "cpu_log": str(REBUILT / "offload-diagnostics/reranker-gpu-layers-0.log"),
            "metal_log": str(REBUILT / "offload-diagnostics/reranker-gpu-layers-99.log"),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = OUTPUT / "report.md"
    if report_path.exists():
        report = report_path.read_text(encoding="utf-8")
        report = report.replace(
            "Hardware is the host reported by `platform.platform()`/`platform.machine()` and the CPU-only llama.cpp build; no GPU device was available to this run.",
            "Hardware is the host reported by `platform.platform()`/`platform.machine()`. This run explicitly used `--gpu-layers 99`; v0.4.1 logs verified the Apple M3 Ultra Metal device (`MTL0`) and 29/29 layer offload in the companion diagnostic.",
        )
        report_path.write_text(report, encoding="utf-8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
