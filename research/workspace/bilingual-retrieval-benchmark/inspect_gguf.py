#!/usr/bin/env python3
"""Inspect reranker metadata/tensor signals without numpy."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from inject_gguf_metadata import parse_header, skip_value, string_at


ROOT = Path(__file__).resolve().parent
MODELS = {
    "qwen3-reranker-0.6b-q8_0": ROOT.parent / "Qwen3-Reranker-0.6B.Q8_0.gguf",
    "qwen3-reranker-4b-q5_k_m": ROOT.parent / "Qwen3-Reranker-4B.Q5_K_M.gguf",
    "qwen3-reranker-4b-q8_0": ROOT.parent / "Qwen3-Reranker-4B.Q8_0.gguf",
    "bge-reranker-v2-m3-q8_0": ROOT.parent / "bge-reranker-v2-m3-q8_0.gguf",
}


def inspect(path: Path) -> dict:
    with path.open("rb") as handle:
        data = handle.read(64 * 1024 * 1024)
    _, tensor_count, kv_count, alignment, kv_end, tensor_info_end = parse_header(data)
    pos = 24
    metadata = {}
    for _ in range(kv_count):
        key, pos = string_at(data, pos)
        value_type = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        value_pos = pos
        pos = skip_value(data, pos, value_type)
        if value_type == 8:
            metadata[key] = string_at(data, value_pos)[0]
        elif value_type == 9:
            metadata[key] = "array"
        elif value_type == 4:
            metadata[key] = struct.unpack_from("<I", data, value_pos)[0]
    pos = kv_end
    tensors = []
    for _ in range(tensor_count):
        name, pos = string_at(data, pos)
        n_dims = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        dims = struct.unpack_from("<" + "Q" * n_dims, data, pos)
        pos += 8 * n_dims + 4 + 8
        tensors.append({"name": name, "shape": list(dims)})
    cls_tensors = [tensor for tensor in tensors if "cls" in tensor["name"]]
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "general.name": metadata.get("general.name"),
        "general.architecture": metadata.get("general.architecture"),
        "general.source.url": metadata.get("general.source.url"),
        "tokenizer.chat_template.rerank": "tokenizer.chat_template.rerank" in metadata,
        "classifier_output_labels_key": any(key.endswith("classifier.output_labels") for key in metadata),
        "cls_tensors": cls_tensors,
        "token_count": len(metadata.get("tokenizer.ggml.tokens", [])) if isinstance(metadata.get("tokenizer.ggml.tokens"), list) else None,
        "tensor_count": tensor_count,
        "alignment": alignment,
        "data_offset": ((tensor_info_end + alignment - 1) // alignment) * alignment,
    }


if __name__ == "__main__":
    out = {name: inspect(path) for name, path in MODELS.items()}
    (ROOT / "model_compatibility.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
