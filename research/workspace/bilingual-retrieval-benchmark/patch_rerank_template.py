#!/usr/bin/env python3
"""Make a copy of a GGUF with the Qwen3 reranker template metadata added."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inject_gguf_metadata import TEMPLATE, add_string_metadata  # type: ignore  # noqa: E402


def patch(input_path: Path, output_path: Path) -> None:
    add_string_metadata(input_path, output_path, "tokenizer.chat_template.rerank", TEMPLATE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    patch(args.input, args.output)
