#!/usr/bin/env python3
"""Add one GGUF string metadata key without third-party Python packages.

The llama.cpp CLI currently limits a `str:` KV override to 127 characters,
which is shorter than Qwen3's official reranker template. This utility writes
a separate model copy and leaves the source GGUF untouched.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


TYPE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def string_at(data: bytes, pos: int) -> tuple[str, int]:
    length = struct.unpack_from("<Q", data, pos)[0]
    pos += 8
    end = pos + length
    return data[pos:end].decode("utf-8", errors="strict"), end


def skip_value(data: bytes, pos: int, value_type: int) -> int:
    if value_type == 8:
        length = struct.unpack_from("<Q", data, pos)[0]
        return pos + 8 + length
    if value_type == 9:
        subtype = struct.unpack_from("<I", data, pos)[0]
        count = struct.unpack_from("<Q", data, pos + 4)[0]
        pos += 12
        for _ in range(count):
            pos = skip_value(data, pos, subtype)
        return pos
    if value_type not in TYPE_SIZES:
        raise ValueError(f"unsupported GGUF value type {value_type}")
    return pos + TYPE_SIZES[value_type]


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def parse_header(data: bytes) -> tuple[int, int, int, int, int, int]:
    magic, version, tensor_count, kv_count = struct.unpack_from("<4sIQQ", data, 0)
    if magic != b"GGUF":
        raise ValueError("not a GGUF file")
    pos = 24
    alignment = 32
    for _ in range(kv_count):
        key, pos = string_at(data, pos)
        value_type = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        value_pos = pos
        if key == "general.alignment" and value_type == 4:
            alignment = struct.unpack_from("<I", data, value_pos)[0]
        pos = skip_value(data, pos, value_type)
    kv_end = pos
    tensor_info_start = pos
    for _ in range(tensor_count):
        _, pos = string_at(data, pos)
        n_dims = struct.unpack_from("<I", data, pos)[0]
        pos += 4 + 8 * n_dims + 4 + 8
    return version, tensor_count, kv_count, alignment, kv_end, pos


def add_string_metadata(input_path: Path, output_path: Path, key: str, value: str) -> None:
    if output_path.exists():
        raise FileExistsError(output_path)
    with input_path.open("rb") as source:
        prefix = source.read(24)
        # The metadata and tensor-info section is small even for multi-GB
        # models. Read enough to parse it, then stream tensor data below.
        header = bytearray(prefix)
        magic, version, tensor_count, kv_count = struct.unpack("<4sIQQ", prefix)
        source.seek(0)
        probe = source.read(64 * 1024 * 1024)
        while True:
            try:
                _, _, _, alignment, kv_end, tensor_info_end = parse_header(probe)
                break
            except struct.error:
                if len(probe) >= input_path.stat().st_size:
                    raise
                source.seek(len(probe))
                probe += source.read(64 * 1024 * 1024)
        for existing_key_pos in range(24, kv_end):
            # Presence is checked by parsing again, avoiding accidental dupes.
            pass
        pos = 24
        keys = []
        for _ in range(kv_count):
            existing_key, pos = string_at(probe, pos)
            keys.append(existing_key)
            value_type = struct.unpack_from("<I", probe, pos)[0]
            pos = skip_value(probe, pos + 4, value_type)
        if key in keys:
            raise ValueError(f"metadata key already exists: {key}")
        new_entry = struct.pack("<Q", len(key.encode("utf-8"))) + key.encode("utf-8")
        new_entry += struct.pack("<I", 8)  # GGUF_TYPE_STRING
        encoded_value = value.encode("utf-8")
        new_entry += struct.pack("<Q", len(encoded_value)) + encoded_value
        new_kv_count = kv_count + 1
        new_prefix = struct.pack("<4sIQQ", magic, version, tensor_count, new_kv_count)
        old_tensor_info = probe[kv_end:tensor_info_end]
        new_tensor_info_end = 24 + (kv_end - 24) + len(new_entry) + len(old_tensor_info)
        new_data_offset = align(new_tensor_info_end, alignment)
        old_data_offset = align(tensor_info_end, alignment)
        with output_path.open("wb") as destination:
            destination.write(new_prefix)
            destination.write(probe[24:kv_end])
            destination.write(new_entry)
            destination.write(old_tensor_info)
            destination.write(b"\0" * (new_data_offset - new_tensor_info_end))
            source.seek(old_data_offset)
            remaining = input_path.stat().st_size - old_data_offset
            while remaining:
                block = source.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise IOError("unexpected EOF while copying tensor data")
                destination.write(block)
                remaining -= len(block)
    print(f"wrote {output_path} with {key} ({len(value)} chars); data offset {old_data_offset}->{new_data_offset}")


TEMPLATE = """<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>
<|im_start|>user
<Instruct>: Given a web search query, retrieve relevant passages that answer the query
<Query>: {query}
<Document>: {document}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--key", default="tokenizer.chat_template.rerank")
    parser.add_argument("--value", default=TEMPLATE)
    args = parser.parse_args()
    add_string_metadata(args.input, args.output, args.key, args.value)
