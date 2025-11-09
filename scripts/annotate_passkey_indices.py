#!/usr/bin/env python3
import argparse
import json
import re
from typing import Any, Dict, List, Tuple
import os
import sys

PASSKEY_REGEX = re.compile(r"\b\d{5}\b")

def find_passkey_chunk_index(sample: Dict[str, Any]) -> int:
    """
    Return the index in sample['ctxs'] whose text contains a 5-digit passkey.
    If multiple chunks contain a 5-digit number, return the first match.
    If none found or ctxs missing, return -1.
    """
    ctxs = sample.get("ctxs")
    if not isinstance(ctxs, list):
        return -1
    for i, ch in enumerate(ctxs):
        # Consider both text and title if present
        title = (ch.get("title") or "")
        text = (ch.get("text") or "")
        blob = f"{title}\n{text}".strip()
        if not blob:
            continue
        if PASSKEY_REGEX.search(blob):
            return i
    return -1

def process_samples(data: Any) -> Any:
    """
    Add 'passkey_chunk_index' field to each sample.
    Supports list of samples or dict with 'results' list.
    """
    if isinstance(data, list):
        for s in data:
            s["passkey_chunk_index"] = find_passkey_chunk_index(s)
        return data
    elif isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        for s in data["results"]:
            s["passkey_chunk_index"] = find_passkey_chunk_index(s)
        return data
    else:
        # Treat as a single sample dict
        if isinstance(data, dict):
            data["passkey_chunk_index"] = find_passkey_chunk_index(data)
        return data

def main():
    parser = argparse.ArgumentParser(description="Annotate passkey chunk index per sample.")
    parser.add_argument("--input", required=True, help="Path to input JSON dataset (e.g., passkey_musique.json)")
    parser.add_argument("--output", required=True, help="Path to write annotated JSON")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    annotated = process_samples(data)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(annotated, f, indent=2)

    print(f"Wrote annotated dataset to {args.output}")

if __name__ == "__main__":
    main()


