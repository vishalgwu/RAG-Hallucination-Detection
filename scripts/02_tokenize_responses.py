"""
02_tokenize_responses.py
========================
Tokenize model responses into word-level tokens for human annotation.

Each entry in the input JSON gains:
    "tokens": [
        {"idx": 0, "text": "Knox",   "start": 0, "end": 4},
        {"idx": 1, "text": "County", "start": 5, "end": 11},
        ...
    ]

Tokenization: word-level + punctuation-as-separate-token.
Chosen over LLaMA BPE because BPE subwords are hard for humans to annotate.
In Week 5 we'll map word labels -> BPE tokens: each BPE token inherits the
label of the word it is a subword of.

Usage (from project root):
    python scripts/02_tokenize_responses.py \\
        --input data/annotations/pilot_50.json \\
        --output data/annotations/pilot_50_tokenized.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_'\-]*|[^\w\s]", re.UNICODE)


def tokenize_text(text: str) -> list[dict]:
    return [
        {"idx": i, "text": m.group(), "start": m.start(), "end": m.end()}
        for i, m in enumerate(TOKEN_RE.finditer(text))
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/annotations/pilot_50.json")
    parser.add_argument("--output", default="data/annotations/pilot_50_tokenized.json")
    args = parser.parse_args()

    inp, out = Path(args.input), Path(args.output)
    if not inp.exists():
        print(f"ERROR: input not found: {inp}", file=sys.stderr)
        sys.exit(1)

    print(f"[tokenize] Loading {inp} ...")
    with open(inp, "r", encoding="utf-8") as f:
        responses = json.load(f)

    if not isinstance(responses, list):
        print(f"ERROR: expected JSON array, got {type(responses).__name__}", file=sys.stderr)
        sys.exit(1)

    counts = []
    for entry in responses:
        if "model_response" not in entry:
            print(f"ERROR: entry {entry.get('id','?')} missing 'model_response'", file=sys.stderr)
            sys.exit(1)
        toks = tokenize_text(entry["model_response"])
        entry["tokens"] = toks
        entry["annotation_token_count"] = len(toks)
        counts.append(len(toks))

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)

    print(f"[tokenize] Done.")
    print(f"  Responses:      {len(responses)}")
    print(f"  Total tokens:   {sum(counts):,}")
    print(f"  Mean/Min/Max:   {sum(counts)//len(counts) if counts else 0} / "
          f"{min(counts) if counts else 0} / {max(counts) if counts else 0}")
    print(f"  Saved:          {out}")


if __name__ == "__main__":
    main()
