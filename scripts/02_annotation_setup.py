"""
02_tokenize_responses.py
========================
Tokenize model responses into word-level tokens for human annotation.

INPUT:  data/generated/responses.json  (1500 responses from Week 2)
OUTPUT: data/generated/responses_tokenized.json  (same + 'tokens' field per entry)

Tokenization scheme:
    Word-level + punctuation-as-separate-token (via regex).
    Chosen over LLaMA BPE because BPE subwords (like 'Ġthe', '▁Paris') are
    hard for humans to annotate and hurt inter-rater agreement.
    In Week 5 (signal extraction) we'll map word labels -> BPE tokens:
    each BPE token inherits the label of the word it is a subword of.

Each response gains:
    "tokens": [
        {"idx": 0, "text": "Knox",     "start": 0,  "end": 4},
        {"idx": 1, "text": "County",   "start": 5,  "end": 11},
        {"idx": 2, "text": "Regional", "start": 12, "end": 20},
        ...
    ]

Usage (from project root):
    python scripts/02_tokenize_responses.py
    python scripts/02_tokenize_responses.py --input custom.json --output custom_out.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Regex: a "token" is either a run of word chars (letters/digits/_/'/-)
# or a single non-whitespace, non-word character (punctuation).
# Apostrophes and internal hyphens stay attached to the word ("don't", "Polish-born").
TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_'\-]*|[^\w\s]", re.UNICODE)


def tokenize_text(text: str) -> list[dict]:
    """Split text into annotation tokens preserving char offsets."""
    tokens = []
    for idx, m in enumerate(TOKEN_RE.finditer(text)):
        tokens.append({
            "idx": idx,
            "text": m.group(),
            "start": m.start(),
            "end": m.end(),
        })
    return tokens


def tokenize_responses(input_path: Path, output_path: Path) -> dict:
    """Load responses.json, add 'tokens' field to each entry, write out."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        responses = json.load(f)

    if not isinstance(responses, list):
        raise ValueError(f"Expected JSON array at top level, got {type(responses).__name__}")

    total_tokens = 0
    token_counts = []
    for entry in responses:
        if "model_response" not in entry:
            raise KeyError(f"Entry {entry.get('id', '?')} missing 'model_response'")
        toks = tokenize_text(entry["model_response"])
        entry["tokens"] = toks
        entry["annotation_token_count"] = len(toks)
        token_counts.append(len(toks))
        total_tokens += len(toks)

    # Ensure output dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)

    stats = {
        "num_responses": len(responses),
        "total_tokens": total_tokens,
        "mean_tokens_per_response": round(sum(token_counts) / len(token_counts), 2) if token_counts else 0,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "output_file": str(output_path),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--input",
        default="data/generated/generated_50_example.json",
        help="Path to responses.json (default: data/generated/generated_50_examples.json)",
    )
    parser.add_argument(
        "--output",
        default="data/generated/responses_50_tokenized.json",
        help="Path for tokenized output (default: data/generated/responses_50_tokenized.json)",
    )
    args = parser.parse_args()

    print(f"[02_tokenize] Loading {args.input} ...")
    try:
        stats = tokenize_responses(Path(args.input), Path(args.output))
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"[02_tokenize] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[02_tokenize] Done.")
    print(f"  Responses:               {stats['num_responses']}")
    print(f"  Total annotation tokens: {stats['total_tokens']:,}")
    print(f"  Mean tokens/response:    {stats['mean_tokens_per_response']}")
    print(f"  Min / Max tokens:        {stats['min_tokens']} / {stats['max_tokens']}")
    print(f"  Saved to:                {stats['output_file']}")


if __name__ == "__main__":
    main()