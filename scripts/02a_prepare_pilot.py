"""
02a_prepare_pilot.py
====================
Pick N random examples (stratified by source_dataset) from responses.json
for human annotation.

Uses seed=42 for reproducibility: both annotators running the script
(or one person running and sharing) get the exact same 50 examples in
the exact same order.

Usage (from project root):
    # default: 50 examples from responses.json, seed=42
    python scripts/02a_prepare_pilot.py

    # custom sizes
    python scripts/02a_prepare_pilot.py --n 50
    python scripts/02a_prepare_pilot.py --n 200 --seed 123
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from collections import Counter


def stratified_sample(responses, n_total, seed):
    """Stratified random sample preserving source_dataset proportions."""
    rng = random.Random(seed)
    by_source = {}
    for r in responses:
        by_source.setdefault(r["source_dataset"], []).append(r)

    total = len(responses)
    sampled = []
    for source, items in by_source.items():
        quota = round(n_total * len(items) / total)
        quota = min(quota, len(items))
        picked = rng.sample(items, quota)
        sampled.extend(picked)
        print(f"  [{source}] pool={len(items):4d}  sampled={quota}")

    # Correct rounding errors
    while len(sampled) > n_total:
        sampled.pop()
    while len(sampled) < n_total:
        remaining = [r for r in responses if r not in sampled]
        if not remaining:
            break
        sampled.append(rng.choice(remaining))

    # Shuffle final order so annotators don't see all of one source first
    rng.shuffle(sampled)
    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/generated/responses.json")
    parser.add_argument("--output-dir", default="data/annotations")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"ERROR: input not found: {inp}", file=sys.stderr)
        sys.exit(1)

    with open(inp, "r", encoding="utf-8") as f:
        responses = json.load(f)
    print(f"[prepare_pilot] Loaded {len(responses)} responses")

    if args.n > len(responses):
        print(f"ERROR: requested {args.n} but only {len(responses)} available", file=sys.stderr)
        sys.exit(1)

    source_counts = Counter(r["source_dataset"] for r in responses)
    print(f"  Source distribution: {dict(source_counts)}")

    print(f"\n[prepare_pilot] Sampling {args.n} (seed={args.seed}):")
    pilot = stratified_sample(responses, args.n, args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pilot_path = out_dir / f"pilot_{args.n}.json"
    ids_path = out_dir / f"pilot_{args.n}_ids.txt"

    with open(pilot_path, "w", encoding="utf-8") as f:
        json.dump(pilot, f, ensure_ascii=False, indent=2)
    with open(ids_path, "w", encoding="utf-8") as f:
        f.write("\n".join(r["id"] for r in pilot) + "\n")

    pilot_sources = Counter(r["source_dataset"] for r in pilot)
    print(f"\n[prepare_pilot] Done.")
    print(f"  Pilot size:    {len(pilot)}")
    print(f"  Source mix:    {dict(pilot_sources)}")
    print(f"  Saved:         {pilot_path}")
    print(f"                 {ids_path}")
    print(f"\nNext:  python scripts/02_tokenize_responses.py "
          f"--input {pilot_path} --output {out_dir / (pilot_path.stem + '_tokenized.json')}")


if __name__ == "__main__":
    main()
