#!/usr/bin/env python3
"""
download_datasets_v2.py
=======================
FIXED version of download_datasets.py.

What changed vs v1 (the version that produced the broken examples.pkl):

1. Bug Fix #1 — HotpotQA ID was always blank
   v1:  example.get("_id", "")    <-- wrong field name
   v2:  example.get("id", "")     <-- correct field name (verified via streaming load)

2. Bug Fix #2 — HotpotQA context was garbage like "t i t l e s e n t _ i d"
   v1:  iterated over `supporting_facts` dict, which produced its KEYS as text
   v2:  reads from `context["sentences"]` (the actual passage text), and uses
        `supporting_facts` only as POINTERS to which sentences are gold-relevant
        (we still join all paragraphs as the full retrievable context for RAG).

3. Defensive hardening
   - Validates every example has a non-empty id and non-empty context after standardization
   - Reports per-source counts of any rejected examples
   - Refuses to write the pickle if validation fails (better to error loudly than ship garbage)

4. Output naming
   - Writes examples.pkl by default so we don't accidentally overwrite the
     broken examples.pkl. Use --in-place to overwrite (creates .bak first).

Usage (from project root):
    python Scripts/download_datasets_v2.py
    python Scripts/download_datasets_v2.py --skip-truthfulqa   # if you don't want TruthfulQA
    python Scripts/download_datasets_v2.py --in-place          # overwrite examples.pkl
    python Scripts/download_datasets_v2.py --hotpotqa-only     # only re-download HotpotQA

Note: this writes examples.pkl in the SAME format as v1 (same field names,
same shape) so downstream scripts (01_generate_responses.py) work without changes.
The new file just has correct values where v1 had garbage.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Match original config
TRUTHFULQA_SIZE = 500
HOTPOTQA_SIZE = 1000
SQUAD_V2_SIZE = 500
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SEED = 42


# =============================================================================
# Per-source standardizers
# =============================================================================

def standardize_hotpotqa(example: Dict) -> Optional[Dict]:
    """
    HotpotQA structure (verified via streaming load on 2026-04-17):
      id: str                   <-- not "_id"
      question: str
      answer: str
      context: {
          "title":     [str, str, ...]      # one per paragraph
          "sentences": [[str, ...], [str, ...], ...]   # parallel, one inner list per paragraph
      }
      supporting_facts: {
          "title":   [str, str, ...]   # which titles contain gold sentences
          "sent_id": [int, int, ...]   # which sentence indices within those paragraphs
      }

    Strategy: build the full context as concatenation of all paragraphs (for
    retrieval to work over). Save supporting_facts info as metadata so we can
    later compute Type A/B retrieval metrics if desired.
    """
    ex_id = str(example.get("id", "")).strip()
    if not ex_id:
        return None  # caller will count and report

    ctx = example.get("context", {})
    titles = ctx.get("title", []) if isinstance(ctx, dict) else []
    sentences = ctx.get("sentences", []) if isinstance(ctx, dict) else []

    if not titles or not sentences or len(titles) != len(sentences):
        return None

    # Build the full context: each paragraph as "Title: sent1 sent2 ..."
    paragraphs = []
    for title, sents in zip(titles, sentences):
        # sents is a list of sentence strings (each may end with whitespace/punct)
        paragraph_text = " ".join(s.strip() for s in sents if isinstance(s, str) and s.strip())
        if paragraph_text:
            paragraphs.append(f"{title}: {paragraph_text}")
    full_context = "\n\n".join(paragraphs)

    if not full_context.strip():
        return None

    # Save supporting_facts as gold-sentence pointers (for later Type A/B analysis)
    sf = example.get("supporting_facts", {})
    sf_titles = sf.get("title", []) if isinstance(sf, dict) else []
    sf_sent_ids = sf.get("sent_id", []) if isinstance(sf, dict) else []
    supporting = list(zip(sf_titles, sf_sent_ids))

    return {
        "id": ex_id,
        "question": example["question"],
        "context": full_context,
        "answer": example["answer"],
        "source": "hotpotqa",
        "supporting_facts": supporting,  # list of (title, sent_idx) tuples; new field
    }


def standardize_squad_v2(example: Dict) -> Optional[Dict]:
    """SQuAD v2 — already worked correctly in v1, kept for completeness."""
    ex_id = str(example.get("id", "")).strip()
    if not ex_id:
        return None
    answers = example.get("answers", {})
    answer_text = answers["text"][0] if answers.get("text") else ""
    context = example.get("context", "")
    if not context.strip():
        return None
    return {
        "id": ex_id,
        "question": example["question"],
        "context": context,
        "answer": answer_text,
        "source": "squad_v2",
    }


def standardize_truthfulqa(example: Dict, fallback_idx: int) -> Optional[Dict]:
    """TruthfulQA — original gave id from question_index, often missing. Use fallback."""
    raw_id = str(example.get("question_index", "")).strip()
    ex_id = raw_id if raw_id else f"truthfulqa_{fallback_idx:04d}"
    best = example.get("best_answer", "")
    if not example.get("question") or not best:
        return None
    return {
        "id": ex_id,
        "question": example["question"],
        "context": best,
        "answer": best,
        "source": "truthfulqa",
    }


# =============================================================================
# Helpers
# =============================================================================

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split text into overlapping word-windows."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def validate(examples: List[Dict]) -> List[str]:
    """Return list of integrity issues. Empty list = OK."""
    issues = []
    if not examples:
        return ["Zero examples produced"]
    blanks = [i for i, e in enumerate(examples) if not str(e.get("id", "")).strip()]
    if blanks:
        issues.append(f"{len(blanks)} examples with blank id")
    seen_ids = {}
    for i, e in enumerate(examples):
        seen_ids.setdefault(e.get("id"), []).append(i)
    dupes = {k: v for k, v in seen_ids.items() if len(v) > 1}
    if dupes:
        issues.append(f"{len(dupes)} duplicate ids (e.g. {list(dupes.items())[:3]})")
    bad_ctx = [
        i for i, e in enumerate(examples)
        if not str(e.get("context", "")).strip()
        or e.get("context") == "t i t l e s e n t _ i d"   # the v1 bug signature
    ]
    if bad_ctx:
        issues.append(f"{len(bad_ctx)} examples with empty/garbage context")
    required = ["id", "question", "context", "answer", "source"]
    for i, e in enumerate(examples):
        missing = [k for k in required if k not in e]
        if missing:
            issues.append(f"Example {i} missing fields: {missing}")
            if len(issues) > 8:
                issues.append("(more issues, truncated)")
                break
    return issues


# =============================================================================
# Main pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", default="data", help="Project data root (default: data)")
    parser.add_argument("--in-place", action="store_true",
                        help="Overwrite examples.pkl/.json. Creates .bak first.")
    parser.add_argument("--skip-truthfulqa", action="store_true",
                        help="Skip TruthfulQA (recommended — it wasn't used in Week 2)")
    parser.add_argument("--hotpotqa-only", action="store_true",
                        help="Only re-process HotpotQA (skip SQuAD + TruthfulQA)")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Skip computing chunk embeddings (faster; embeddings unused in Week 2)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    raw_dir = data_root / "raw"
    proc_dir = data_root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    (proc_dir / "splits").mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("DATASET PIPELINE v2 (HotpotQA bugfix)")
    logger.info("=" * 70)

    all_examples: List[Dict] = []
    rejection_counts: Dict[str, int] = {}

    # ---- HotpotQA ---------------------------------------------------------
    logger.info("\n[HotpotQA] Downloading...")
    try:
        hp = load_dataset("hotpot_qa", "distractor", split="train")
        hp = hp.shuffle(seed=SEED).select(range(min(HOTPOTQA_SIZE, len(hp))))
        rejected = 0
        for ex in tqdm(hp, desc="  HotpotQA standardize"):
            std = standardize_hotpotqa(ex)
            if std is None:
                rejected += 1
            else:
                all_examples.append(std)
        rejection_counts["hotpotqa"] = rejected
        logger.info(f"[HotpotQA] kept {HOTPOTQA_SIZE - rejected}, rejected {rejected}")
    except Exception as e:
        logger.error(f"[HotpotQA] FAILED: {e}")
        if not args.hotpotqa_only:
            logger.warning("  Continuing without HotpotQA")
        else:
            raise

    # ---- SQuAD v2 ----------------------------------------------------------
    if not args.hotpotqa_only:
        logger.info("\n[SQuAD v2] Downloading...")
        try:
            sq = load_dataset("squad_v2", split="train")
            sq_ans = sq.filter(lambda x: len(x["answers"]["text"]) > 0)
            sq_ans = sq_ans.shuffle(seed=SEED).select(range(min(SQUAD_V2_SIZE, len(sq_ans))))
            rejected = 0
            for ex in tqdm(sq_ans, desc="  SQuAD standardize"):
                std = standardize_squad_v2(ex)
                if std is None:
                    rejected += 1
                else:
                    all_examples.append(std)
            rejection_counts["squad_v2"] = rejected
            logger.info(f"[SQuAD v2] kept {SQUAD_V2_SIZE - rejected}, rejected {rejected}")
        except Exception as e:
            logger.error(f"[SQuAD v2] FAILED: {e}")

    # ---- TruthfulQA --------------------------------------------------------
    if not args.hotpotqa_only and not args.skip_truthfulqa:
        logger.info("\n[TruthfulQA] Downloading...")
        try:
            tq = load_dataset("truthful_qa", "generation")["train"]
            tq = tq.shuffle(seed=SEED).select(range(min(TRUTHFULQA_SIZE, len(tq))))
            rejected = 0
            for i, ex in enumerate(tqdm(tq, desc="  TruthfulQA standardize")):
                std = standardize_truthfulqa(ex, i)
                if std is None:
                    rejected += 1
                else:
                    all_examples.append(std)
            rejection_counts["truthfulqa"] = rejected
            logger.info(f"[TruthfulQA] kept {TRUTHFULQA_SIZE - rejected}, rejected {rejected}")
        except Exception as e:
            logger.error(f"[TruthfulQA] FAILED: {e}")

    if not all_examples:
        logger.error("No examples produced. Aborting.")
        return

    logger.info(f"\nTotal standardized examples: {len(all_examples)}")

    # ---- Chunk documents ---------------------------------------------------
    logger.info("\nChunking contexts...")
    for ex in tqdm(all_examples, desc="  Chunking"):
        chunks = chunk_document(ex.get("context", ""), CHUNK_SIZE, CHUNK_OVERLAP)
        ex["context_chunks"] = chunks
        ex["n_chunks"] = len(chunks)

    # ---- Embeddings (optional) --------------------------------------------
    if not args.no_embeddings:
        logger.info("\nComputing chunk embeddings (~10-20 min on CPU)...")
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(EMBEDDING_MODEL)
        for ex in tqdm(all_examples, desc="  Embeddings"):
            chunks = ex.get("context_chunks", [])
            if chunks:
                emb = embed_model.encode(chunks, batch_size=32, convert_to_numpy=True,
                                         show_progress_bar=False)
                ex["chunk_embeddings"] = emb.astype(np.float32)
            else:
                ex["chunk_embeddings"] = np.array([]).reshape(0, 384)
    else:
        logger.info("Skipping embeddings (--no-embeddings)")

    # ---- VALIDATE before writing ------------------------------------------
    logger.info("\nValidating output...")
    issues = validate(all_examples)
    if issues:
        logger.error("VALIDATION FAILED:")
        for issue in issues:
            logger.error(f"  - {issue}")
        logger.error("Refusing to write output. Investigate before retrying.")
        raise SystemExit(2)
    logger.info(f"✓ Validation passed: {len(all_examples)} examples, all fields present, all ids unique")

    # ---- Write -------------------------------------------------------------
    pkl_path = proc_dir / "examples.pkl" if args.in_place else proc_dir / "examples.pkl"
    json_path = proc_dir / "examples.json" if args.in_place else proc_dir / "examples.json"

    if args.in_place:
        for p in (pkl_path, json_path):
            if p.exists():
                bak = p.with_suffix(p.suffix + ".bak")
                shutil.copy2(p, bak)
                logger.info(f"  Backed up: {p.name} -> {bak.name}")

    # Pickle
    with open(pkl_path, "wb") as f:
        pickle.dump(all_examples, f)
    logger.info(f"✓ Wrote {pkl_path} ({pkl_path.stat().st_size / 1e6:.1f} MB)")

    # JSON (embeddings -> list for serialization)
    json_examples = []
    for ex in all_examples:
        je = {k: v for k, v in ex.items() if k != "chunk_embeddings"}
        if "chunk_embeddings" in ex and isinstance(ex["chunk_embeddings"], np.ndarray):
            je["chunk_embeddings"] = ex["chunk_embeddings"].tolist()
        json_examples.append(je)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_examples, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Wrote {json_path} ({json_path.stat().st_size / 1e6:.1f} MB)")

    # Summary
    from collections import Counter
    src_counts = Counter(e["source"] for e in all_examples)
    summary = {
        "total_examples": len(all_examples),
        "per_source": dict(src_counts),
        "rejection_counts": rejection_counts,
        "total_chunks": sum(e.get("n_chunks", 0) for e in all_examples),
        "version": "v2_hotpotqa_fixed",
    }
    summary_path = proc_dir / ("summary.json" if args.in_place else "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Wrote {summary_path}")

    # Show 1 sample per source so user can eyeball
    logger.info("\n=== SANITY CHECK: 1 example per source ===")
    for src in sorted(src_counts.keys()):
        ex = next(e for e in all_examples if e["source"] == src)
        logger.info(f"\n[{src}] id={ex['id']!r}")
        logger.info(f"  Q: {ex['question'][:100]}")
        logger.info(f"  CTX[:200]: {ex['context'][:200]!r}")
        logger.info(f"  Answer: {ex['answer']!r}")

    logger.info("\n" + "=" * 70)
    logger.info("✅ PIPELINE v2 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nNext step: re-generate HotpotQA responses (see chat instructions).")


if __name__ == "__main__":
    main()