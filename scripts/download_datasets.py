#!/usr/bin/env python3
"""Download and preprocess datasets for hallucination detection"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List
import logging

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_ROOT = Path("../data")
DATA_RAW = DATA_ROOT / "raw"
DATA_PROCESSED = DATA_ROOT / "processed"

TRUTHFULQA_SIZE = 500
HOTPOTQA_SIZE = 1000
SQUAD_V2_SIZE = 500

CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

SEED = 42

def standardize_example(example: Dict, source: str) -> Dict:
    """Convert different dataset formats to standard format"""

    if source == "truthfulqa":
        return {
            "id": str(example.get("question_index", "")),
            "question": example["question"],
            "context": example.get("best_answer", ""),
            "answer": example.get("best_answer", ""),
            "source": "truthfulqa"
        }

    elif source == "hotpotqa":
        supporting_facts = example.get("supporting_facts", [])
        context = " ".join([fact for sublist in supporting_facts for fact in sublist])
        return {
            "id": example.get("_id", ""),
            "question": example["question"],
            "context": context,
            "answer": example["answer"],
            "source": "hotpotqa"
        }

    elif source == "squad_v2":
        answers = example.get("answers", {})
        answer_text = answers["text"][0] if answers.get("text") else ""
        return {
            "id": example["id"],
            "question": example["question"],
            "context": example["context"],
            "answer": answer_text,
            "source": "squad_v2"
        }

def chunk_document(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks

def main():
    """Main pipeline"""

    logger.info("="*70)
    logger.info("STREAMING HALLUCINATION DETECTION - DATASET PIPELINE")
    logger.info("="*70)

    # Create directories
    logger.info("\nCreating directories...")
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    (DATA_RAW / "truthfulqa").mkdir(exist_ok=True)
    (DATA_RAW / "hotpotqa").mkdir(exist_ok=True)
    (DATA_RAW / "squad_v2").mkdir(exist_ok=True)
    (DATA_PROCESSED / "annotations").mkdir(exist_ok=True)
    (DATA_PROCESSED / "signals").mkdir(exist_ok=True)
    (DATA_PROCESSED / "splits").mkdir(exist_ok=True)
    logger.info("✓ Directories ready")

    # Download datasets
    logger.info("\nDownloading datasets...")
    all_examples = []

    # TruthfulQA
    try:
        logger.info("  Downloading TruthfulQA...")
        truthfulqa = load_dataset("truthful_qa", "generation")["train"]
        truthfulqa = truthfulqa.shuffle(seed=SEED).select(range(min(TRUTHFULQA_SIZE, len(truthfulqa))))

        for ex in truthfulqa:
            standardized = standardize_example(ex, "truthfulqa")
            all_examples.append(standardized)

        logger.info(f"  ✓ TruthfulQA: {len(truthfulqa)} examples")
    except Exception as e:
        logger.warning(f"  ⚠ TruthfulQA failed: {e}")

    # HotpotQA
    try:
        logger.info("  Downloading HotpotQA...")
        hotpotqa = load_dataset("hotpot_qa", "distractor")["train"]
        hotpotqa = hotpotqa.shuffle(seed=SEED).select(range(min(HOTPOTQA_SIZE, len(hotpotqa))))

        for ex in hotpotqa:
            standardized = standardize_example(ex, "hotpotqa")
            all_examples.append(standardized)

        logger.info(f"  ✓ HotpotQA: {len(hotpotqa)} examples")
    except Exception as e:
        logger.warning(f"  ⚠ HotpotQA failed: {e}")

    # SQuAD v2
    try:
        logger.info("  Downloading SQuAD v2...")
        squad = load_dataset("squad_v2")["train"]
        squad_answerable = squad.filter(lambda x: len(x['answers']['text']) > 0)
        squad_sample = squad_answerable.shuffle(seed=SEED).select(range(min(SQUAD_V2_SIZE, len(squad_answerable))))

        for ex in squad_sample:
            standardized = standardize_example(ex, "squad_v2")
            all_examples.append(standardized)

        logger.info(f"  ✓ SQuAD v2: {len(squad_sample)} examples")
    except Exception as e:
        logger.warning(f"  ⚠ SQuAD v2 failed: {e}")

    if not all_examples:
        logger.error("No datasets could be downloaded!")
        return

    logger.info(f"\nTotal examples: {len(all_examples)}")

    # Preprocess (chunk documents)
    logger.info("\nPreprocessing (chunking documents)...")
    for ex in tqdm(all_examples, desc="Chunking"):
        chunks = chunk_document(ex.get("context", ""), chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        ex["context_chunks"] = chunks
        ex["n_chunks"] = len(chunks)

    logger.info("✓ Preprocessing complete")

    # Compute embeddings
    logger.info("\nComputing embeddings (this may take 10-20 minutes on CPU)...")
    logger.info("  Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    for ex in tqdm(all_examples, desc="Embeddings"):
        chunks = ex.get("context_chunks", [])
        if chunks:
            chunk_embeddings = embedding_model.encode(chunks, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
            ex["chunk_embeddings"] = chunk_embeddings.astype(np.float32)
        else:
            ex["chunk_embeddings"] = np.array([]).reshape(0, 384)

    logger.info("✓ Embeddings complete")

    # Save data
    logger.info("\nSaving processed data...")

    # JSON (readable)
    json_examples = []
    for ex in all_examples:
        json_ex = ex.copy()
        if "chunk_embeddings" in json_ex:
            json_ex["chunk_embeddings"] = json_ex["chunk_embeddings"].tolist()
        json_examples.append(json_ex)

    with open(DATA_PROCESSED / "examples.json", 'w') as f:
        json.dump(json_examples, f, indent=2)
    logger.info(f"  ✓ Saved: {DATA_PROCESSED / 'examples.json'}")

    # Pickle (fast)
    with open(DATA_PROCESSED / "examples.pkl", 'wb') as f:
        pickle.dump(all_examples, f)
    logger.info(f"  ✓ Saved: {DATA_PROCESSED / 'examples.pkl'}")

    # Summary
    summary = {
        "total_examples": len(all_examples),
        "total_chunks": sum(ex.get("n_chunks", 0) for ex in all_examples),
        "sources": list(set(ex["source"] for ex in all_examples)),
        "sources_count": {
            source: len([e for e in all_examples if e["source"] == source])
            for source in set(ex["source"] for ex in all_examples)
        }
    }

    with open(DATA_PROCESSED / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  ✓ Saved: {DATA_PROCESSED / 'summary.json'}")

    # Create splits
    logger.info("\nCreating train/val/test splits...")
    n_examples = len(all_examples)
    n_train = int(n_examples * 0.8)
    n_test = int(n_examples * 0.1)

    indices = np.random.permutation(n_examples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + (n_examples - n_train - n_test)]
    test_indices = indices[n_train + len(val_indices):]

    splits = {
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "split_info": {
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_indices),
            "total": n_examples
        }
    }

    with open(DATA_PROCESSED / "splits" / "standard_split.json", 'w') as f:
        json.dump(splits, f, indent=2)
    logger.info(f"  ✓ Splits: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    logger.info("\n" + "="*70)
    logger.info("✅ DATASET PIPELINE COMPLETE!")
    logger.info("="*70)
    logger.info(f"\nDataset Summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    main()x