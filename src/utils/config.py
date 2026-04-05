"""
Configuration and path management for RAG Hallucination Detection project.
All paths are managed here - update this file to change paths globally.
"""

import os
from pathlib import Path
import json

# ============================================================================
# PROJECT ROOT & MAIN DIRECTORIES
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up 3 levels: utils -> src -> root
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
DOCS_DIR = PROJECT_ROOT / "docs"

print(f"[CONFIG] Project root: {PROJECT_ROOT}")

# ============================================================================
# DATA PATHS (Week 1-4)
# ============================================================================

# Raw datasets
RAW_DATA_PATH = DATA_DIR / "raw"
HOTPOTQA_PATH = RAW_DATA_PATH / "hotpotqa"
SQUADV2_PATH = RAW_DATA_PATH / "squad_v2"
TRUTHFULQA_PATH = RAW_DATA_PATH / "truthfulqa"
CUSTOM_RAG_PATH = RAW_DATA_PATH / "custom_rag"

# Processed data (cleaned, split examples)
PROCESSED_DATA_PATH = DATA_DIR / "processed"
EXAMPLES_JSON = PROCESSED_DATA_PATH / "examples.json"
EXAMPLES_PKL = PROCESSED_DATA_PATH / "examples.pkl"
SUMMARY_JSON = PROCESSED_DATA_PATH / "summary.json"
SPLITS_JSON = PROCESSED_DATA_PATH / "splits" / "standard_split.json"

# Generated data (Week 2+)
GENERATED_DATA_PATH = DATA_DIR / "generated"
RESPONSES_JSON = GENERATED_DATA_PATH / "responses.json"  # Week 2 output
RESPONSES_METADATA = GENERATED_DATA_PATH / "responses_metadata.json"

# Annotations (Week 3-4)
ANNOTATIONS_PATH = DATA_DIR / "annotations"
ANNOTATIONS_JSON = ANNOTATIONS_PATH / "annotations.json"
AGREEMENT_REPORT = ANNOTATIONS_PATH / "agreement_report.json"
PROTOCOL_MD = ANNOTATIONS_PATH / "protocol.md"
CLEANED_DATASET = ANNOTATIONS_PATH / "cleaned_dataset.pkl"

# Cache (embeddings)
CACHE_PATH = DATA_DIR / "cache"
EMBEDDINGS_CACHE = CACHE_PATH / "embeddings.pkl"

# ============================================================================
# MODEL PATHS (Downloaded & Trained)
# ============================================================================

# LLM Models (downloaded Week 2+)
LLM_PATH = MODELS_DIR / "llm"
LLAMA2_7B_PATH = LLM_PATH / "llama-2-7b-chat"
TINYLLAMA_PATH = LLM_PATH / "tinyllama"  # Week 11
LLAMA3_8B_PATH = LLM_PATH / "llama-3-8b"  # Week 11

# Embedding models
EMBEDDINGS_PATH = MODELS_DIR / "embeddings"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_CACHE_FULL_PATH = EMBEDDINGS_PATH / EMBEDDING_MODEL_NAME

# Trained models (models you create)
TRAINED_MODELS_PATH = MODELS_DIR / "trained"
BASELINE_MODEL_W7 = TRAINED_MODELS_PATH / "week7_baseline.pkl"
FINAL_MODEL_W12 = TRAINED_MODELS_PATH / "week12_final.pkl"

# ============================================================================
# SIGNALS & EXTRACTED FEATURES (Week 5)
# ============================================================================

SIGNALS_PATH = PROCESSED_DATA_PATH / "signals"
SIGNALS_PKL = SIGNALS_PATH / "signals.pkl"
SIGNAL_STATS = SIGNALS_PATH / "signal_statistics.json"

# ============================================================================
# RESULTS PATHS (Outputs from analysis)
# ============================================================================

# Evaluation results
EVAL_RESULTS_PATH = RESULTS_DIR / "evaluation"
CV_RESULTS = EVAL_RESULTS_PATH / "cv_results.json"  # Week 7
TEST_RESULTS = EVAL_RESULTS_PATH / "test_results.json"  # Week 8
TYPE_AB_RESULTS = EVAL_RESULTS_PATH / "type_ab_breakdown.json"  # Week 9
DOMAIN_RESULTS = EVAL_RESULTS_PATH / "domain_results.json"  # Week 10
MODEL_RESULTS = EVAL_RESULTS_PATH / "model_results.json"  # Week 11
ERROR_ANALYSIS = EVAL_RESULTS_PATH / "error_analysis.json"  # Week 12

# Figures
FIGURES_PATH = RESULTS_DIR / "figures"
ROC_CURVE = FIGURES_PATH / "roc_curve.png"
SIGNAL_DISTRIBUTIONS = FIGURES_PATH / "signal_distributions.png"
TYPE_AB_COMPARISON = FIGURES_PATH / "type_ab_comparison.png"

# Intervention results
INTERVENTIONS_PATH = RESULTS_DIR / "interventions"
RERETRIEVAL_RESULTS = INTERVENTIONS_PATH / "reretrieval_results.json"
SUPPRESSION_RESULTS = INTERVENTIONS_PATH / "suppression_results.json"

# Logs
LOGS_PATH = RESULTS_DIR / "logs"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data: dict, path: Path):
    """Save JSON file."""
    ensure_dir(path.parent)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_pkl(path: Path):
    """Load pickle file."""
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pkl(data, path: Path):
    """Save pickle file."""
    import pickle
    ensure_dir(path.parent)
    with open(path, 'wb') as f:
        pickle.dump(data, f)

# ============================================================================
# INITIALIZATION: Create all directories
# ============================================================================

def init_directories():
    """Create all necessary directories at startup."""
    dirs = [
        DATA_DIR, RAW_DATA_PATH, PROCESSED_DATA_PATH, GENERATED_DATA_PATH, ANNOTATIONS_PATH, CACHE_PATH,
        MODELS_DIR, LLM_PATH, EMBEDDINGS_PATH, TRAINED_MODELS_PATH,
        RESULTS_DIR, EVAL_RESULTS_PATH, FIGURES_PATH, INTERVENTIONS_PATH, LOGS_PATH,
        SIGNALS_PATH,
    ]
    for d in dirs:
        ensure_dir(d)
    print("[CONFIG] All directories initialized ✓")

# Auto-initialize when imported
init_directories()

# ============================================================================
# VERIFICATION: Print summary
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PROJECT PATHS VERIFICATION")
    print("="*80)
    print(f"\n📁 PROJECT ROOT: {PROJECT_ROOT}")
    print(f"\n📊 DATA PATHS:")
    print(f"  Raw data:        {RAW_DATA_PATH}")
    print(f"  Processed data:  {PROCESSED_DATA_PATH}")
    print(f"  Generated data:  {GENERATED_DATA_PATH} (Week 2)")
    print(f"  Annotations:     {ANNOTATIONS_PATH} (Week 3-4)")
    print(f"\n🤖 MODEL PATHS:")
    print(f"  LLaMA-2-7B:      {LLAMA2_7B_PATH} (Week 2)")
    print(f"  Embeddings:      {EMBEDDING_CACHE_FULL_PATH}")
    print(f"  Trained models:  {TRAINED_MODELS_PATH}")
    print(f"\n📈 RESULTS PATHS:")
    print(f"  Evaluation:      {EVAL_RESULTS_PATH}")
    print(f"  Figures:         {FIGURES_PATH}")
    print(f"  Interventions:   {INTERVENTIONS_PATH}")
    print(f"\n✅ All paths initialized successfully!")
    print("="*80 + "\n")