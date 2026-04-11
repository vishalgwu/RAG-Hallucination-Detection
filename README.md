# RAG Hallucination Detection: Streaming Detection in Retrieval-Augmented Generation

## Project Structure

### Data Directory (`data/`)
- `raw/`: Original datasets (HotpotQA, SQuAD v2, domain-specific)
- `processed/`: Cleaned, split examples (1500 total)
- `generated/`: LLM-generated responses (Week 2+)
- `annotations/`: Token-level labels (Week 3-4)
- `cache/`: Embeddings cache

### Models Directory (`models/`)
- `llm/`: Downloaded LLM weights (LLaMA-2-7B, etc.)
- `embeddings/`: Embedding model cache
- `trained/`: Trained models from our pipeline

### Code (`src/`)
- `data/`: Data loading and preprocessing
- `signals/`: 7 signal extraction functions
- `models/`: Hallucination score function + training
- `training/`: Training loops
- `evaluation/`: Metrics and validation
- `interventions/`: Real-time intervention strategies
- `utils/`: Configuration and helpers

### Results (`results/`)
- `evaluation/`: Metrics and analysis results
- `figures/`: Plots and visualizations
- `logs/`: Execution logs
- `interventions/`: Intervention experiment results

### Scripts (`scripts/`)
Executable runners for each week of the project.

## Setup
```bash
python -m venv rag
source rag/bin/activate  # or: rag\Scripts\activate (Windows)
pip install -r requirements.txt
```

## Weeks Timeline
- **Week 1**: Setup 
- **Week 2**: Response generation
- **Weeks 3-4**: Annotation
- **Weeks 5-7**: Signal extraction + model
- **Weeks 8-12**: Validation
- **Weeks 13-20**: Paper
- **Weeks 21-24**: Buffer + submission
