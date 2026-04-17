"""
Week 2: Generate LLM responses for all 1500 examples
Outputs: data/generated/responses.json
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_DIR / "logs" / "week2_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'batch_size': 1,
    'max_response_length': 150,  # Reduced for CPU
    'temperature': 0.7,
    'top_p': 0.9,
    'context_chunks': 3,  # Top-3 chunks (was 5)
    'device': 'cpu'  # CPU only
}


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_examples():
    """Load processed examples from Week 1"""
    logger.info("Loading examples...")
    try:
        examples = load_pkl(EXAMPLES_PKL)
        logger.info(f"✅ Loaded {len(examples)} examples")
        return examples
    except Exception as e:
        logger.error(f"❌ Error loading examples: {e}")
        raise


# ============================================================================
# STEP 2: BUILD RETRIEVAL SYSTEM
# ============================================================================

def build_retrieval_system(examples: List[Dict]):
    """Build FAISS index for retrieval"""
    logger.info("Building retrieval system...")

    try:
        # Load embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Embedding model loaded")

        # Get contexts
        contexts = [ex.get('context', '') for ex in examples]

        # Encode contexts
        logger.info(f"Encoding {len(contexts)} contexts...")
        embeddings = embedding_model.encode(
            contexts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Build FAISS index
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        faiss_index.add(embeddings)

        logger.info(f"✅ FAISS index built with {len(embeddings)} documents")

        return {
            'embedding_model': embedding_model,
            'faiss_index': faiss_index,
            'contexts': contexts,
            'examples': examples
        }
    except Exception as e:
        logger.error(f"❌ Error building retrieval: {e}")
        raise


# ============================================================================
# STEP 3: LOAD LLM
# ============================================================================

def load_llm():
    """Load LLaMA-2-7B model"""
    logger.info(f"Loading LLaMA-2-7B from {LLAMA2_7B_PATH}...")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(LLAMA2_7B_PATH))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("✅ Tokenizer loaded")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            str(LLAMA2_7B_PATH),
            torch_dtype=torch.float32,
            output_attentions=False,
            output_hidden_states=False
        )

        logger.info(f"✅ Model loaded on CPU")
        return tokenizer, model
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise


# ============================================================================
# STEP 4: RETRIEVE CONTEXT
# ============================================================================

def retrieve_context(retrieval_system: Dict, query: str, k: int = 3) -> List[str]:
    """Retrieve top-k context chunks"""
    try:
        embedding_model = retrieval_system['embedding_model']
        faiss_index = retrieval_system['faiss_index']
        contexts = retrieval_system['contexts']

        # Embed query
        query_embedding = embedding_model.encode(query, convert_to_numpy=True)
        query_embedding = np.array([query_embedding], dtype=np.float32)

        # Search
        distances, indices = faiss_index.search(query_embedding, k=k)

        # Get contexts
        retrieved = [contexts[idx] for idx in indices[0] if idx < len(contexts)]
        return retrieved
    except Exception as e:
        logger.warning(f"Error retrieving context: {e}")
        return [""]


# ============================================================================
# STEP 5: GENERATE RESPONSE
# ============================================================================

def generate_response(tokenizer, model, question: str, context_chunks: List[str]) -> Dict:
    """Generate response"""
    try:
        # Format prompt
        context_text = "\n".join([c[:300] for c in context_chunks if c])
        prompt = f"""Context:
{context_text}

Question: {question}

Answer:"""

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Generate
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=CONFIG['max_response_length'],
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generation_time = (time.time() - start_time) * 1000

        # Decode
        response_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        return {
            'response': response_text,
            'generation_time_ms': generation_time,
            'token_count': len(response_ids)
        }
    except Exception as e:
        logger.error(f"Error generating: {e}")
        return {
            'response': '[ERROR]',
            'generation_time_ms': 0,
            'token_count': 0
        }


# ============================================================================
# STEP 6: MAIN LOOP
# ============================================================================

def generate_all_responses(examples: List[Dict], tokenizer, model, retrieval_system: Dict) -> List[Dict]:
    """Generate responses for all examples"""

    logger.info(f"Starting generation for {len(examples)} examples...")
    logger.info(f"Device: {CONFIG['device']}")

    results = []
    errors = []

    pbar = tqdm(enumerate(examples), total=len(examples), desc="Generating")

    for idx, example in pbar:
        try:
            question = example.get('question', '')
            retrieved = retrieve_context(retrieval_system, question, k=CONFIG['context_chunks'])

            gen_result = generate_response(tokenizer, model, question, retrieved)

            result = {
                'id': example.get('id', f'example_{idx}'),
                'question': question,
                'context_retrieved': retrieved,
                'model_response': gen_result['response'],
                'ground_truth_answer': example.get('answer', ''),
                'source_dataset': example.get('source', 'unknown'),
                'generation_time_ms': gen_result['generation_time_ms'],
                'token_count': gen_result['token_count']
            }

            results.append(result)

            if (idx + 1) % 50 == 0:
                avg_time = np.mean([r['generation_time_ms'] for r in results[-50:]])
                logger.info(f"Processed {idx + 1}/{len(examples)} (avg time: {avg_time:.0f}ms)")

        except Exception as e:
            logger.error(f"Error at {idx}: {e}")
            errors.append({'index': idx, 'error': str(e)})
            results.append({
                'id': example.get('id', f'example_{idx}'),
                'question': example.get('question', ''),
                'model_response': '[ERROR]'
            })

    logger.info(f"✅ Generation complete: {len(results)} responses, {len(errors)} errors")
    return results, errors


# ============================================================================
# STEP 7: SAVE RESULTS
# ============================================================================

def save_results(results: List[Dict], errors: List[Dict]):
    """Save responses"""
    logger.info("Saving results...")

    try:
        GENERATED_DATA_PATH.mkdir(parents=True, exist_ok=True)

        # Save responses
        with open(RESPONSES_JSON, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"✅ Saved {len(results)} responses to {RESPONSES_JSON}")

        # Save metadata
        metadata = {
            'total_examples': len(results),
            'successful': len([r for r in results if r.get('model_response') != '[ERROR]']),
            'failed': len(errors),
            'avg_time_ms': np.mean([r['generation_time_ms'] for r in results if r.get('generation_time_ms')]),
            'device': CONFIG['device']
        }

        with open(GENERATED_DATA_PATH / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✅ Saved metadata")
    except Exception as e:
        logger.error(f"❌ Error saving: {e}")
        raise


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution"""
    logger.info("=" * 80)
    logger.info("WEEK 2: RESPONSE GENERATION")
    logger.info("=" * 80)

    try:
        # Load data
        examples = load_examples()

        # Build retrieval
        retrieval_system = build_retrieval_system(examples)

        # Load model
        tokenizer, model = load_llm()

        # Generate
        results, errors = generate_all_responses(examples, tokenizer, model, retrieval_system)

        # Save
        save_results(results, errors)

        logger.info("=" * 80)
        logger.info("✅ WEEK 2 COMPLETE")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"❌ FAILED: {e}")
        raise


if __name__ == "__main__":
    main()