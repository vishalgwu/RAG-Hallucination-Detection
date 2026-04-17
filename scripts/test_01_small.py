"""Test generation on 10 examples only"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
from tqdm import tqdm
import json

# Load model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(str(LLAMA2_7B_PATH))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(str(LLAMA2_7B_PATH), torch_dtype=torch.float32)

# Load examples (ONLY 10)
print("Loading examples...")
examples = load_pkl(EXAMPLES_PKL)[:10]
print(f"✅ Loaded {len(examples)} examples for testing")

# Build embedding index
print("Building retrieval system...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
contexts = [ex.get('context', '') for ex in examples]
embeddings = embedding_model.encode(contexts, batch_size=32, convert_to_numpy=True)
embeddings = np.array(embeddings, dtype=np.float32)
faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(embeddings)

# Generate on 10 examples
print("\nGenerating responses on 10 examples...")
results = []

for idx, example in enumerate(examples):
    try:
        question = example.get('question', '')

        # Retrieve
        query_emb = embedding_model.encode(question, convert_to_numpy=True)
        distances, indices = faiss_index.search(np.array([query_emb], dtype=np.float32), k=3)
        retrieved = [contexts[i] for i in indices[0]]

        # Generate
        context_text = "\n".join([c[:300] for c in retrieved if c])
        prompt = f"""Context:
{context_text}

Question: {question}

Answer:"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        result = {
            'id': example.get('id', f'test_{idx}'),
            'question': question,
            'model_response': response,
            'ground_truth': example.get('answer', '')
        }
        results.append(result)
        print(f"[{idx + 1}/10] ✅ Generated")

    except Exception as e:
        print(f"[{idx + 1}/10] ❌ Error: {e}")
        results.append({'id': f'test_{idx}', 'error': str(e)})

# Save test results
with open(GENERATED_DATA_PATH / "test_results_10.json", 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print(f"✅ TEST COMPLETE: {len(results)} examples processed")
print(f"Saved to: data/generated/test_results_10.json")
print("=" * 80)