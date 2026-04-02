#!/usr/bin/env python3
"""
Phase 1: Create three conditions (no-RAG, good-RAG, bad-RAG)
For each question, we need three versions of input for the model
"""

import json
import os
import random
from typing import Dict, List, Optional

def load_triviaqa_data(dev_file: str, num_samples: int = 50) -> List[Dict]:
    """Load TriviaQA data and return first N samples"""
    with open(dev_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data['Data'][:num_samples]
    print(f"Loaded {len(samples)} samples from TriviaQA")
    return samples


def create_three_conditions(sample: Dict) -> Dict:
    """
    Given one TriviaQA sample, create three conditions:
    A) Query alone (no RAG)
    B) Query + good RAG (top-ranked search result)
    C) Query + bad RAG (mid-ranked search result)
    
    Args:
        sample: One TriviaQA sample (question + answer + search results)
    
    Returns:
        dict with three conditions
    """
    
    question = sample['Question']
    ground_truth = sample['Answer']['Value']
    search_results = sample['SearchResults']
    question_id = sample['QuestionId']
    
    # Condition A: Question alone
    condition_a = {
        'input': question,
        'type': 'no_rag',
        'document': None
    }
    
    # Condition B: Good RAG (use rank 0, most relevant)
    condition_b = None
    if search_results and len(search_results) > 0:
        good_doc = search_results[0]['Description']
        condition_b = {
            'input': f"{question} {good_doc}",
            'type': 'good_rag',
            'document': good_doc,
            'rank': 0
        }
    
    # Condition C: Bad RAG (use rank 3-5, less relevant)
    condition_c = None
    if search_results and len(search_results) > 3:
        # Use rank 3 or 4 (less relevant than rank 0, but still from search results)
        bad_rank = min(3 + random.randint(0, 1), len(search_results) - 1)
        bad_doc = search_results[bad_rank]['Description']
        condition_c = {
            'input': f"{question} {bad_doc}",
            'type': 'bad_rag',
            'document': bad_doc,
            'rank': bad_rank
        }
    
    return {
        'question_id': question_id,
        'question': question,
        'ground_truth': ground_truth,
        'num_search_results': len(search_results),
        'condition_a': condition_a,
        'condition_b': condition_b,
        'condition_c': condition_c
    }


def main():
    # Load data
    dev_file = '../Data/unfiltered-web-dev.json'
    samples = load_triviaqa_data(dev_file, num_samples=50)
    
    # Create conditions for all samples
    conditions_list = []
    
    print("\nCreating three conditions for each sample...")
    for i, sample in enumerate(samples):
        conditions = create_three_conditions(sample)
        conditions_list.append(conditions)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples")
    
    # Save conditions to file (for next step)
    output_file = '../outputs/phase1_conditions.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conditions_list, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(conditions_list)} conditions to {output_file}")
    
    # Print sample inspection
    print("\n" + "=" * 70)
    print("SAMPLE INSPECTION")
    print("=" * 70)
    
    sample_idx = 0
    sample = conditions_list[sample_idx]
    
    print(f"\nQuestion ID: {sample['question_id']}")
    print(f"Question: {sample['question']}")
    print(f"Ground Truth: {sample['ground_truth']}")
    print(f"Number of search results: {sample['num_search_results']}")
    
    print(f"\n--- Condition A (No RAG) ---")
    print(f"Input: {sample['condition_a']['input']}")
    
    if sample['condition_b']:
        print(f"\n--- Condition B (Good RAG) ---")
        print(f"Document (rank {sample['condition_b']['rank']}): {sample['condition_b']['document'][:100]}...")
        print(f"Input: {sample['condition_b']['input'][:150]}...")
    
    if sample['condition_c']:
        print(f"\n--- Condition C (Bad RAG) ---")
        print(f"Document (rank {sample['condition_c']['rank']}): {sample['condition_c']['document'][:100]}...")
        print(f"Input: {sample['condition_c']['input'][:150]}...")
    
    print("\n" + "=" * 70)
    print("Phase 1 - Conditions created successfully!")
    print("Next step: phase1_generate_outputs.py")
    print("=" * 70)


if __name__ == "__main__":
    main()