#!/usr/bin/env python3
"""
Phase 1: Label data
Compute F1 scores and determine if RAG is helpful/neutral/harmful
"""

import json
import os


def compute_f1(prediction: str, reference: str) -> float:
    """
    Compute F1 score (token-level)
    
    Args:
        prediction: Generated text
        reference: Ground truth answer
    
    Returns:
        F1 score (0.0 to 1.0)
    """
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    
    if not ref_tokens:
        return 0.0
    
    common = len(pred_tokens & ref_tokens)
    
    if not pred_tokens:
        precision = 0.0
    else:
        precision = common / len(pred_tokens)
    
    recall = common / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def label_rag_effect(f1_with_rag: float, f1_without_rag: float, 
                    threshold: float = 0.05) -> int:
    """
    Label whether RAG helped or hurt
    
    Returns:
        +1: Helpful (improved by at least threshold)
         0: Neutral (no significant change)
        -1: Harmful (degraded by at least threshold)
    """
    diff = f1_with_rag - f1_without_rag
    
    if diff >= threshold:
        return 1  # Helpful
    elif diff <= -threshold:
        return -1  # Harmful
    else:
        return 0  # Neutral


def main():
    # Load outputs
    outputs_file = '../outputs/phase1_outputs.jsonl'
    
    results = []
    with open(outputs_file, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    
    print(f"Loaded {len(results)} results\n")
    
    # Label each result
    labeled_results = []
    
    print("Computing F1 scores and labeling...")
    for i, result in enumerate(results):
        
        # Compute F1 for each condition
        f1_no_rag = compute_f1(result['outputs']['no_rag'], result['ground_truth'])
        f1_good_rag = compute_f1(result['outputs']['good_rag'], result['ground_truth']) if result['outputs']['good_rag'] else 0.0
        f1_bad_rag = compute_f1(result['outputs']['bad_rag'], result['ground_truth']) if result['outputs']['bad_rag'] else 0.0
        
        # Label RAG effects
        label_good_rag = label_rag_effect(f1_good_rag, f1_no_rag)
        label_bad_rag = label_rag_effect(f1_bad_rag, f1_no_rag)
        
        # Create labeled result
        labeled_result = {
            'question_id': result['question_id'],
            'question': result['question'],
            'ground_truth': result['ground_truth'],
            'f1_scores': {
                'no_rag': round(f1_no_rag, 4),
                'good_rag': round(f1_good_rag, 4),
                'bad_rag': round(f1_bad_rag, 4)
            },
            'labels': {
                'good_rag': label_good_rag,  # +1, 0, or -1
                'bad_rag': label_bad_rag     # +1, 0, or -1
            },
            'outputs': result['outputs']
        }
        
        labeled_results.append(labeled_result)
        
        if (i + 1) % 10 == 0:
            print(f"  Labeled {i + 1}/{len(results)} samples")
    
    # Save labeled results
    labeled_file = '../outputs/phase1_labeled_data.jsonl'
    os.makedirs(os.path.dirname(labeled_file), exist_ok=True)
    
    with open(labeled_file, 'w', encoding='utf-8') as f:
        for result in labeled_results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nSaved {len(labeled_results)} labeled samples to {labeled_file}")
    
    # Compute statistics
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    
    good_rag_labels = [r['labels']['good_rag'] for r in labeled_results]
    bad_rag_labels = [r['labels']['bad_rag'] for r in labeled_results]
    
    print("\nGood RAG distribution:")
    print(f"  Helpful (+1): {good_rag_labels.count(1)} ({100*good_rag_labels.count(1)//len(good_rag_labels)}%)")
    print(f"  Neutral (0):  {good_rag_labels.count(0)} ({100*good_rag_labels.count(0)//len(good_rag_labels)}%)")
    print(f"  Harmful (-1): {good_rag_labels.count(-1)} ({100*good_rag_labels.count(-1)//len(good_rag_labels)}%)")
    
    print("\nBad RAG distribution:")
    print(f"  Helpful (+1): {bad_rag_labels.count(1)} ({100*bad_rag_labels.count(1)//len(bad_rag_labels)}%)")
    print(f"  Neutral (0):  {bad_rag_labels.count(0)} ({100*bad_rag_labels.count(0)//len(bad_rag_labels)}%)")
    print(f"  Harmful (-1): {bad_rag_labels.count(-1)} ({100*bad_rag_labels.count(-1)//len(bad_rag_labels)}%)")
    
    # Print sample
    print("\n" + "=" * 70)
    print("SAMPLE LABELED DATA")
    print("=" * 70)
    
    sample = labeled_results[0]
    print(f"\nQuestion: {sample['question']}")
    print(f"Ground Truth: {sample['ground_truth']}")
    print(f"\nF1 Scores:")
    print(f"  No RAG:  {sample['f1_scores']['no_rag']}")
    print(f"  Good RAG: {sample['f1_scores']['good_rag']}")
    print(f"  Bad RAG:  {sample['f1_scores']['bad_rag']}")
    print(f"\nLabels:")
    print(f"  Good RAG: {sample['labels']['good_rag']} (1=helpful, 0=neutral, -1=harmful)")
    print(f"  Bad RAG:  {sample['labels']['bad_rag']} (1=helpful, 0=neutral, -1=harmful)")
    
    print("\n" + "=" * 70)
    print("Phase 1 COMPLETE!")
    print("=" * 70)
    print(f"\nYou now have:")
    print(f"  - phase1_labeled_data.jsonl (50 labeled samples)")
    print(f"\nReady for Phase 2: Signal Extraction")


if __name__ == "__main__":
    main()