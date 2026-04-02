#!/usr/bin/env python3
"""
Phase 1: Generate model outputs for three conditions
Uses a small, fast model (GPT-2)
"""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def load_conditions(conditions_file: str):
    """Load conditions from previous step"""
    with open(conditions_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def setup_model(model_name: str = "gpt2"):
    """Load tokenizer and model"""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer, model


def generate_answer(model, tokenizer, prompt: str, max_tokens: int = 50) -> str:
    """
    Generate answer given prompt
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_tokens: Max tokens to generate
    
    Returns:
        Generated text (just the new part, not the prompt)
    """
    # Tokenize
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Truncate if too long
    if inputs.shape[1] > 300:
        inputs = inputs[:, :300]
    
    prompt_length = inputs.shape[1]
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            inputs,
            max_length=prompt_length + max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Return only the generated part
    generated = full_text[len(prompt):].strip()
    
    return generated


def main():
    print("DEBUG: Starting main()", flush=True)
    # Load conditions
    conditions_file = '../outputs/phase1_conditions.json'
    print(f"DEBUG: Loading from {conditions_file}", flush=True)
    conditions_list = load_conditions(conditions_file)
    
    print(f"Loaded {len(conditions_list)} conditions\n", flush=True)
    
    # Setup model
    tokenizer, model = setup_model("gpt2")
    
    # Generate outputs
    results = []
    
    print("Generating outputs for three conditions...")
    for i, sample in enumerate(tqdm(conditions_list, desc="Generating")):
        
        # Generate for each condition
        output_a = generate_answer(
            model, tokenizer,
            sample['condition_a']['input']
        )
        
        output_b = None
        if sample['condition_b']:
            output_b = generate_answer(
                model, tokenizer,
                sample['condition_b']['input']
            )
        
        output_c = None
        if sample['condition_c']:
            output_c = generate_answer(
                model, tokenizer,
                sample['condition_c']['input']
            )
        
        # Store results
        results.append({
            'question_id': sample['question_id'],
            'question': sample['question'],
            'ground_truth': sample['ground_truth'],
            'outputs': {
                'no_rag': output_a,
                'good_rag': output_b,
                'bad_rag': output_c
            }
        })
    
    # Save results
    output_file = '../outputs/phase1_outputs.jsonl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nSaved {len(results)} outputs to {output_file}")
    
    # Print sample
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT")
    print("=" * 70)
    sample_result = results[0]
    print(f"\nQuestion: {sample_result['question']}")
    print(f"Ground Truth: {sample_result['ground_truth']}")
    print(f"\nOutput (no RAG): {sample_result['outputs']['no_rag'][:80]}...")
    if sample_result['outputs']['good_rag']:
        print(f"Output (good RAG): {sample_result['outputs']['good_rag'][:80]}...")
    if sample_result['outputs']['bad_rag']:
        print(f"Output (bad RAG): {sample_result['outputs']['bad_rag'][:80]}...")
    
    print("\n" + "=" * 70)
    print("Phase 1 - Outputs generated successfully!")
    print("Next step: phase1_label_data.py")
    print("=" * 70)


if __name__ == "__main__":
    main()