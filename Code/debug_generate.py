#!/usr/bin/env python3
"""Debug script for generate_outputs.py"""

import json
import os

print("=" * 60)
print("DEBUGGING generate_outputs.py")
print("=" * 60)

print("\nStep 1: Loading conditions with UTF-8")
try:
    conditions_file = '../outputs/phase1_conditions.json'
    with open(conditions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} conditions")
    print(f"  First condition keys: {list(data[0].keys())}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nStep 2: Setting up model")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading GPT-2 model...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    print(f"✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nStep 3: Testing generate_answer function")
try:
    import torch
    
    def generate_answer(model, tokenizer, prompt: str, max_tokens: int = 50) -> str:
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        if inputs.shape[1] > 300:
            inputs = inputs[:, :300]
        prompt_length = inputs.shape[1]
        
        with torch.no_grad():
            output = model.generate(
                inputs,
                max_length=prompt_length + max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated = full_text[len(prompt):].strip()
        return generated
    
    test_prompt = "What is 2+2?"
    print(f"Generating answer for: '{test_prompt}'")
    answer = generate_answer(model, tokenizer, test_prompt)
    print(f"✓ Generated: '{answer}'")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
