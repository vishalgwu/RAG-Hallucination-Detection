"""Test if LLaMA model loads correctly"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import LLAMA2_7B_PATH
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_model_loading():
    """Test loading and basic inference"""
    print("=" * 80)
    print("TESTING LLAMA-2-7B MODEL LOADING")
    print("=" * 80)

    # Check paths
    print(f"\n[1] Checking paths...")
    print(f"    Model path: {LLAMA2_7B_PATH}")
    print(f"    Exists: {LLAMA2_7B_PATH.exists()}")

    if not LLAMA2_7B_PATH.exists():
        print(f"    ⏳ Model still downloading...")
        return False

    # Load tokenizer
    print(f"\n[2] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(LLAMA2_7B_PATH))
        print(f"    ✅ Tokenizer loaded (vocab size: {len(tokenizer)})")
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False

    # Load model (CPU version - simpler)
    print(f"\n[3] Loading model (CPU mode)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(LLAMA2_7B_PATH),
            torch_dtype=torch.float32
        )
        print(f"    ✅ Model loaded")
        print(f"    Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params")
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False

    # Test inference
    print(f"\n[4] Testing inference (this may take 1-2 min on CPU)...")
    try:
        prompt = "What is the capital of France?"
        inputs = tokenizer(prompt, return_tensors="pt")

        print(f"    Running inference...")
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=50,
                do_sample=False
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"    ✅ Inference successful")
        print(f"    Response: {response}")
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False

    print(f"\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - MODEL IS READY")
    print("=" * 80)
    return True

if __name__ == "__main__":
    test_model_loading()

