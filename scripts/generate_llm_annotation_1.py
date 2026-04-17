import json
import os
import concurrent.futures
from pathlib import Path
from dotenv import load_dotenv
import requests

# --- LOAD ENVIRONMENT VARIABLES FROM .env FILE ---
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

print(f"📂 Looking for .env at: {env_path}")
print(f"✅ .env file found: {env_path.exists()}")

# --- API CONFIGURATION ---
HF_API_KEY = os.getenv("HF_API_KEY")  # Get from https://huggingface.co/settings/tokens

if not HF_API_KEY:
    print("⚠️  WARNING: HF_API_KEY not found in .env file")
else:
    print("✅ HF_API_KEY loaded successfully")


def get_system_prompt():
    return """Act as a rigorous linguistic auditor for a Research Project. 
Your goal is to identify if an AI is hallucinating based ONLY on a provided context.

### DEFINITIONS:
- [G] = GROUNDED: Verbatim in text, logically inferred, or common knowledge (e.g. "Paris is a city", "the", "and").
- [H] = HALLUCINATED: Information NOT in the text (specific dates, numbers, or names not provided).

### THE GHOST RULE:
If a noun is hallucinated (e.g., "The blue car" when no car exists), the article "The" and adjective "blue" are also [H] because they point to a non-existent entity.

### FORMAT:
You MUST return the response with a [G] or [H] inside every bracket. 
Do not provide any other text. Example: The [G] cat [H] sat [G] on [G] the [H] moon [H]."""


def get_user_prompt(example):
    context = " ".join(example.get("context_retrieved", []))
    response = example.get("model_response", "")
    tokens = response.split()
    token_format = " ".join([f"{t} [ ]" for t in tokens])

    return f"QUESTION: {example.get('question')}\nCONTEXT: {context}\nAI RESPONSE TO LABEL:\n{token_format}"


# --- MODEL WRAPPERS USING HUGGING FACE FREE API ---

def call_llama2(prompt):
    """Call Llama 2 70B via Hugging Face"""
    try:
        api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}

        payload = {
            "inputs": f"{get_system_prompt()}\n\n{prompt}",
            "parameters": {"max_length": 512}
        }

        response = requests.post(api_url, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            return result[0]['generated_text'][-200:]  # Last 200 chars
        else:
            error_msg = response.json().get('error', str(response.status_code))
            print(f"❌ Llama2 Error: {error_msg}")
            return f"[Llama2 Error: {error_msg}]"
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Llama2 Error: {error_msg}")
        return f"[Llama2 Error: {error_msg}]"


def call_mistral(prompt):
    """Call Mistral 7B via Hugging Face"""
    try:
        api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}

        payload = {
            "inputs": f"{get_system_prompt()}\n\n{prompt}",
            "parameters": {"max_length": 512}
        }

        response = requests.post(api_url, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            return result[0]['generated_text'][-200:]  # Last 200 chars
        else:
            error_msg = response.json().get('error', str(response.status_code))
            print(f"❌ Mistral Error: {error_msg}")
            return f"[Mistral Error: {error_msg}]"
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Mistral Error: {error_msg}")
        return f"[Mistral Error: {error_msg}]"


def call_zephyr(prompt):
    """Call Zephyr 7B via Hugging Face"""
    try:
        api_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}

        payload = {
            "inputs": f"{get_system_prompt()}\n\n{prompt}",
            "parameters": {"max_length": 512}
        }

        response = requests.post(api_url, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            return result[0]['generated_text'][-200:]  # Last 200 chars
        else:
            error_msg = response.json().get('error', str(response.status_code))
            print(f"❌ Zephyr Error: {error_msg}")
            return f"[Zephyr Error: {error_msg}]"
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Zephyr Error: {error_msg}")
        return f"[Zephyr Error: {error_msg}]"


# --- MAIN EXECUTION ---

def process_example(example):
    """Process a single example through all 3 open-source LLMs in parallel"""
    prompt = get_user_prompt(example)
    example_id = example.get('id', 'N/A')
    print(f"⏳ Annotating ID: {example_id}...")

    # Run all 3 LLMs in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_llama = executor.submit(call_llama2, prompt)
        future_mistral = executor.submit(call_mistral, prompt)
        future_zephyr = executor.submit(call_zephyr, prompt)

        # Wait for all futures to complete
        llama_result = future_llama.result()
        mistral_result = future_mistral.result()
        zephyr_result = future_zephyr.result()

        result = {
            "id": example_id,
            "question": example.get("question"),
            "original_response": example.get("model_response"),
            "annotations": {
                "llama2_70b": llama_result,
                "mistral_7b": mistral_result,
                "zephyr_7b": zephyr_result,
                "human": ""  # Ready for manual annotation
            }
        }

        print(f"✅ Completed ID: {example_id}")
        return result


def main():
    """Main execution function"""
    data_path = Path(r'C:\college\Github\Research\RAG-Hallucination-Detection\data\generated\generated_50_example.json')

    if not data_path.exists():
        print(f"❌ ERROR: File not found at {data_path}")
        return

    print(f"📂 Loading data from: {data_path}")

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded {len(data)} examples")
    except Exception as e:
        print(f"❌ ERROR: Could not read file: {e}")
        return

    print("\n" + "=" * 60)
    print("🚀 STARTING ANNOTATION PROCESS (Open-Source LLMs)")
    print("=" * 60 + "\n")

    try:
        final_dataset = [process_example(ex) for ex in data]
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        return
    except Exception as e:
        print(f"\n❌ ERROR during processing: {e}")
        return

    output_path = Path('multi_llm_consensus_results_opensource.json')

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_dataset, f, indent=4, ensure_ascii=False)

        print("\n" + "=" * 60)
        print("✅ ANNOTATION COMPLETE")
        print("=" * 60)
        print(f"📊 Results saved to: {output_path.absolute()}")
        print(f"📈 Total examples processed: {len(final_dataset)}")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"❌ ERROR: Could not save results: {e}")


if __name__ == "__main__":
    main()