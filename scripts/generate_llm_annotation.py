import json
import os
import concurrent.futures
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
import google.genai as genai

# --- LOAD ENVIRONMENT VARIABLES FROM .env FILE (Parent Directory) ---
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

print(f"📂 Looking for .env at: {env_path}")
print(f"✅ .env file found: {env_path.exists()}")

# --- API CONFIGURATION ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Validate API keys
if not OPENAI_API_KEY:
    print("⚠️  WARNING: OPENAI_API_KEY not found in .env file")
else:
    print("✅ OPENAI_API_KEY loaded successfully")

if not ANTHROPIC_API_KEY:
    print("⚠️  WARNING: ANTHROPIC_API_KEY not found in .env file")
else:
    print("✅ ANTHROPIC_API_KEY loaded successfully")

if not GOOGLE_API_KEY:
    print("⚠️  WARNING: GOOGLE_API_KEY not found in .env file")
else:
    print("✅ GOOGLE_API_KEY loaded successfully")

# Initialize clients
OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
CLAUDE_CLIENT = Anthropic(api_key=ANTHROPIC_API_KEY)
GEMINI_CLIENT = genai.Client(api_key=GOOGLE_API_KEY)  # New google-genai API


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


# --- MODEL WRAPPERS WITH ERROR HANDLING ---

def call_gpt(prompt):
    """Call GPT-4o with error handling"""
    try:
        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        print(f"❌ GPT-4o Error: {error_msg}")
        return f"[GPT-4o Error: {error_msg}]"


def call_claude(prompt):
    """Call Claude 3.5 Sonnet with error handling"""
    try:
        response = CLAUDE_CLIENT.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            system=get_system_prompt(),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Claude Error: {error_msg}")
        return f"[Claude Error: {error_msg}]"


def call_gemini(prompt):
    """Call Gemini 1.5 Pro with error handling (using new google-genai API)"""
    try:
        response = GEMINI_CLIENT.models.generate_content(
            model="gemini-1.5-pro",
            contents=f"{get_system_prompt()}\n\n{prompt}"
        )
        return response.text
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Gemini Error: {error_msg}")
        return f"[Gemini Error: {error_msg}]"


# --- MAIN EXECUTION ---

def process_example(example):
    """Process a single example through all 3 LLMs in parallel"""
    prompt = get_user_prompt(example)
    example_id = example.get('id', 'N/A')
    print(f"⏳ Annotating ID: {example_id}...")

    # Run all 3 LLMs in parallel to save time
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_gpt = executor.submit(call_gpt, prompt)
        future_claude = executor.submit(call_claude, prompt)
        future_gemini = executor.submit(call_gemini, prompt)

        # Wait for all futures to complete
        gpt_result = future_gpt.result()
        claude_result = future_claude.result()
        gemini_result = future_gemini.result()

        result = {
            "id": example_id,
            "question": example.get("question"),
            "original_response": example.get("model_response"),
            "annotations": {
                "gpt4o": gpt_result,
                "claude3.5": claude_result,
                "gemini1.5": gemini_result,
                "human": ""  # Ready for manual annotation
            }
        }

        print(f"✅ Completed ID: {example_id}")
        return result


def main():
    """Main execution function"""
    # Use pathlib.Path with raw string to handle Windows paths correctly
    data_path = Path(r'C:\college\Github\Research\RAG-Hallucination-Detection\data\generated\generated_50_example.json')

    # Validate input file exists
    if not data_path.exists():
        print(f"❌ ERROR: File not found at {data_path}")
        print(f"📁 Please verify the path exists")
        return

    print(f"📂 Loading data from: {data_path}")

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded {len(data)} examples")
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON in input file: {e}")
        return
    except Exception as e:
        print(f"❌ ERROR: Could not read file: {e}")
        return

    # Process all examples
    print("\n" + "=" * 60)
    print("🚀 STARTING ANNOTATION PROCESS")
    print("=" * 60 + "\n")

    try:
        final_dataset = [process_example(ex) for ex in data]
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        return
    except Exception as e:
        print(f"\n❌ ERROR during processing: {e}")
        return

    # Save results
    output_path = Path('multi_llm_consensus_results.json')

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