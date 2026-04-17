import json
import random
import os


def generate_pilot_set():
    # 1. Define the absolute paths based on your setup
    input_path = r'C:\college\Github\Research\RAG-Hallucination-Detection\data\generated\responses.json'
    output_path = r'C:\college\Github\Research\RAG-Hallucination-Detection\data\generated\generated_50_example.json'

    print(f"--- Pilot Generation Started ---")

    # 2. Check if the input file actually exists
    if not os.path.exists(input_path):
        print(f"ERROR: Could not find the source file at: {input_path}")
        print(
            "Please check if the folder exists or if there's a typo in the filename (e.g., 'responce' vs 'response').")
        return

    try:
        # 3. Load the full response data
        with open(input_path, 'r', encoding='utf-8') as f:
            all_responses = json.load(f)

        total_count = len(all_responses)
        sample_size = min(50, total_count)

        print(f"Successfully loaded {total_count} responses.")

        # 4. Randomly select 50 examples
        # We use a seed so if you run this again, you get the same 'random' set
        random.seed(42)
        pilot_set = random.sample(all_responses, sample_size)

        # 5. Save the 50 examples to the new location
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pilot_set, f, indent=4)

        print(f"SUCCESS: Generated {sample_size} examples for the pilot study.")
        print(f"Location: {output_path}")
        print(f"--- Ready for Annotation ---")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    generate_pilot_set()