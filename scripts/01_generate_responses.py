from src.utils import config
from src.data.generator import ResponseGenerator

# Paths
examples = config.load_pkl(config.EXAMPLES_PKL)
responses_output = config.RESPONSES_JSON

# Generate
generator = ResponseGenerator(model_path=config.LLM_MODEL_PATH)
responses = generator.generate(examples)

# Save
config.save_json(responses, responses_output)
print(f"✅ Saved {len(responses)} responses to {responses_output}")