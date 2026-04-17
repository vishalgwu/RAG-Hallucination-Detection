"""Download LLaMA-2-7B model from HuggingFace"""

from huggingface_hub import snapshot_download
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import LLAMA2_7B_PATH

model_name = 'meta-llama/Llama-2-7b-chat-hf'

print("=" * 80)
print(f"Downloading {model_name}...")
print(f"Saving to: {LLAMA2_7B_PATH}")
print("This will take 30-60 minutes depending on internet speed")
print("=" * 80)
print()

snapshot_download(
    model_name,
    cache_dir=str(LLAMA2_7B_PATH),
    local_dir=str(LLAMA2_7B_PATH)
)

print()
print("=" * 80)
print("✅ Download complete!")
size_gb = sum(f.stat().st_size for f in LLAMA2_7B_PATH.rglob('*') if f.is_file()) / 1e9
print(f"Model size: {size_gb:.1f} GB")
print("=" * 80)