# ============================================
#  Gemma-4 E2B IT  •  Dual T4  •  Kaggle
#  CELL 1 — Install & Load  (run once)
# ============================================
import subprocess, sys

print(">>> Upgrading transformers (gemma4 fix) ...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "git+https://github.com/huggingface/transformers.git",
    "accelerate"
])
print(">>> Done.\n")

import torch
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1"

# ── check both T4s ──────────────────────────────
assert torch.cuda.device_count() == 2, "Need 2 GPUs!"
print(f"  GPU 0 : {torch.cuda.get_device_name(0)}")
print(f"  GPU 1 : {torch.cuda.get_device_name(1)}")

# ── processor ───────────────────────────────────
processor = AutoProcessor.from_pretrained(
    MODEL,
    local_files_only=True,
)

# ── model split across BOTH T4s ─────────────────
#  E2B at FP16 ≈ 9.6 GB total, 35 transformer layers
#  max_memory forces the split so NEITHER GPU holds it all
#  → ~18 layers on GPU 0, ~17 layers on GPU 1
#  CPU is NOT used (zero offload → max tokens/sec)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16,              # T4 = FP16 (no native BF16)
    device_map="auto",                # let accelerate decide
    max_memory={0: "7GiB", 1: "7GiB"}, # force dual-GPU split
    attn_implementation="sdpa",       # fastest attention on T4
    local_files_only=True,
)
model.eval()

# ── verify layer distribution ───────────────────
gpu_layers = {0: 0, 1: 0}
for name, param in model.named_parameters():
    if "model.layers." in name:
        gpu = param.device.index
        if gpu in gpu_layers:
            gpu_layers[gpu] += 1

print()
print("  Layers on GPU 0 :", gpu_layers[0])
print("  Layers on GPU 1 :", gpu_layers[1])
print("  CPU offload     :", sum(1 for n,p in model.named_parameters()
                                if "model.layers." in n and p.device.type == "cpu"))
print()
print(">>> MODEL READY — chat below this cell.\n")

# chat history (preserved between Cell 2 re-runs)
chat_history = []
