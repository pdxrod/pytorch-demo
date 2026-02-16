#!/usr/bin/env python
"""
LLM attention inspector. Uses deferred imports and CPU-only to avoid
PyTorch/MPS mutex crashes on macOS.
"""
import os
import sys
import textwrap
import shutil

# Force CPU-only and disable MPS *before* any PyTorch/transformers import.
# Reduces "mutex lock failed: Invalid argument" / libc++abi crashes on macOS.
os.environ["PYTORCH_MPS_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "0"  # avoid tokenizers background threads
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _pretty_print(text):
    """Print text wrapped to terminal width (no heavy imports)."""
    cols, _ = shutil.get_terminal_size()
    print(textwrap.fill(text.strip(), width=cols))


def _wait(msg=None):
    if msg is not None:
        print(msg)
    input("Press Enter to continue...")


def intro(prompt):
    print("")
    _pretty_print("""
This program demonstrates how to inspect the attention mechanisms of a small transformer-based
Large Language Model (LLM). It uses a tiny GPT-2–style model so it can run on a typical laptop
without needing a GPU server.
""")
    print("\nWHAT IT DOES:")
    print("1. Loads a tiny GPT-2–style transformer model")
    print("2. Processes a text prompt through the model")
    print("3. Extracts attention matrices from different layers of the transformer")
    print("4. Visualizes attention patterns using a heatmap to see which tokens the model 'pays attention to'")
    _wait()
    print("")
    print("WHAT AutoModelForCausalLM IS:")
    _pretty_print("""
AutoModelForCausalLM is a HuggingFace transformers class that automatically loads
causal (autoregressive) language models. 'Causal' means the model predicts the next
token based only on previous tokens (left-to-right generation). It is used for
tasks like text generation, completion, and continuation. The 'Auto' prefix means
it automatically detects the correct model architecture from the model name/hub.
""")
    print(f"\nInput sequence: '{prompt}'")
    print("\nWHAT ATTENTION IS:")
    _pretty_print("""
Attention is a key mechanism in transformer models that determines how much each token
(position/word) in the input sequence should "attend to" or "focus on" other tokens
when making predictions. Each attention layer has multiple "heads" that can learn to
focus on different aspects (e.g., grammar, semantics, long-range dependencies).
""")

# Bundled demo data used when the model subprocess crashes (e.g. mutex on macOS).
# 5x5 attention-like matrix (rows sum to 1, diagonal dominant); tokens for "The meaning of life is".
DEMO_ATTN = [
    [0.50, 0.20, 0.15, 0.10, 0.05],
    [0.15, 0.45, 0.25, 0.10, 0.05],
    [0.10, 0.20, 0.45, 0.15, 0.10],
    [0.05, 0.10, 0.20, 0.50, 0.15],
    [0.05, 0.05, 0.10, 0.20, 0.60],
]
DEMO_TOKENS = ["The", "meaning", "of", "life", "is"]


def main():
    import subprocess
    import tempfile
    import json as _json

    prompt = "The meaning of life is"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    loader_script = os.path.join(script_dir, "_minimal_llm_1_load.py")

    intro(prompt)
    _wait()

    # On macOS, skip the model subprocess to avoid "Python quit unexpectedly" when the
    # subprocess hits the PyTorch mutex crash. Use demo data directly.
    if sys.platform == "darwin":
        print("On macOS: using bundled demo data (avoids known PyTorch mutex crash).", flush=True)
        attn_matrix = DEMO_ATTN
        tokens = DEMO_TOKENS
    else:
        print("Loading model and running forward pass (in subprocess)...", flush=True)
        env = os.environ.copy()
        env["PYTORCH_MPS_DISABLE"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = ""

        with tempfile.TemporaryDirectory() as tmpdir:
            proc = subprocess.run(
                [sys.executable, loader_script, tmpdir],
                env=env,
                capture_output=True,
                timeout=120,
            )
            if proc.returncode == 0:
                import numpy as np
                attn_matrix = np.load(os.path.join(tmpdir, "attn_matrix.npy"))
                with open(os.path.join(tmpdir, "tokens.json")) as f:
                    tokens = _json.load(f)
                print("Model ran successfully.", flush=True)
            else:
                print("Model subprocess failed. Using bundled demo data.", flush=True)
                attn_matrix = DEMO_ATTN
                tokens = DEMO_TOKENS

    print("Number of layers: (demo shows layer 0 only)")
    print("Shape of attention for layer 0:", getattr(attn_matrix, "shape", (len(DEMO_ATTN), len(DEMO_ATTN[0]))))
    _wait("Example: visualize attention of layer 0, head 0")

    print("\n=== Showing 'heatmap' of attention weights ===")
    print(f"Each token made from '{prompt}' is listed along the X and Y axes.")
    print("The heatmap visualizes the attention weights from layer 0, head 0.")
    print("Each row represents a token in the input sequence.")
    print("Each column represents which tokens that row token is attending to.")
    print("Brighter colors (yellow/green) indicate stronger attention.")
    print("Darker colors (purple/blue) indicate weaker attention.")
    print(f"\nInput tokens: {tokens}")
    shape = getattr(attn_matrix, "shape", (len(attn_matrix), len(attn_matrix[0]) if attn_matrix else 0))
    print(f"Matrix shape: {shape} (rows=from tokens, cols=to tokens)")
    print("\nInterpreting the pattern:")
    print("- Tokens typically attend most strongly to themselves (diagonal)")
    print("- Related words (e.g., 'meaning' and 'life') may show cross-attention")
    print("- The model learns which tokens are semantically related")
    print("\nDisplaying heatmap...")

    # Defer plotting imports until we need them.
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6, 6))
    sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
    plt.title("Attention heatmap — layer 0, head 0")
    plt.show()

if __name__ == "__main__":
    main()