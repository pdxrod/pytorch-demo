#!/usr/bin/env python

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import my_utils

def intro( prompt ):
    print("")
    
    my_utils.pretty_print("""
This program demonstrates how to inspect the attention mechanisms of a small transformer-based
Large Language Model (LLM), specifically Falcon-7b. It was written by Cursor AI.
""")
    print("\nWHAT IT DOES:")
    print("1. Loads the Falcon-7b model, a 7-billion parameter transformer model")
    print("2. Processes a text prompt through the model")
    print("3. Extracts attention matrices from different layers of the transformer")
    print("4. Visualizes attention patterns using a heatmap to see which tokens the model 'pays attention to'")
    my_utils.wait_for_user_input()
    print("")
    print("WHAT AutoModelForCausalLM IS:")
    my_utils.pretty_print("""

AutoModelForCausalLM is a HuggingFace transformers class that automatically loads
causal (autoregressive) language models. 'Causal' means the model predicts the next
token based only on previous tokens (left-to-right generation). It is used for
tasks like text generation, completion, and continuation. The 'Auto' prefix means
it automatically detects the correct model architecture from the model name/hub.
""")
    print(f"\ninput sequence: '{prompt}'")
    print("\nWHAT ATTENTION IS:")
    my_utils.pretty_print("""   
Attention is a key mechanism in transformer models that determines how much each token
(position/word) in the input sequence should "attend to" or "focus on" other tokens
when making predictions. Each attention layer has multiple "heads" that can learn to
focus on different aspects (e.g., grammar, semantics, long-range dependencies).
""")
    
def main():
    prompt = "The meaning of life is"

    intro( prompt )
    my_utils.wait_for_user_input()

    model_name = "tiiuae/falcon-7b"
    print(f"Loading tokenizer {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )
    model.eval()

    print(f"Tokenizing prompt: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt")

    print("Running model forward pass with output_attentions and output_hidden_states...")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

    print("Get attention from last layer")
    attns = outputs.attentions  # tuple: one tensor per layer
    hidden = outputs.hidden_states  # tuple: hidden state after each layer

    print("Number of layers:", len(attns))
    print("Shape of attention for layer 0:", attns[0].shape)
    print("Shape of final hidden state:", hidden[-1].shape)

    my_utils.wait_for_user_input("Example: visualize attention of layer 0, head 0")
    attn_matrix = attns[0][0, 0].cpu().numpy()  # batch 0, head 0
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    print("\n=== Showing 'heatmap' of attention weights ===")
    print(f"Each token made from '{prompt}' is listed along the X and Y axes.")
    print("The heatmap visualizes the attention weights from layer 0, head 0.")
    print("Each row represents a token in the input sequence.")
    print("Each column represents which tokens that row token is attending to.")
    print("Brighter colors (yellow/green) indicate stronger attention.")
    print("Darker colors (purple/blue) indicate weaker attention.")
    print(f"\nInput tokens: {tokens}")
    print(f"Matrix shape: {attn_matrix.shape} (rows=from tokens, cols=to tokens)")
    print("\nInterpreting the pattern:")
    print("- Tokens typically attend most strongly to themselves (diagonal)")
    print("- Related words (e.g., 'meaning' and 'life') may show cross-attention")
    print("- The model learns which tokens are semantically related")
    print("\nDisplaying heatmap...")

    plt.figure(figsize=(6,6))
    sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
    plt.title("Attention heatmap — layer 0, head 0")
    plt.show()

if __name__ == "__main__":
    main()