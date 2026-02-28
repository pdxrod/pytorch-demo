#!/usr/bin/env python
# Written by ChatGPT5

import torch
import torch.nn as nn
import torch.nn.functional as F
import my_utils

def intro():
    print("")
    print("=" * 70)
    print("Tiny Transformer - A Minimal Implementation")
    print("=" * 70)
    print("""
This program demonstrates a minimal transformer model implementation that captures
the core concepts of attention-based neural networks.

WHAT IT DOES:
1. Implements a character-level tokenizer that converts text to numbers
2. Builds a tiny transformer with multi-head attention layers
3. Processes a text prompt through the model
4. Shows the attention matrices (how tokens relate to each other)
5. Outputs logits (raw predictions) for each position

KEY COMPONENTS:
- CharTokenizer: Converts characters to token IDs (32-127 ASCII printable chars)
- TinyAttention: Implements multi-head self-attention mechanism
- TinyTransformer: Stack of attention layers with embeddings and language model head
""")
    my_utils.wait_for_user_input()
    print("""
WHAT THIS SHOWS ABOUT TRANSFORMERS:
- Self-Attention: Each token can attend to any other token in the sequence
- Multi-Head: Multiple parallel attention heads learn different relationships
- Layer Stacking: Multiple layers build increasingly abstract representations
- Residual Connections: Allows information to flow through deep networks

This is a simplified, untrained model - real transformers like GPT have millions/billions
of parameters and are trained on massive datasets. This demonstrates the architecture
without requiring training.
""")
    print("=" * 70)
    print("")

# -----------------------------
# 1. Simple character tokenizer
# -----------------------------

class CharTokenizer:
    def __init__(self):
        chars = [chr(i) for i in range(32, 127)]
        self.vocab = {c: i for i, c in enumerate(chars)}
        self.inv = {i: c for c, i in self.vocab.items()}
        self.pad_id = 0

    def encode(self, text):
        return [self.vocab.get(c, self.pad_id) for c in text]

    def decode(self, ids):
        return "".join(self.inv.get(i, "?") for i in ids)

tokenizer = CharTokenizer()

# -----------------------------
# 2. Minimal transformer block
# -----------------------------

class TinyAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        H = self.n_heads
        D = self.head_dim

        q = self.q_proj(x).view(B, T, H, D)
        k = self.k_proj(x).view(B, T, H, D)
        v = self.v_proj(x).view(B, T, H, D)

        att = (q @ k.transpose(-1, -2)) / (D ** 0.5)
        att = att.softmax(dim=-1)

        y = (att @ v).contiguous().view(B, T, C)
        y = self.o_proj(y)

        return y, att

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=128, d_model=64, n_heads=4, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TinyAttention(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        h = self.embed(idx)
        all_att = []

        for layer in self.layers:
            y, att = layer(h)
            h = h + y
            all_att.append(att)

        h = self.ln(h)
        logits = self.lm_head(h)
        return logits, all_att

def main():
    intro()
    model = TinyTransformer()
    prompt = "Hello world!"
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    logits, attentions = model(ids)

    print("=== Input tokens ===")
    print(ids)

    my_utils.wait_for_user_input("Attention matrices")
    print("\n=== Attention matrices (summary) ===")
    for i, att in enumerate(attentions):
        print(f"\nLayer {i+1}: shape {att.shape}")
        # Show summary statistics instead of full tensor
        max_att = att[0].max().item()
        avg_att = att[0].mean().item()
        print(f"  Max attention: {max_att:.4f}, Avg attention: {avg_att:.4f}")
        # Show first token's attention pattern (first head only) for understanding
        first_token_att = att[0, 0, 0, :].tolist()
        print(f"  First token attention (head 0): {[f'{x:.3f}' for x in first_token_att[:8]]}...")
        print(f"  (Shows how the first token attends to other tokens in the sequence)")

    print("\n=== Output logits (per token) ===")
    print("Logits are raw prediction scores for each possible token, before the softmax activation is applied.")
    print("Each position in the sequence gets logits for all 128 possible characters.")
    print("Higher logit values indicate the model's prediction leans toward those characters.")
    print("After applying softmax, these become probability distributions.")
    print("\nLogits shape:", logits.shape)
    print("Sample logits for first token (first 10 characters):")
    print(logits[0, 0, :10].tolist())

if __name__ == "__main__":
    main()
    
  

