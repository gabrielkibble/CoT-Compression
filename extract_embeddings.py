"""
One-time utility: dump the model's input embedding matrix to disk.

Why we need this: the anisotropy correction "projects out the current-
token direction" from each residual state. That means, for the token
actually generated at a given position, we remove the component of the
residual that points along THAT token's own embedding direction (residual
streams tend to strongly encode "which token is this" as a side effect of
eventually feeding the unembedding matrix, which inflates raw similarity
between any two states that happen to end in similar tokens).

We only need the embedding matrix once -- not the whole model -- so this
is a cheap one-off dump you run a single time, not part of the main
pipeline.

Saves: embed_matrix.npy, shape (vocab_size, hidden_dim), float16.
float16 keeps the file a manageable size (~1GB for Llama-3.1-8B's
128k vocab x 4096 hidden) while being precise enough for direction
projection (we only need unit directions, not exact magnitudes).
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

print("Loading model (CPU, just to grab the embedding matrix)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="cpu",
)

embed_matrix = model.get_input_embeddings().weight.detach().to(torch.float16).numpy()
print(f"Embedding matrix shape: {embed_matrix.shape}, dtype: {embed_matrix.dtype}")

np.save("embed_matrix.npy", embed_matrix)
print("Saved -> embed_matrix.npy")