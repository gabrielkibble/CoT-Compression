"""
Adapted from Bowen1911/Difficulty-Perception-of-LLMs' deepmath_weighted_llm_emb.py
for DeepSeek-R1-Distill-Qwen-1.5B (hidden_size=1536).

Run this INSIDE your CREST container (needs datasets, transformers, torch
already installed there). Run detached, e.g.:
    nohup python3 extract_embeddings_1_5b.py > extract_embeddings.log 2>&1 &
    tail -f extract_embeddings.log
"""
import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # adjust if needed inside your container

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_PATH = "data/deepmath_embedding_1.5b.parquet"

# ---------------------------------------------------------------------------
# 1. Load DeepMath and select a balanced subset (same difficulty levels/caps
#    as the original paper's script)
# ---------------------------------------------------------------------------
print("Loading DeepMath-103K...")
ds = load_dataset("zwhe99/DeepMath-103K")

target_difficulties = [4.5, 5.0, 4.0, 3.0, 5.5, 8.0, 6.5, 8.5, 7.0, 9.0, 3.5, 7.5, 6.0]

records = []
for dt in ds["train"]:
    if dt["difficulty"] in target_difficulties:
        records.append({
            "difficulty": dt["difficulty"],
            "question": dt["question"],
            "final_answer": dt["final_answer"],
            "emb": None,
        })

df = pd.DataFrame(records)
dfd = df.groupby("difficulty").head(900).reset_index(drop=True)
print(f"Selected {len(dfd)} examples across {dfd['difficulty'].nunique()} difficulty levels.")

# ---------------------------------------------------------------------------
# 2. Load the model/tokenizer and extract final-token, final-layer embeddings
# ---------------------------------------------------------------------------
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
model.eval()

with torch.no_grad():
    for idx, row in tqdm(dfd.iterrows(), total=len(dfd)):
        chat = [{"role": "user", "content": row["question"]}]
        inputs = tokenizer.apply_chat_template(
            chat, tokenize=True, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)

        outputs = model(inputs, output_hidden_states=True)
        # hidden_states[-1] = final layer; [:, -1, :] = final token
        emb = outputs.hidden_states[-1][:, -1, :].squeeze().cpu().tolist()
        dfd.at[idx, "emb"] = emb

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
dfd.to_parquet(OUTPUT_PATH, index=False)
print(f"Saved embeddings to {OUTPUT_PATH}")
