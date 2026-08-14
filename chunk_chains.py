"""
Phase 2: Chunk chains into semantic steps.

Phase 1 saved, per chain, a residual vector for EVERY generated token.
That's too fine-grained for DTW -- token-level alignment mostly captures
shared wording, not shared computation. Here we group tokens into
semantic "steps" (roughly: sentences / lines of the reasoning) and
collapse each step down to one representative residual vector.

For each chunk we keep two summaries:
  - residual_last: residual at the chunk's FINAL token (the state right
    as the model "closes out" that step)
  - residual_mean: residual averaged over all tokens in the chunk

We don't know yet which one DTW will find more informative -- keeping
both is cheap and lets Phase 3 compare.

Input:  chain_captures/<problem_id>.pt   (Phase 1 output)
Output: chunked_captures/<problem_id>.pt

Run this after capture_chains.py. It only reads Phase 1 output, no model
needed, so it's fast and CPU-only.
"""

import os
import re
import glob
import torch

IN_DIR = "chain_captures"
OUT_DIR = "chunked_captures"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Splitting the chain text into semantic chunks
# ---------------------------------------------------------------------------
# Split on newlines first (CoT chains are often naturally one step per
# line), then further split any long line on sentence-ending punctuation.
# The negative lookbehind/lookahead on the period avoids splitting decimal
# numbers like "3.5" into "3." + "5".
SENTENCE_SPLIT_RE = re.compile(r"(?<!\d)([.!?])(?!\d)\s+")


def split_into_chunks(text: str) -> list[str]:
    """
    Split chain text into a list of non-empty chunk strings.
    Simple two-stage split: by line, then by sentence within long lines.
    """
    chunks = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Re-join the sentence-split pieces so punctuation stays attached
        # to the sentence it ends (split keeps the delimiter as its own
        # captured group because of the parens in the regex).
        pieces = SENTENCE_SPLIT_RE.split(line)
        sentence = ""
        rebuilt = []
        for piece in pieces:
            if piece in (".", "!", "?"):
                sentence += piece
                rebuilt.append(sentence.strip())
                sentence = ""
            else:
                sentence += piece
        if sentence.strip():
            rebuilt.append(sentence.strip())
        chunks.extend(c for c in rebuilt if c)
    return chunks


# ---------------------------------------------------------------------------
# Mapping chunk text back onto token indices
# ---------------------------------------------------------------------------
def build_char_offsets(tokenizer, token_ids: torch.Tensor):
    """
    Decode tokens one at a time, tracking the character offset each token
    starts at in the fully-decoded text. This lets us map a chunk's
    character span (found via string search) back to a token index range.

    Returns: full_text, list of (token_idx, start_char, end_char)
    """
    offsets = []
    cursor = 0
    running_text = ""
    for i, tid in enumerate(token_ids.tolist()):
        prev_text = running_text
        running_text = tokenizer.decode(token_ids[: i + 1].tolist(), skip_special_tokens=True)
        # The newly added text for this token is whatever got appended.
        # Using decode-of-prefix rather than decode-of-single-token because
        # BPE detokenization is context dependent (spacing etc.).
        start = len(prev_text)
        end = len(running_text)
        offsets.append((i, start, end))
        cursor = end
    return running_text, offsets


def map_chunks_to_token_ranges(full_text: str, chunks: list[str], offsets):
    """
    For each chunk string, find its character span in full_text (search
    forward from the end of the previous chunk to keep repeated substrings
    unambiguous), then find which token indices overlap that span.
    Returns list of (chunk_text, token_start_idx, token_end_idx_inclusive).
    """
    results = []
    search_from = 0
    for chunk_text in chunks:
        idx = full_text.find(chunk_text, search_from)
        if idx == -1:
            # Shouldn't normally happen since chunks were derived from
            # full_text, but guard against whitespace-normalization edge
            # cases (e.g. multiple spaces collapsed during our split).
            continue
        char_start = idx
        char_end = idx + len(chunk_text)
        search_from = char_end

        token_start_idx = None
        token_end_idx = None
        for tok_i, s, e in offsets:
            if e <= char_start:
                continue
            if s >= char_end:
                break
            if token_start_idx is None:
                token_start_idx = tok_i
            token_end_idx = tok_i

        if token_start_idx is None:
            continue
        results.append((chunk_text, token_start_idx, token_end_idx))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def chunk_chain(chain: dict, tokenizer) -> dict:
    """
    Given one Phase-1 chain dict, produce chunk-level summaries.
    Adds a "chunks" key to the chain dict (keeps original per-token data
    too, in case Phase 3 wants to fall back to it).
    """
    token_ids = chain["token_ids"]
    residuals = chain["residuals"]      # (n_tokens, hidden)
    prob_trace = chain["prob_trace"]    # (n_tokens,)

    full_text, offsets = build_char_offsets(tokenizer, token_ids)
    raw_chunks = split_into_chunks(full_text)
    mapped = map_chunks_to_token_ranges(full_text, raw_chunks, offsets)

    chunk_records = []
    for chunk_text, tok_start, tok_end in mapped:
        seg_residuals = residuals[tok_start: tok_end + 1]   # (n, hidden)
        seg_probs = prob_trace[tok_start: tok_end + 1]       # (n,)
        chunk_records.append({
            "text": chunk_text,
            "token_start": tok_start,
            "token_end": tok_end,
            "residual_last": seg_residuals[-1].clone(),
            "residual_mean": seg_residuals.mean(dim=0),
            "prob_last": seg_probs[-1].item(),
            "prob_mean": seg_probs.mean().item(),
        })

    chain_out = dict(chain)  # shallow copy, keep original fields
    chain_out["chunks"] = chunk_records
    return chain_out


def main():
    # We only need the tokenizer here (for decoding), not the full model --
    # much cheaper than Phase 1.
    from transformers import AutoTokenizer
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    in_paths = sorted(glob.glob(os.path.join(IN_DIR, "*.pt")))
    print(f"Found {len(in_paths)} problem files in {IN_DIR}")

    for path in in_paths:
        data = torch.load(path, weights_only=False)
        pid = data["problem_id"]

        new_chains = []
        chunk_count_stats = []
        for chain in data["chains"]:
            chunked = chunk_chain(chain, tokenizer)
            new_chains.append(chunked)
            chunk_count_stats.append(len(chunked["chunks"]))

        data["chains"] = new_chains
        out_path = os.path.join(OUT_DIR, f"{pid}.pt")
        torch.save(data, out_path)

        avg_chunks = sum(chunk_count_stats) / len(chunk_count_stats) if chunk_count_stats else 0
        print(f"[{pid}] {len(new_chains)} chains, "
              f"avg {avg_chunks:.1f} chunks/chain -> saved {out_path}")


if __name__ == "__main__":
    main()