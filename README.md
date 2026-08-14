# CoT Compression

Internal-trajectory analysis of Llama-3.1-8B-Instruct chains of thought on
GSM8K, and a DBSCAN-based method for compressing a chain by dropping
internally redundant reasoning steps.

This repo contains only the pipeline **code**. None of the generated data,
model caches, or intermediate artifacts are included (they're large --
multiple GB -- and easily regenerated). This README walks through producing
everything needed, in order, ending with the DBSCAN compression scripts.

## Requirements

```bash
pip install torch transformers accelerate numpy scikit-learn matplotlib --break-system-packages
```

- **GPU** with enough VRAM for Llama-3.1-8B-Instruct in bf16 (~16GB minimum).
- **Hugging Face access** to `meta-llama/Llama-3.1-8B-Instruct` (accept the
  license on the model's HF page, then set `HF_TOKEN` in your environment):
  ```bash
  export HF_TOKEN=hf_your_token_here
  ```
- **A Phase-0-filtered problem set** (see below) -- NOT included in this repo.

## Prerequisite data: the Phase 0 filtered problem set

Every script downstream assumes you already have a `.jsonl` file of GSM8K
problems that the model gets **wrong without chain-of-thought but right
with it**. This filter matters: it ensures every result is evidence about
reasoning, not about the model already knowing the answer.

Each line needs at minimum:
```json
{"question": "...", "gold_answer": "18", "no_cot_correct": false, "cot_correct": true}
```

If you don't already have this file, produce it by: for each GSM8K problem,
sample a no-CoT completion and a CoT completion from the target model, check
correctness of each against the gold answer, and keep only rows where
`no_cot_correct=false` and `cot_correct=true`.

Update `PROBLEMS_PATH` at the top of `capture_chains.py` to point at your file.

## Pipeline: run in this order

### 1. Extract the embedding matrix (one-time, ~1GB output)
```bash
python3 extract_embeddings.py
```
Produces `embed_matrix.npy`. Needed for the anisotropy correction used
throughout (projects out each token's own embedding direction from its
residual state).

### 2. Sample chains of thought (GPU, the expensive step)
```bash
python3 capture_chains.py
```
Edit `MAX_PROBLEMS` and `N_CHAINS_PER_PROBLEM` at the top first (defaults are
conservative -- start small to sanity check before scaling up). Produces
`chain_captures/<problem_id>.pt` -- for every sampled chain, every token's
residual stream at a middle layer (`LAYER=16` by default) plus a running
answer-probability trace (logit-lens).

### 3. Chunk chains into reasoning steps (CPU, fast)
```bash
python3 chunk_chains.py
```
Reads `chain_captures/`, splits each chain into semantic chunks (roughly one
reasoning step each), produces `chunked_captures/<problem_id>.pt`.

### 4. DTW alignment + anisotropy correction (CPU, fast)
```bash
python3 dtw_align_v2.py
```
Reads `chunked_captures/` and `embed_matrix.npy`. Fits the anisotropy
correction (mean-centering, token-direction projection, per-dimension
standardization) on the full dataset and computes DTW distances between
every pair of sampled chains. Produces `anisotropy_correction.npz` (needed
by every later script) and `phase3_pairwise.pkl`.

### 5. Build consensus routes (CPU, fast)
```bash
python3 build_consensus_routes.py
```
Picks, per problem, the "medoid" correct chain -- the one with lowest
average DTW distance to every other correct chain for that problem.
Produces `consensus_routes.pkl`.

### 6. Layer sweep -- extract all-layer residuals (GPU, moderate cost)
```bash
python3 layer_sweep.py
```
For every chain, re-runs it through the model as a single teacher-forced
forward pass (no sampling) capturing residuals at **every** layer at once.
Produces `layer_sweep_residuals.pkl` -- **this is what the DBSCAN
compression scripts actually read from**, since it has both layer 16 and
the final layer cached without needing another forward pass. Also prints a
layer-by-layer comparison of which layer best separates correct from
incorrect reasoning (empirically, layer 16 wins on this dataset).

## The DBSCAN compression approach

With the above artifacts in place, three scripts explore compressing a
chain by dropping internally redundant steps (steps whose residual states
cluster together within a single chain):

```bash
python3 dbscan_compress.py           # exploratory: cluster + compress a
                                      # handful of problems' medoid chains,
                                      # print the resulting text
python3 verify_compressed_chains.py  # rigorous check: strip the answer
                                      # line from compressed chains, force
                                      # the model to re-derive it, measure
                                      # real accuracy + token reduction
python3 eps_scaling_law.py           # sweep DBSCAN's eps parameter finely,
                                      # trace accuracy and token-reduction
                                      # as curves -> eps_scaling_law.csv/.png
```

Key parameters to know about (top of each file):
- `EPS_VALUES` / `EPS_GRID` -- DBSCAN's neighborhood radius (cosine
  distance). Larger eps = more aggressive compression. Empirically on this
  dataset, ~0.35-0.45 preserves accuracy well; ~0.65+ starts dropping
  load-bearing computation, not just redundant narration.
- `LAYERS_TO_TRY` (in `dbscan_compress.py`) -- layer 16 (best for
  correct/incorrect separation) shows almost no *within-chain* redundancy
  to exploit; the final layer picks up more surface-level/stylistic
  redundancy and is what the compression scripts default to.
- `N_PROBLEMS` -- how many problems to run on. Start small, scale up once
  you trust the pipeline (all of the above scripts checkpoint or cache
  where possible).

## Other scripts in this repo

- `guided_generate.py` -- Phase 5: best-of-K search generating a new short
  chain, scored by answer-probability, route-proximity, or both, against
  the consensus route from step 5 above.
- `dtw_align_velocity.py` -- same DTW comparison as step 4, but on
  step-to-step *changes* in residual state rather than the states
  themselves.
- `validate_latent_rewards.py` -- checks whether DTW distance to a
  problem's consensus route correlates with chain correctness (a cheap
  offline validation before investing in RL-based training).
- `cluster_stats.py` -- clustered bootstrap confidence intervals and a
  within-problem permutation test on the DTW distances from step 4.
- `compare_methods.py` -- aggregates accuracy-vs-tokens across whichever
  methods you've run, into one comparison table/chart.

## A note on scale

Everything above was developed and validated on 10-100 filtered GSM8K
problems. Intermediate files get large fast -- `chain_captures/` alone was
~5-6GB at 100 problems -- so if you're on a shared/quota-limited machine,
consider pointing the working directory at scratch storage rather than a
home directory from the start.