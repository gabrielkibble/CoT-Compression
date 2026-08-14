"""
Diagnostic: is the injected (de-standardized) consensus vector even the
right SCALE compared to what the network naturally produces at layer 16?

The residual_steering.py experiment found that injecting
    raw_target = residual_corrected * std + mean
almost always produces generic punctuation as the immediate next token,
regardless of problem or step. One simple, cheap explanation to rule in
or out first: if raw_target's norm is systematically smaller (or larger)
than a real, naturally-occurring residual at that layer, the later layers
may be effectively ignoring it (too weak to registers as "real" content)
or destabilizing on it (too strong) -- either would produce something
close to noise/boilerplate rather than coherent content.

This just compares distributions of ||raw_target|| (from the consensus
routes) against ||natural residual|| (sampled from real generated chunks
in chunked_captures, i.e. what layer 16 actually looks like during normal
generation).

Requires: consensus_routes.pkl, anisotropy_correction.npz, chunked_captures/
"""

import glob
import pickle
import random
import torch
import numpy as np

ROUTES_PATH = "consensus_routes.pkl"
CORRECTION_PATH = "anisotropy_correction.npz"
CHUNKED_DIR = "chunked_captures"
RESIDUAL_KEY = "residual_last"
N_NATURAL_SAMPLES = 3000  # subsample of natural residuals for speed
RANDOM_SEED = 0

random.seed(RANDOM_SEED)


def summarize(name, values):
    arr = np.array(values)
    print(f"{name:30s} n={len(arr):5d}  mean={arr.mean():.2f}  std={arr.std():.2f}  "
          f"median={np.median(arr):.2f}  min={arr.min():.2f}  max={arr.max():.2f}")


def main():
    correction = np.load(CORRECTION_PATH)
    mean, std = correction["mean"], correction["std"]

    with open(ROUTES_PATH, "rb") as f:
        routes = pickle.load(f)

    # ---- Injected vector norms: de-standardize every consensus chunk ----
    injected_norms = []
    for pid, route in routes.items():
        for c in route["chunks"]:
            raw_target = c["residual_corrected"] * std + mean
            injected_norms.append(np.linalg.norm(raw_target))

    # ---- Natural residual norms: sample real layer-16 residuals from
    # actual generated chains (uncorrected, as they naturally occur) ----
    all_paths = sorted(glob.glob(f"{CHUNKED_DIR}/*.pt"))
    natural_norms = []
    attempts = 0
    while len(natural_norms) < N_NATURAL_SAMPLES and attempts < N_NATURAL_SAMPLES * 3:
        attempts += 1
        path = random.choice(all_paths)
        data = torch.load(path, weights_only=False)
        chain = random.choice(data["chains"])
        if len(chain["chunks"]) < 1:
            continue
        chunk = random.choice(chain["chunks"])
        residual = chunk[RESIDUAL_KEY].numpy()
        natural_norms.append(np.linalg.norm(residual))

    print("=== Vector norm comparison at layer 16 ===\n")
    summarize("Injected (de-standardized)", injected_norms)
    summarize("Natural (real generation)", natural_norms)

    ratio = np.mean(injected_norms) / np.mean(natural_norms)
    print(f"\nRatio of means (injected / natural) = {ratio:.3f}")
    if ratio < 0.7:
        print("-> Injected vectors are NOTABLY SMALLER than natural residuals. "
              "This could explain the generic-punctuation collapse: the "
              "injected signal may be too weak relative to what the network "
              "'expects' at this layer, getting effectively washed out by "
              "later layers rather than treated as meaningful content.")
    elif ratio > 1.4:
        print("-> Injected vectors are NOTABLY LARGER than natural residuals. "
              "This could push the network out-of-distribution, causing "
              "unstable/generic behavior rather than coherent steering.")
    else:
        print("-> Injected and natural norms are roughly comparable in scale. "
              "The punctuation-collapse pattern is likely NOT a simple "
              "magnitude mismatch -- more likely the direction itself (with "
              "the token-projection component removed) genuinely carries "
              "mostly structural/boundary information rather than content.")

    # Also break down what's contributing to injected norm: how much is
    # just the mean vector vs. the corrected*std component.
    mean_norm = np.linalg.norm(mean)
    print(f"\n||mean vector|| alone = {mean_norm:.2f}  "
          f"(for reference -- if this is close to the injected norms above, "
          f"the mean-centering term is dominating the injected vector, and "
          f"the corrected*std 'signal' component is comparatively small)")


if __name__ == "__main__":
    main()