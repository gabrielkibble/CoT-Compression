import pickle
with open("consensus_routes.pkl", "rb") as f:
    routes = pickle.load(f)

for pid in [0, 6, 7]:  # 0=healthy, 6=weak (only 2 correct chains), 7=longest
    route = routes[pid]
    print(f"=== Problem {pid} (medoid chain {route['chain_idx']}, "
          f"n_correct={route['n_correct_chains']}) ===")
    for i, c in enumerate(route["chunks"]):
        print(f"  {i:2d}: {c['text']!r}")
    print()