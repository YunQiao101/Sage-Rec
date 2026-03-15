import json
import os
import numpy as np

DATASET = "Instruments"
RESULTS_DIR = f"./results/{DATASET}/"
SEEDS = [42,101 ,2026 ,8891]

all_results = []
seed_files = []

for seed in SEEDS:
    result_file = os.path.join(RESULTS_DIR, f"tiger_seed{seed}.json")
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            data = json.load(f)
            all_results.append(data['results'])
            seed_files.append(result_file)
    else:
        print(f"Warning: {result_file} not found, skipping...")

if not all_results:
    print("No result files found!")
    exit(1)

# 统计 min, max, mean, std
metrics = list(all_results[0].keys())
aggregated = {}

for metric in metrics:
    values = [res[metric] for res in all_results]
    aggregated[metric] = {
        'mean': float(np.mean(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'std': float(np.std(values))
    }

output_file = os.path.join(RESULTS_DIR, f"tiger_aggregated_{len(all_results)}seeds.json")
with open(output_file, 'w') as f:
    json.dump({
        'seeds': SEEDS,
        'individual_results': all_results,
        'aggregated': aggregated
    }, f, indent=4)

print("=" * 50)
print("Aggregated Results:")
print("=" * 50)
for metric, values in aggregated.items():
    print(f"{metric}:")
    print(f"  Mean: {values['mean']:.4f}")
    print(f"  Min:  {values['min']:.4f}")
    print(f"  Max:  {values['max']:.4f}")
    print(f"  Std:  {values['std']:.4f}")
print("=" * 50)
print(f"\nSaved to: {output_file}")
