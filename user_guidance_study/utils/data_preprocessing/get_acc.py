import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Get accuracy of trained models")

parser.add_argument('--results', type=str, help='Path to results.')

args = parser.parse_args()

res_dir = args.results

model_dirs = sorted(os.listdir(res_dir))

res = []

for mod_dir in model_dirs:
    ckpts = sorted(os.listdir(os.path.join(res_dir, mod_dir)))
    metric = [el for el in ckpts if 'metric' in el]
    final = [el for el in ckpts if 'final' in el]

    assert len(metric) == 1
    acc = float(metric[0].split('=')[1][:-3])
    res.append((mod_dir, acc))
with open('results/res.txt', 'w') as f:
    for r in res:
        f.write(f'{r[0]} : {r[1]}\n')
