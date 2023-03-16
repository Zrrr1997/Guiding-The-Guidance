import numpy as np
import SimpleITK as sitk
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Determing the spacing for a dataset")

parser.add_argument('--dataset', type=str, help='Path to dataset.')

args = parser.parse_args()

d_src = args.dataset

spacings = []
sizes = []

for fn in tqdm([el for el in os.listdir(d_src) if el[0] != '.']):
    im = sitk.ReadImage(os.path.join(d_src, fn))
    spacing = np.array(im.GetSpacing())
    size = np.array(im.GetSize())
    spacings.append(spacing)
    sizes.append(size)

spacings = np.array(spacings)
sizes = np.array(sizes)
print('Mean:', np.mean(spacings, keepdims=True, axis=0))
print('Median:', np.median(spacings, keepdims=True, axis=0))
print('Std:', np.std(spacings, keepdims=True, axis=0))
print('Min:', np.min(spacings, keepdims=True, axis=0))
print('Max:', np.max(spacings, keepdims=True, axis=0))

np.save(os.path.join(os.path.dirname(os.path.dirname(d_src)), 'spacings.npy'), spacings)
np.save(os.path.join(os.path.dirname(os.path.dirname(d_src)), 'sizes.npy'), sizes)
