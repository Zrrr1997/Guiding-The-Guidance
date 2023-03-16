import nibabel as nib
import numpy as np
import numpy as np
import argparse
import os
from tqdm import tqdm
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description="Determing the voxel intensity normalization for a dataset")

parser.add_argument('--dataset', type=str, help='Path to dataset.')
parser.add_argument('--load_cache', default=False, action='store_true', help='Load computed values from cache.')


args = parser.parse_args()

d_src = args.dataset

if not args.load_cache:
    intensities = []
    intensities_aoi = [] # intensities of anatomy of interest
    all_files = [el for el in sorted(os.listdir(os.path.join(d_src, 'imagesTr'))) if el[0] != '.']
    all_files = all_files[200:]
    for i, fn in enumerate(tqdm(all_files)):

        if i == 10:
            continue
        im = nib.load(os.path.join(d_src, 'imagesTr', fn)).get_fdata()
        label = nib.load(os.path.join(d_src, 'labelsTr', fn)).get_fdata()
        aoi = im * label

        im = im.flatten()
        aoi = aoi.flatten()

        intensities += list(im[im != 0])
        intensities_aoi += list(aoi[aoi != 0])


    intensities = np.array(intensities)
    intensities_aoi = np.array(intensities_aoi)


    np.save(os.path.join(d_src, 'all_HU_2.npy'), intensities)
    np.save(os.path.join(d_src, 'aoi_HU_2.npy'), intensities_aoi)
else:
    print('Loading all HUs')
    intensities = np.load(os.path.join(d_src, 'all_HU.npy'))
    print('Loading aoi HUs')
    intensities_aoi = np.load(os.path.join(d_src, 'aoi_HU.npy'))

print(np.mean(intensities_aoi), np.mean(intensities))
#plt.hist(intensities, label='before')
plt.hist(intensities_aoi, label='aoi only')
print(np.min(intensities_aoi), np.max(intensities_aoi))
print(np.percentile(intensities_aoi, 0.05), np.percentile(intensities_aoi, 99.95))

print(len(intensities_aoi[intensities_aoi > 250]))
print(len(intensities_aoi))

plt.legend()
plt.savefig(os.path.join(d_src, 'aoi.png'))
