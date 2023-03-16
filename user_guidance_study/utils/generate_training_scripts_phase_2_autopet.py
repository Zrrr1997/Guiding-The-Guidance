import numpy as np

def inp_enc(x):
    if x == 'conv1d':
        return ' --conv1d true'
    elif x == 'conv1s':
        return ' --conv1s true'
    else:
        return ' '
name = 'name'
base_script = f'python train_autopet.py --output output_autopet/name/ --cache_dir cache_autopet/name/ --sigma SIGMA_VAL xxx'

scripts = []
# Fixed parameters
sigma = 1
theta = 0.0

for p in [0.5, 0.75, 1.0]:
    for enc in ['concat', 'conv1s', 'conv1d']:
        # Heatmaps
        name = f'heatmaps_sigma={sigma}_p={p}_enc={enc}'
        script_1 = base_script.replace('xxx', inp_enc(enc))

        script_heatmaps = script_1.replace('name', name)
        script_heatmaps = script_heatmaps.replace('SIGMA_VAL', '0')



        # Exponentialized Geodesic Distance
        script_exp_geos = script_1 + ' --exp_geos true --disks true'
        name = f'exp_geos_sigma={sigma}_p={p}_enc={enc}'
        script_exp_geos = script_exp_geos.replace('name', name)
        script_exp_geos = script_exp_geos.replace('SIGMA_VAL', '5')


        script_disks = script_1 + ' --disks true'
        name = f'disks_sigma={sigma}_p={p}_enc={enc}'
        script_disks = script_disks.replace('name', name)
        script_disks = script_disks.replace('SIGMA_VAL', '0')



        scripts.append(script_heatmaps)
        scripts.append(script_exp_geos)
        scripts.append(script_disks)

        script_gdt = script_1 + f' --gdt true --disks true --gdt_th 10'
        script_edt = script_1 + f' --edt true --disks true --gdt_th 50'

        name = f'edt_sigma={sigma}_p={p}_enc={enc}_gdt-theta=50'
        script_edt = script_edt.replace('name', name)
        script_edt = script_edt.replace('SIGMA_VAL', '1')


        name = f'gdt_sigma={sigma}_p={p}_enc={enc}_gdt-theta=10'
        script_gdt = script_gdt.replace('name', name)
        script_gdt = script_gdt.replace('SIGMA_VAL', '5')


        scripts.append(script_edt)
        scripts.append(script_gdt)
with open ('scripts_phase_2_autopet.txt', 'w') as f:
    for e in sorted(scripts):
        f.write(e + '\n')
print('Generated', len(scripts), 'scripts...')
