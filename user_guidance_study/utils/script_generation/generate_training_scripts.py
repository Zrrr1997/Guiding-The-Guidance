import numpy as np

def inp_enc(x):
    if x == 'conv1d':
        return ' --conv1d true'
    elif x == 'conv1s':
        return ' --conv1s true'
    else:
        return ' '
name = 'name'
base_script = f'python train_autopet.py --output output_autopet/name/ --cache_dir cache_autopet/name/ --sigma sigma_val'

# Fixed Parameters
scripts = []
p = 1.0
enc = 'cat'
for sigma in [0, 1, 5, 9, 13]:
    script_1 = base_script.replace('sigma_val', str(sigma))
    # Heatmaps
    name = f'heatmaps_sigma={sigma}_p={p}_enc={enc}'
    script_heatmaps = script_1.replace('name', name)

    script_exp_geos = script_1 + ' --exp_geos true --disks true'
    name = f'exp_geos_sigma={sigma}_p={p}_enc={enc}'
    script_exp_geos = script_exp_geos.replace('name', name)

    script_disks = script_1 + ' --disks true'
    name = f'disks_sigma={sigma}_p={p}_enc={enc}'
    script_disks = script_disks.replace('name', name)


    scripts.append(script_heatmaps)
    scripts.append(script_exp_geos)
    scripts.append(script_disks)

    for theta in [0, 10, 30, 50]:
        script_gdt = script_1 + f' --gdt true --disks true --gdt_th {theta}'
        script_edt = script_1 + f' --edt true --disks true --gdt_th {theta}'

        name = f'edt_sigma={sigma}_p={p}_enc={enc}_gdt-theta={theta}'
        script_edt = script_edt.replace('name', name)

        name = f'gdt_sigma={sigma}_p={p}_enc={enc}_gdt-theta={theta}'
        script_gdt = script_gdt.replace('name', name)

        scripts.append(script_edt)
        scripts.append(script_gdt)
with open ('scripts_autopet.txt', 'w') as f:
    for e in scripts:
        f.write(e + '\n')
print('Generated', len(scripts), 'scripts...')
