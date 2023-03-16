import numpy as np


name = 'name'
base_script = f'python train_autopet_mirrorUNet.py --output output_Mirror_UNet/name/ --cache_dir cache_Mirror_UNet/name/ --guidance_exp experiment'
# Fixed Parameters
scripts = []

for exp in ['exp_1', 'exp_2', 'exp_3']:
    script_1 = base_script.replace('experiment', exp)

    if exp == 'exp_1':
        # PET sample probability
        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            script_exp_1 = script_1 + f' --PET_prob {p}'
            script_exp_1_disks = script_exp_1
            script_exp_1_disks = script_exp_1_disks.replace('name', f'{exp}_disks_p={p}')
            script_exp_1_geos_pet = script_exp_1 + ' --geos_image_seed pet --gdt true'
            script_exp_1_geos_pet = script_exp_1_geos_pet.replace('name', f'{exp}_geos_pet_p={p}')

            script_exp_1_geos_ct = script_exp_1 + ' --geos_image_seed ct --gdt true'
            script_exp_1_geos_ct = script_exp_1_geos_ct.replace('name', f'{exp}_geos_ct_p={p}')

            scripts.append(script_exp_1_disks)
            scripts.append(script_exp_1_geos_pet)
            scripts.append(script_exp_1_geos_ct)
    elif exp == 'exp_2':
        script_exp_2_disks = script_1
        script_exp_2_disks = script_exp_2_disks.replace('name', f'{exp}_disks')
        script_exp_2_geos_pet = script_1 + ' --geos_image_seed pet --gdt true'
        script_exp_2_geos_pet = script_exp_2_geos_pet.replace('name', f'{exp}_geos_pet')

        script_exp_2_geos_ct = script_1 + ' --geos_image_seed ct --gdt true'
        script_exp_2_geos_ct = script_exp_2_geos_ct.replace('name', f'{exp}_geos_ct')

        scripts.append(script_exp_2_disks)
        scripts.append(script_exp_2_geos_pet)
        scripts.append(script_exp_2_geos_ct)
    elif exp == 'exp_3':
        script_exp_3_disks_disks = script_1.replace('name', f'{exp}_disks_disks')
        script_exp_3_disks_geos = script_1.replace('name', f'{exp}_disks_geos')
        script_exp_3_disks_geos = script_exp_3_disks_geos + ' --gdt_pet true'

        script_exp_3_geos_disks = script_1.replace('name', f'{exp}_geos_disks')
        script_exp_3_geos_disks = script_exp_3_geos_disks + ' --gdt true'

        script_exp_3_geos_geos = script_1.replace('name', f'{exp}_geos_geos')
        script_exp_3_geos_geos = script_exp_3_geos_geos + ' --gdt true --gdt_pet true'

        scripts.append(script_exp_3_disks_disks)
        scripts.append(script_exp_3_disks_geos)
        scripts.append(script_exp_3_geos_disks)
        scripts.append(script_exp_3_geos_geos)

with open ('scripts_Mirror_UNet.txt', 'w') as f:
    for e in scripts:
        f.write(e + '\n')
print('Generated', len(scripts), 'scripts...')
