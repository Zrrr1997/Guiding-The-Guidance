import pandas as pd
import nibabel as nib
import os

df_cv = pd.read_csv('autopet_5folds_augmented.csv')
df_cv['study_location'] = df_cv['study_location'].str.replace('/projects/datashare/tio/autopet/FDG-PET-CT-Lesions/', '/home/haicore-project-kit-iar-cvhci/zk6393/my_workspace/autoPET/FDG-PET-CT-Lesions/')

print(len(df_cv))

df_cv = df_cv[df_cv['diagnosis'] != 'NEGATIVE']

df_cv_train = df_cv[df_cv['kfold'] != 0]
df_cv_test = df_cv[df_cv['kfold'] == 0]


print(len(df_cv_train), len(df_cv_test))


train_files = df_cv_train['study_location'].to_numpy()
test_files = df_cv_test['study_location'].to_numpy()

print(len(train_files), len(test_files))

fn = train_files[0]

with open('train_files_CT.txt', 'w') as f:
    for fn in train_files:
        f.write(fn + 'CTres.nii.gz\n')
with open('test_files_CT.txt', 'w') as f:
    for fn in test_files:
        f.write(fn + 'CTres.nii.gz\n')

for i, fn in enumerate(train_files):
    print(i)
    os.system(f'cp "{fn}CTres.nii.gz" "AutoPET/imagesTr/"')
    os.system(f'mv "AutoPET/imagesTr/CTres.nii.gz" "AutoPET/imagesTr_CT/tumor_{i}.nii.gz"')

    #os.system(f'cp "{fn}SEG.nii.gz" "AutoPET/labelsTr/"')
    #os.system(f'mv "AutoPET/labelsTr/SEG.nii.gz" "AutoPET/labelsTr/tumor_{i}.nii.gz"')
'''
for i, fn in enumerate(test_files):
    print(i)
    os.system(f'cp "{fn}SUV.nii.gz" "AutoPET/imagesTs/"')
    os.system(f'mv "AutoPET/imagesTs/SUV.nii.gz" "AutoPET/imagesTs/tumor_{i}.nii.gz"')

    os.system(f'cp "{fn}SEG.nii.gz" "AutoPET/labelsTs/"')
    os.system(f'mv "AutoPET/labelsTs/SEG.nii.gz" "AutoPET/labelsTs/tumor_{i}.nii.gz"')
'''
