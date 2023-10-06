import os
import numpy as np
import nibabel as nib
from functools import partial
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool
from dotenv import load_dotenv

load_dotenv()


IMAGE_SIZE = 256


train_dataset_config_1 = {
    'src_dir_path': os.environ.get("TRAIN_HGG_SRC_PATH"),
    'dst_dir_path': os.environ.get("TRAIN_BRATS_DST_PATH"),
    'modalities': [
        {'name': 'T1', 'pattern': 't1', 'save_pattern': 't1'},
        {'name': 'T1CE', 'pattern': 't1ce', 'save_pattern': 't1ce'},
        {'name': 'T2', 'pattern': 't2', 'save_pattern': 't2'},
        {'name': 'FLAIR', 'pattern': 'flair', 'save_pattern': 'flair'},
        {'name': 'SEG', 'pattern': 'seg', 'save_pattern': 'seg'},
    ],
}


train_dataset_config_2 = {
    'src_dir_path': os.environ.get("TRAIN_LGG_SRC_PATH"),
    'dst_dir_path': os.environ.get("TRAIN_BRATS_DST_PATH"),
    'modalities': [
        {'name': 'T1', 'pattern': 't1', 'save_pattern': 't1'},
        {'name': 'T1CE', 'pattern': 't1ce', 'save_pattern': 't1ce'},
        {'name': 'T2', 'pattern': 't2', 'save_pattern': 't2'},
        {'name': 'FLAIR', 'pattern': 'flair', 'save_pattern': 'flair'},
        {'name': 'SEG', 'pattern': 'seg', 'save_pattern': 'seg'},
    ],
}


def z_score_normalize(array):
    array = array.astype(np.float32)
    mask = array > 0
    mean = np.mean(array[mask])
    std = np.std(array[mask])
    array -= mean
    array /= std
    return array


def preprocess(patient_id, config):
    print(patient_id)

    patient_dir_path = os.path.join(
        config['src_dir_path'], patient_id,
    )

    for modality in config['modalities']:
        file_path = os.path.join(
            patient_dir_path,
            patient_id + '_' + modality['pattern'] + '.nii.gz'
        )
        nii_file = nib.load(file_path)
        series = nii_file.get_data()

        if modality['name'] == 'SEG':
            series = series.astype(np.int32)
            bincount = np.bincount(series.ravel())

            if 'Training' in config['src_dir_path'] and modality['pattern'] == 'seg':
                if len(bincount) > 3:
                    assert bincount[3] == 0

                series[series == 4] = 3  # 3: ET (GD-enhancing tumor)
                series[series == 2] = 2  # 2: ED (peritumoral edema)
                # 1: NCR/NET (non-enhancing tumor core)
                series[series == 1] = 1
                series[series == 0] = 0  # 0: Background

        else:
            series = z_score_normalize(series)

        for i in range(series.shape[2]):
            slice = series[..., i]
            slice = np.rot90(slice, k=3)

            if modality['name'] == 'SEG':
                slice = np.array(Image.fromarray(slice).resize(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    resample=Image.NEAREST,
                ))

            else:
                slice = np.array(Image.fromarray(slice).resize(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    resample=Image.BILINEAR,
                ))

            dst_patient_dir_path = os.path.join(
                config['dst_dir_path'], patient_id
            )
            os.makedirs(dst_patient_dir_path, exist_ok=True)

            save_path = os.path.join(
                dst_patient_dir_path,
                patient_id + '_' +
                modality['save_pattern'] + '_' + str(i).zfill(4) + '.npy'
            )

            np.save(save_path, slice)


if __name__ == '__main__':

    for config in [train_dataset_config_1, train_dataset_config_2]:

        _preprocess = partial(preprocess, config=config)

        patient_ids = os.listdir(config['src_dir_path'])

        p = Pool(32)
        p.map(_preprocess, patient_ids)
