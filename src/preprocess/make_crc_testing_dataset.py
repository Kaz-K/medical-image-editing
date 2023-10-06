import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv


load_dotenv()
train_data_dir_path = os.environ.get("TRAIN_DATA_DIR_PATH")
candidate_dir_path = os.environ.get("CANDIDATE_DIR_PATH")
dist_dir_path = os.environ.get("DIST_DIR_PATH")


IMAGE_SIZE = 512


def parse_patient_id(file_path):
    basename = os.path.basename(file_path).split('_')[:2]
    patient_id = '_'.join(basename)
    return patient_id


def minmax_normalize(image, scale=255.0):
    a_min = image.min()
    a_max = image.max()
    image -= a_min
    image /= (a_max - a_min)
    image *= scale
    return image


if __name__ == '__main__':
    training_patients = os.listdir(train_data_dir_path)

    assert len(training_patients) == 289

    image_files = glob.glob(os.path.join(
        candidate_dir_path, '*_image.nii.gz'
    ))

    for image_file in image_files:
        patient_id = parse_patient_id(image_file)

        if patient_id not in training_patients:
            image = nib.load(image_file).get_fdata()
            image = minmax_normalize(image)

            for i in range(image.shape[2]):
                img = image[..., i]
                img = img[::-1, ...]
                img = np.rot90(img)

                img = np.array(Image.fromarray(img).resize(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    resample=Image.BILINEAR,
                ))

                save_dir_path = os.path.join(
                    dist_dir_path, patient_id
                )

                os.makedirs(save_dir_path, exist_ok=True)

                save_image_path = os.path.join(
                    save_dir_path, str(i).zfill(4) + '.npy'
                )

                np.save(save_image_path, img)
