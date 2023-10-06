import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


SRC_ROOT_DIR_PATH = os.environ.get("SRC_CRC_DIR_PATH")
DST_ROOT_DIR_PATH = os.environ.get("DST_CRC_DIR_PATH")
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
    image_files = glob.glob(os.path.join(
        SRC_ROOT_DIR_PATH, '*_image.nii.gz'
    ))

    for image_file in image_files:
        patient_id = parse_patient_id(image_file)
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
                DST_ROOT_DIR_PATH, patient_id
            )

            os.makedirs(save_dir_path, exist_ok=True)

            save_image_path = os.path.join(
                save_dir_path, str(i).zfill(4) + '.npy'
            )

            np.save(save_image_path, img)
