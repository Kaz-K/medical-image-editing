from typing import Optional
import os
import pathlib
import random
import glob
import numpy as np
from tqdm import tqdm
from torch.utils import data
import nibabel as nib
import pandas as pd
import scipy.misc
import matplotlib.pyplot as plt

from utils import normalize


class CRCDataset(data.Dataset):

    def __init__(self,
                 root_dir_path: str,
                 transform: Optional['torchvision.transforms.Compose'] = None,
                 ):
        super().__init__()

        self.root_dir_path = pathlib.Path(root_dir_path)
        self.transform = transform

        self.files = self.build_file_paths()

        random.shuffle(self.files)

    def parse_slice_num(self, path):
        return int(os.path.splitext(os.path.basename(path))[0])

    def build_file_paths(self):
        files = []

        for patient_id in os.listdir(self.root_dir_path):
            patient_dir_path = self.root_dir_path / patient_id

            image_file_paths = glob.glob(str(patient_dir_path / '*.npy'))

            for image_file_path in sorted(image_file_paths):
                slice_num = self.parse_slice_num(image_file_path)

                files.append({
                    'patient_id': patient_id,
                    'slice_num': slice_num,
                    'image_path': image_file_path,
                })

        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = self.files[index]

        image = np.load(sample['image_path']).astype(np.float32)

        sample.update({'image': image})

        if self.transform:
            sample = self.transform(sample)

        return sample
