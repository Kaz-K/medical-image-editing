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


class NCCLungDataset(data.Dataset):

    def __init__(self,
                 root_dir_path: str,
                 transform: Optional['torchvision.transforms.Compose'] = None,
                 window_width: Optional[float] = None,
                 window_center: Optional[float] = None,
                 window_scale: Optional[float] = None,
                 ):
        super().__init__()

        self.root_dir_path = pathlib.Path(root_dir_path)
        self.transform = transform

        self.window_width = window_width
        self.window_center = window_center
        self.window_scale = window_scale

        self.files = self.build_file_paths()

        random.shuffle(self.files)

    def parse_slice_num(self, path):
        return int(os.path.splitext(os.path.basename(path))[0].split('_')[-1])

    def build_file_paths(self):
        files = []

        for patient_id in os.listdir(self.root_dir_path):
            patient_dir_path = self.root_dir_path / patient_id

            image_file_paths = glob.glob(str(patient_dir_path / '*_img_*'))

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

        if all((self.window_width is not None, self.window_center is not None, self.window_scale is not None)):
            image = normalize(image,
                              width=self.window_width,
                              center=self.window_center,
                              scale=self.window_scale)

        sample.update({'image': image})

        if self.transform:
            sample = self.transform(sample)

        return sample
