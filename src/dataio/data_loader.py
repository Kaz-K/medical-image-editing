from typing import Optional
from torch.utils import data
from torchvision import transforms

from .miccai_dataset import MICCAIBraTSDataset
from .lung_dataset import NCCLungDataset
from .crc_dataset import CRCDataset
from .transforms import ToTensor
from .transforms import SqueezeAxis
from .transforms import NormalizeIntensity
from .transforms import RandomAffineTransform
from .transforms import RandomHorizontalFlipTransform


def get_data_loader(mode: str,
                    dataset_name: str,
                    root_dir_path: str,
                    batch_size: int,
                    num_workers: int,
                    modality: Optional[str] = None,
                    augmentations: Optional[list] = None,
                    drop_last: bool = False,
                    window_width: Optional[float] = None,
                    window_center: Optional[float] = None,
                    window_scale: Optional[float] = None,
                    ):

    assert mode in {'train', 'val', 'test'}
    assert dataset_name in {'MICCAIBraTSDataset', 'NCCLungDataset', 'CRCDataset'}

    if dataset_name == 'MICCAIBraTSDataset':
        if mode == 'train':
            transform_list = [ToTensor()]

            if 'RandomAffineTransform' in augmentations:
                transform_list.append(
                    RandomAffineTransform(p=0.5, degrees=(-5, 5), translate=(0.01, 0.01), scale=(0.9, 1.1), resample='BILINEAR')
                )

            if 'RandomHorizontalFlipTransform' in augmentations:
                transform_list.append(
                    RandomHorizontalFlipTransform(p=0.5),
                )

            transform_list.extend([NormalizeIntensity(), SqueezeAxis()])
            shuffle = True

        elif mode == 'val':
            assert augmentations is None

            transform_list = [ToTensor(), NormalizeIntensity()]
            shuffle = True

        elif mode == 'test':
            assert augmentations is None

            transform_list = [ToTensor(), NormalizeIntensity()]
            shuffle = False

        transform = transforms.Compose(transform_list)

        dataset = MICCAIBraTSDataset(
            root_dir_path=root_dir_path,
            modality=modality,
            transform=transform,
        )

    elif dataset_name == 'NCCLungDataset':
        if mode == 'train':
            transform_list = [ToTensor()]

            if 'RandomAffineTransform' in augmentations:
                transform_list.append(
                    RandomAffineTransform(p=0.5, degrees=(-5, 5), translate=(0.01, 0.01), scale=(0.9, 1.1), resample='BILINEAR')
                )

            if 'RandomHorizontalFlipTransform' in augmentations:
                transform_list.append(
                    RandomHorizontalFlipTransform(p=0.5),
                )

            transform_list.extend([SqueezeAxis()])
            shuffle = True

        elif mode == 'val':
            assert augmentations is None

            transform_list = [ToTensor()]
            shuffle = True

        elif mode == 'test':
            assert augmentations is None

            transform_list = [ToTensor()]
            shuffle = False

        transform = transforms.Compose(transform_list)

        dataset = NCCLungDataset(
            root_dir_path=root_dir_path,
            transform=transform,
            window_width=window_width,
            window_center=window_center,
            window_scale=window_scale,
        )

    elif dataset_name == 'CRCDataset':
        if mode == 'train':
            transform_list = [ToTensor()]

            if 'RandomAffineTransform' in augmentations:
                transform_list.append(
                    RandomAffineTransform(p=0.5, degrees=(-5, 5), translate=(0.01, 0.01), scale=(0.9, 1.1), resample='BILINEAR')
                )

            if 'RandomHorizontalFlipTransform' in augmentations:
                transform_list.append(
                    RandomHorizontalFlipTransform(p=0.5),
                )

            transform_list.extend([NormalizeIntensity(), SqueezeAxis()])
            shuffle = True

        elif mode == 'val':
            assert augmentations is None

            transform_list = [ToTensor(), NormalizeIntensity()]
            shuffle = True

        elif mode == 'test':
            assert augmentations is None

            transform_list = [ToTensor(), NormalizeIntensity()]
            shuffle = False

        transform = transforms.Compose(transform_list)

        dataset = CRCDataset(
            root_dir_path=root_dir_path,
            transform=transform,
        )

    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
