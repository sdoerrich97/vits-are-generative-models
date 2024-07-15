"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Data loader class.

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
import cv2
import random
import albumentations as A
import torch
import numpy

from pathlib import Path
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

# Import own packages
from dataset import Data, UnlabeledDataset, LabeledDataset, MetaLabeledDataset


def seed_worker(worker_id):
    """Set the seed for the current worker."""
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


class ParentLoader:
    """ Parent class. """
    def __init__(self, dataset: str, root_dir: Path, img_size: int, seed: int):
        """
        :param dataset: Dataset to load.
        :param root_dir: Root directory of the dataset.
        :param img_size: Image size.
        :param seed: Seed to use.
        """

        # Store the parameters
        self.dataset = dataset
        self.root_dir = root_dir
        self.img_size = img_size

        if dataset == 'camelyon17' or dataset == 'camelyon17-unlabeled':
            min, max, mean, std = 0.0, 255.0, (0.7441, 0.5895, 0.7214), (0.1301, 0.1489, 0.1107)

        elif dataset == 'rxrx1':
            min, max, mean, std = 0.0, 255.0, (0.0268, 0.0586, 0.0419), (0.0308, 0.0473, 0.0220)

        elif dataset == 'isic2018':
            min, max, mean, std = 0.0, 255.0, (0.7615, 0.5463, 0.5711), (0.0900, 0.1188, 0.1335)

        elif dataset == 'epistr-NKI-VGH':
            min, max, mean, std = 0.0, 255.0, (0.7385, 0.5180, 0.8082), (0.1674, 0.2110, 0.1326)

        elif dataset == 'epistr-VGH-NKI':
            min, max, mean, std = 0.0, 255.0, (0.7935, 0.5130, 0.8761), (0.1499, 0.1873, 0.1045)

        else:
            raise ValueError('Dataset not supported.')

        # Set the seed for the data loading when using multiple workers
        self.g = torch.Generator()
        self.g.manual_seed(seed)

        self.transform_geom = A.Compose([
            # Geometrical Transforms (affects anatomy)
            A.ElasticTransform(p=0.1, alpha=1, sigma=50, alpha_affine=50, interpolation=cv2.INTER_CUBIC),
            A.GridDistortion(p=0.1, interpolation=cv2.INTER_CUBIC),
            A.Flip(p=0.1),
            A.Rotate(p=0.1),
        ])

        self.transform_color = A.Compose([
            # Color Transforms (affects characteristics)
            A.OneOf([
                A.Blur(p=1.0),
                A.Defocus(p=1.0),
                A.GaussianBlur(p=1.0),
                A.GlassBlur(p=1.0),
                A.MedianBlur(p=1.0),
                A.MotionBlur(p=1.0),
            ], p=0.1),

            A.OneOf([
                A.GaussNoise(p=1.0),
                A.ISONoise(p=1.0),
                A.MultiplicativeNoise(p=1.0),
                A.PixelDropout(p=0.1)
            ], p=0.1),

            A.OneOf([
                A.Downscale(scale_min=0.5, scale_max=0.9, p=0.1, interpolation=cv2.INTER_CUBIC),
                A.ImageCompression(p=0.1),
                A.Sharpen(p=0.1),
                A.UnsharpMask(p=0.1),
                A.Posterize(p=0.1)
            ], p=0.1),

            A.OneOf([
                A.InvertImg(p=0.1),
                A.Solarize(threshold=128, p=0.1),
                A.HueSaturationValue(p=0.1),
                A.ColorJitter(p=0.1),
                A.ColorJitter(p=0.1, brightness=0, contrast=0, saturation=0),  # Augment for hue
                A.RGBShift(p=0.1)
            ], p=0.1),

            A.OneOf([
                A.RandomGamma(p=0.1),
                A.RandomBrightnessContrast(p=0.1),
                A.ColorJitter(p=0.1, contrast=0, saturation=0, hue=0),  # Augment for brightness
                A.ColorJitter(p=0.1, brightness=0, saturation=0, hue=0),  # Augment for contrast
                A.ColorJitter(p=0.1, brightness=0, contrast=0, hue=0),  # Augment for saturation
                A.Equalize(p=0.1),
                A.FancyPCA(p=0.1),
            ], p=0.1),
        ])

        self.transform_pre_pretrain = A.Compose([
            A.RandomResizedCrop(height=self.img_size, width=self.img_size, interpolation=cv2.INTER_CUBIC, p=1.0),
        ])

        self.transform_pre_pretrain_prime = A.Compose([
            A.Resize(height=256, width=256, interpolation=cv2.INTER_CUBIC, p=1.0),  # Resize
            A.CenterCrop(height=self.img_size, width=self.img_size, p=1.0),  # Center Crop
        ])

        self.transform_pre_train = A.Compose([
            A.Resize(height=self.img_size, width=self.img_size, interpolation=cv2.INTER_CUBIC, p=1.0),  # Resize
        ])

        self.transform_post = A.Compose([
            A.Normalize(mean=mean, std=std, max_pixel_value=max),
            ToTensorV2()
            ])

        self.pretrain_transform = A.Compose([
            self.transform_pre_pretrain,  # Resizing and Cropping
            self.transform_geom,  # Geometrical Transforms (affects anatomy)
            self.transform_color,  # Color Transforms (affects characteristics)
            self.transform_post  # Normalize and ToTensor
        ])

        self.pretrain_transform_prime = A.Compose([
            self.transform_pre_pretrain_prime,  # Resizing and Cropping
            self.transform_post  # Normalize and ToTensor
        ])

        self.train_transform = A.Compose([
            self.transform_pre_train,  # Resizing and Cropping
            self.transform_geom,  # Geometrical Transforms (affects anatomy)
            self.transform_color,  # Color Transforms (affects characteristics)
            self.transform_post  # Normalize and ToTensor
        ])

        self.train_transform_prime = A.Compose([
            self.transform_pre_train,  # Resizing and Cropping
            self.transform_post  # Normalize and ToTensor
        ])

        # Load the data
        self.data = Data(self.dataset, self.root_dir)


class DatasetLoaderPretrain(ParentLoader):
    """ Data Loader. """
    def __init__(self, dataset: str, root_dir: Path, img_size: int, batch_size: int, seed: int, train=True):
        """
        :param dataset: Dataset to load.
        :param root_dir: Root directory of the dataset.
        :param img_size: Image size.
        :param batch_size: Batch size.
        """

        # Initialize the parent class
        super().__init__(dataset, root_dir, img_size, seed)

        # Get the data
        train_data = self.data.get_train_data()
        val_data = self.data.get_val_data()
        test_data = self.data.get_test_data()

        # Create the dataloaders
        if train:
            train_set = UnlabeledDataset(train_data, self.pretrain_transform)
            self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=self.g)
        else:
            train_set = UnlabeledDataset(train_data, self.pretrain_transform_prime)
            self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=self.g)

        val_set = UnlabeledDataset(val_data, self.pretrain_transform_prime)
        test_set = UnlabeledDataset(test_data, self.pretrain_transform_prime)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=self.g)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=self.g)

    def get_train_loader(self):
        """Get the train loader."""

        return self.train_loader

    def get_val_loader(self):
        """Get the validation loader."""

        return self.val_loader

    def get_test_loader(self):
        """Get the test loader."""

        return self.test_loader


class DatasetLoaderTrain(ParentLoader):
    """ Data Loader. """
    def __init__(self, dataset: str, root_dir: Path, img_size: int, batch_size: int, seed: int, train=True):
        """
        :param dataset: Dataset to load.
        :param root_dir: Root directory of the dataset.
        :param img_size: Image size.
        :param batch_size: Batch size.
        """

        # Initialize the parent class but do not load the unlabeled samples
        if dataset == 'camelyon17-unlabeled':
            dataset = 'camelyon17'

        super().__init__(dataset, root_dir, img_size, seed)

        # Get the data
        train_data = self.data.get_train_data()
        val_data = self.data.get_val_data()
        test_data = self.data.get_test_data()

        # Create the dataloaders
        if train:
            train_set = LabeledDataset(train_data, self.train_transform)
            self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=self.g)
        else:
            train_set = LabeledDataset(train_data, self.train_transform_prime)
            self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=self.g)

        val_set = LabeledDataset(val_data, self.train_transform_prime)
        test_set = LabeledDataset(test_data, self.train_transform_prime)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=self.g)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=self.g)

    def get_train_loader(self):
        """Get the train loader."""

        return self.train_loader

    def get_val_loader(self):
        """Get the validation loader."""

        return self.val_loader

    def get_test_loader(self):
        """Get the test loader."""

        return self.test_loader


class DatasetLoaderExperiment(ParentLoader):
    """ Data Loader. """
    def __init__(self, dataset: str, root_dir: Path, img_size: int, batch_size: int, seed: int):
        """
        :param dataset: Dataset to load.
        :param root_dir: Root directory of the dataset.
        :param img_size: Image size.
        :param batch_size: Batch size.
        """

        # Initialize the parent class but do not load the unlabeled samples
        if dataset == 'camelyon17-unlabeled':
            dataset = 'camelyon17'

        super().__init__(dataset, root_dir, img_size, seed)

        # Get the data
        train_data = self.data.get_train_data()
        val_data = self.data.get_val_data()
        test_data = self.data.get_test_data()

        # Create the dataloaders
        train_set = MetaLabeledDataset(train_data, self.train_transform_prime)
        val_set = MetaLabeledDataset(val_data, self.train_transform_prime)
        test_set = MetaLabeledDataset(test_data, self.train_transform_prime)

        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=self.g)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=self.g)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=self.g)

    def get_train_loader(self):
        """Get the train loader."""

        return self.train_loader

    def get_val_loader(self):
        """Get the validation loader."""

        return self.val_loader

    def get_test_loader(self):
        """Get the test loader."""

        return self.test_loader