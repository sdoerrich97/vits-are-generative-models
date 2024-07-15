"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Custom dataset classes for the training and evaluation of our model.

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
import numpy as np
import torch
import albumentations as A
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, Subset
from wilds import get_dataset


class EpistrDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        label = self.labels[idx]
        return image, label


class Data:
    def __init__(self, dataset: str, root_dir: Path):
        """
        Custom Data class.

        :param dataset: Which dataset to load.
        :param root_dir: Where to load the dataset from.
        """

        # Load the data and split in train, val, test
        if 'camelyon17' in dataset:
            if 'camelyon17' == dataset:
                data = get_dataset(dataset="camelyon17", root_dir=root_dir, download=True)

                self.train_data = data.get_subset("train")
                self.val_data = data.get_subset("val")
                self.test_data = data.get_subset("test")

            elif 'camelyon17-unlabeled' == dataset:
                data_labeled = get_dataset(dataset="camelyon17", root_dir=root_dir, download=True)
                data_unlabeled = get_dataset(dataset="camelyon17", root_dir=root_dir, download=True, unlabeled=True)

                # Get the number of samples in the labeled dataset
                num_labeled_samples = len(data_labeled.get_subset("train"))

                # Create a new unlabeled dataset with the same number of samples
                indices = torch.randperm(len(data_unlabeled.get_subset("train_unlabeled")))[:num_labeled_samples]
                data_unlabeled_subset = Subset(data_unlabeled.get_subset("train_unlabeled"), indices)

                # Now concatenate the labeled and unlabeled datasets
                self.train_data = torch.utils.data.ConcatDataset([data_labeled.get_subset("train"), data_unlabeled_subset])
                self.val_data = data_labeled.get_subset("val")
                self.test_data = data_labeled.get_subset("test")

            else:
                raise ValueError('Dataset not supported.')

        elif 'epistr' in dataset:
            root_dir = Path(root_dir)
            datasets = ['IHC', 'NCH', 'NKI', 'VGH']
            image_paths = {name: [] for name in datasets}
            labels = {name: [] for name in datasets}

            for name in datasets:
                for class_name, label in [('epi', 0), ('str', 1)]:
                    for phase in ['train', 'test']:
                        new_image_paths = list((root_dir / name / phase / class_name).glob('*.jpg')) + list((root_dir / name / phase / class_name).glob('*.png'))
                        image_paths[name].extend(new_image_paths)
                        labels[name].extend([label] * len(new_image_paths))

            if dataset == 'epistr-NKI-VGH':
                self.train_data = EpistrDataset(image_paths['NKI'], labels['NKI'])
                self.val_data = EpistrDataset(image_paths['VGH'], labels['VGH'])

            elif dataset == 'epistr-VGH-NKI':
                self.train_data = EpistrDataset(image_paths['VGH'], labels['VGH'])
                self.val_data = EpistrDataset(image_paths['NKI'], labels['NKI'])

            self.test_data = EpistrDataset(image_paths['IHC'], labels['IHC'])

        else:
            raise ValueError('Dataset not supported.')

    def get_train_data(self):
        return self.train_data

    def get_val_data(self):
        return self.val_data

    def get_test_data(self):
        return self.test_data


class UnlabeledDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: A.Compose):
        """
        Initialize the Dataset for pretraining.

        :param dataset: The dataset to load.
        :param transform: Transforms.
        """

        # Store the parameters
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        """Return the number of samples in the dataset"""

        return len(self.dataset)

    def __getitem__(self, index: int):
        """Load and return a sample together with its synthetically distorted variant."""

        # Extract the sample
        if len(self.dataset[index]) == 1:
            img = self.dataset[index]
        elif len(self.dataset[index]) == 2:
            img, _ = self.dataset[index]
        elif len(self.dataset[index]) == 3:
            img, _, _ = self.dataset[index]
        else:
            raise ValueError('Dataset not supported.')

        # Load the sample as a numpy ndarray
        image = np.array(img)

        # Apply the transformations to the image
        sample = self.transform(image=image)['image']

        # Return the loaded sample
        return sample


class LabeledDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: A.Compose):
        """
        Initialize the Dataset for pretraining.

        :param dataset: The dataset to load.
        :param transform: Transforms.
        """

        # Store the parameters
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        """Return the number of samples in the dataset"""

        return len(self.dataset)

    def __getitem__(self, index: int):
        """Load and return a sample together with its synthetically distorted variant."""

        # Extract the sample
        if len(self.dataset[index]) == 2:
            img, label = self.dataset[index]
        elif len(self.dataset[index]) == 3:
            img, label, _ = self.dataset[index]
        else:
            raise ValueError('Dataset not supported.')

        # Load the sample as a numpy ndarray
        image = np.array(img)

        # Apply the transformations to the image
        sample = self.transform(image=image)['image']

        # Return the loaded sample
        return sample, label


class MetaLabeledDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: A.Compose):
        """
        Initialize the Dataset for pretraining.

        :param dataset: The dataset to load.
        :param transform: Transforms.
        """

        # Store the parameters
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        """Return the number of samples in the dataset"""

        return len(self.dataset)

    def __getitem__(self, index: int):
        """Load and return a sample together with its synthetically distorted variant."""

        # Extract the sample
        if len(self.dataset[index]) == 2:
            img, label = self.dataset[index]

            # Load the sample as a numpy ndarray
            image = np.array(img)

            # Apply the transformations to the image
            sample = self.transform(image=image)['image']

            # Return the loaded sample
            return sample, label

        elif len(self.dataset[index]) == 3:
            img, label, metadata = self.dataset[index]

            # Load the sample as a numpy ndarray
            image = np.array(img)

            # Apply the transformations to the image
            sample = self.transform(image=image)['image']

            # Return the loaded sample
            return sample, label, metadata
