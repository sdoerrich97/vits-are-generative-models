"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Calculate the statistics of the dataset(s).

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

import argparse
import albumentations as A
import cv2
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

from dataset import Data, UnlabeledDataset


def calculate_mean_std(dataloader):
    """
    Calculate the mean and standard deviation of a dataset.
    :param dataloader: Dataloader.
    :return: Min, Max, Mean and standard deviation of the dataset.
    """

    # Iterate through the dataset to calculate mean and standard deviation
    max_pixel, min_pixel = float('-inf'), float('inf')
    mean, std = 0.0, 0.0

    for images in dataloader:
        max_pixel = max(max_pixel, images.max().item())
        min_pixel = min(min_pixel, images.min().item())

        images = images.float()
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)

        mean += (images.mean(2).sum(0) / batch_samples)
        std += (images.std(2).sum(0) / batch_samples)

    # Calculate mean and standard deviation for the entire training dataset
    mean /= len(dataloader)
    std /= len(dataloader)

    mean /= max_pixel
    std /= max_pixel

    return mean, std, max_pixel, min_pixel


if __name__ == '__main__':
    # Read out the command line parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, default="<dataset_name>", type=str, help="Which dataset to calculate the statistics for.")
    parser.add_argument("--root_dir", required=True, default="<data_path>", type=str, help="Where the dataset is stored.")

    args = parser.parse_args()
    dataset = args.dataset
    root_dir = args.root_dir

    transform = A.Compose([
        A.Resize(height=224, width=224, interpolation=cv2.INTER_CUBIC, p=1.0),  # Resize
        ToTensorV2()
    ])

    data = Data(dataset, root_dir)
    train_data = data.get_train_data()
    train_set = UnlabeledDataset(train_data, transform)
    train_loader = DataLoader(train_set, batch_size=512, shuffle=False)

    mean, std, max_pixel, min_pixel = calculate_mean_std(train_loader)

    print(f"Mean: {mean}, Std: {std}, Max: {max_pixel}, Min: {min_pixel}")
