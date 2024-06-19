"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Inference of the training.

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
import argparse
import yaml
import torch
import time
import wandb
import torch.nn as nn
from tqdm import tqdm

from pathlib import Path
from copy import deepcopy
from transformers import set_seed
from sklearn.metrics import accuracy_score

# Import own scripts
from code.utils import Misc
from code.models.classifier import SharpClassifier
from code.data.data_loader import DatasetLoaderTrain as DatasetLoader


class EvaluateTraining:
    def __init__(self, dataset: str, data_path: str, input_path: str, backbone_classifier: str, img_size: int,
                 in_channel: int, num_classes: int, seed: int, batch_size: int, kwargs: dict, device='cuda:0'):
        """
        Initialize the Evaluation.

        :param dataset: Which dataset to use.
        :param data_path: Where the dataset is stored
        :param input_path: Parent directory to where the pre-trained model is stored.
        :param backbone_classifier: Which backbone to use for the classifier.
        :param img_size: Size of the input images.
        :param in_channel: Number of input channels.
        :param num_classes: Number of classes to predict.
        :param batch_size: Batch size to use.
        :param seed: Seed to use.
        :param device: Device to use.
        :param kwargs: Additional parameters (e.g. patch size, batch size, etc.).
        """

        # Store the parameters
        self.params = {'dataset': dataset, 'data_path': Path(data_path), 'input_path': Path(input_path),
                       'backbone_classifier': backbone_classifier, 'img_size': img_size, 'in_channel': in_channel,
                       'num_classes': num_classes, 'seed': seed, 'batch_size': batch_size, 'eff_batch_size': batch_size,
                       'device': device, 'kwargs': kwargs
                       }

        # Set the seed
        set_seed(seed)
        torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark for the evaluation

        # Create the path to where the output shall be stored and create the checkpoint file name
        # Initialize the dataloader
        print("\tInitializing the dataloader...")
        dataset_loader = DatasetLoader(self.params['dataset'], self.params['data_path'], self.params['img_size'],
                                       self.params['batch_size'], self.params['seed'], train=False)

        self.train_loader, self.val_loader, self.test_loader = dataset_loader.get_train_loader(), dataset_loader.get_val_loader(), dataset_loader.get_test_loader()

        # Initialize the pretrained encoder
        print("\tInitializing the model...")
        self.model = SharpClassifier(backbone=self.params['backbone_classifier'], num_classes=self.params['num_classes'])

        ckp_training_path = Path(self.params['input_path']) / f"ckp_train_{self.params['dataset']}_{self.params['backbone_classifier']}_patch{self.params['kwargs']['patch_size']}_embed{self.params['kwargs']['embedding_dim']}_seed{self.params['seed']}_best.pth"

        self.model.load_state_dict(torch.load(ckp_training_path, map_location='cpu'))
        self.model.to(self.params['device'])
        self.model.requires_grad_(False)

        # Create loss criterion and helper variables to store the best performing model based on the training scheme
        print("\tInitializing loss criterion...")
        self.loss_criterion = nn.CrossEntropyLoss()

        # Initialize new weights & biases to track the metrics and losses
        print("\tInitializing weights & biases to track the metrics and losses...")
        name = f"{self.params['dataset']}-{self.params['backbone_classifier']}-patch{self.params['kwargs']['patch_size']}-embed{self.params['kwargs']['embedding_dim']}-seed{self.params['seed']}"

        wandb.init(project="train_evaluation",
                   name=name,
                   config=self.params)

    def evaluate(self):
        """
        Run the evaluation of the training.
        """

        # Start code
        start_time = time.time()

        # Empty the unused memory cache
        print("\tRun the evaluation...")
        print("\tEmptying the unused memory cache...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for set, set_loader in zip(['train', 'val', 'test'], [self.train_loader, self.val_loader, self.test_loader]):
            # Run the inference for the current set
            start_time_set = time.time()
            print(f"\t\tRun evaluation for {set}...")

            nr_batches = len(set_loader)
            print(f"\t\t\tRun evaluation for {set} for {nr_batches} batches...")

            with torch.no_grad():
                # Set the model into evaluation mode
                self.model.eval()

                # Run the inference
                self.run_evaluation_for_single_set(set_loader)

            # Stop the time
            end_time_set = time.time()
            hours_set, minutes_set, seconds_set = Misc.calculate_passed_time(start_time_set, end_time_set)

            print("\t\t\tElapsed time for set '{}': {:0>2}:{:0>2}:{:05.2f}".format(set, hours_set, minutes_set,
                                                                                   seconds_set))

            # Stop the time
        end_time = time.time()
        hours, minutes, seconds = Misc.calculate_passed_time(start_time, end_time)

        print("\t\t\tElapsed time for inference: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))

    def run_evaluation_for_single_set(self, set_loader):
        """
        Run the inference for the given set.

        :param set_loader: Data loader for the current set.
        """

        # Initialize a progress bar
        pbar = tqdm(total=len(set_loader), bar_format='{l_bar}{bar}', ncols=80, initial=0, position=0, leave=False)
        pbar.set_description(f'\t\t\tProgres')

        # Initialize the metrics
        total_loss = 0
        Y_true, Y_pred = torch.tensor([]), torch.tensor([])

        # Iterate through all samples in the data loader
        for i, (X, Y) in enumerate(set_loader):
            # Map the input to the respective device
            X, Y = X.to(self.params['device']), Y.to(self.params['device'])

            # Run the input through the model
            outputs = self.model(X)

            # Calculate the loss
            loss = self.loss_criterion(outputs, Y)
            total_loss += loss.item()  # Update the total loss

            # Create the predictions and ppend the current predictions and targets to the overall predictions and targets
            pred = torch.argmax(outputs, dim=1, keepdim=True)
            Y_true = torch.cat((Y_true, deepcopy(Y).cpu()), dim=0)
            Y_pred = torch.cat((Y_pred, deepcopy(pred).cpu()), dim=0)

            # Update the progress bar
            pbar.update(1)

        # Average the loss value across all batches and compute the performance metrics
        total_loss /= len(set_loader)

        # Calculate the accuracy
        ACC = accuracy_score(Y_true.numpy(), Y_pred.numpy())

        print(f"\t\t\t\tLoss: {total_loss}")
        print(f"\t\t\t\tAccuracy: {ACC}")

        wandb.log(
            {
                f"Loss": total_loss,
                f"Accuracy": ACC
            })


if __name__ == '__main__':
    # Read out the command line parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str, help="Path to the configuration file to use.")
    parser.add_argument("--dataset", required=False, type=str, help="Which dataset to use.")
    parser.add_argument("--data_path", required=False, type=str, help="Where the dataset is stored.")
    parser.add_argument("--backbone_classifier", required=False, type=str, help="Which backbone to us for the classifier head.")
    parser.add_argument("--num_classes", required=False, type=int, help="Number of classes to predict.")
    parser.add_argument("--seed", required=False, type=int, help="Which seed was used during training.")

    args = parser.parse_args()
    config_file = args.config_file

    # Load the parameters and hyperparameters of the configuration file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # If a data set is specified, overwrite the data set in the config file
    if args.dataset:
        config['dataset'] = args.dataset

    # If a data path is specified, overwrite the data path in the config file
    if args.data_path:
        config['data_path'] = args.data_path

    # If a backbone for the classifier is specified, overwrite the backbone in the config file
    if args.backbone_classifier:
        config['backbone_classifier'] = args.backbone_classifier

    # If a backbone for the classifier is specified, overwrite the backbone in the config file
    if args.num_classes:
        config['num_classes'] = args.num_classes

    # If a seed is specified, overwrite the seed in the config file
    if args.seed:
        config['seed'] = args.seed

    # Initialize the evaluator
    evaluater = EvaluateTraining(config["dataset"], config["data_path"], config["input_path"],
                                 config["backbone_classifier"], config["img_size"], config["in_channel"],
                                 config["num_classes"], config["seed"], config["batch_size"], config['kwargs'],
                                 config['device'])

    # Run the evaluation
    evaluater.evaluate()
