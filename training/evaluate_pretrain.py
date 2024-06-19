"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Inference of the pretraining.

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
import argparse
import yaml
import torch
import time
import wandb
from tqdm import tqdm

from pathlib import Path
from transformers import set_seed

# Import own scripts
from code.utils import Misc
from code.models.mixer import SharpMixer
from code.data.data_loader import DatasetLoaderPretrain as DatasetLoader


class EvaluatePretraining:
    def __init__(self, dataset: str, data_path: str, input_path: str, output_path: str, img_size: int, in_channel: int,
                 num_mixes: int, batch_size: int, seed: int, kwargs: dict, device='cuda:0'):
        """
        Initialize the Evaluation.

        :param dataset: Which dataset to use.
        :param data_path: Where the dataset is stored
        :param input_path: Parent directory to where the trained model is stored.
        :param output_path: Parent directory to where the trained model shall be stored.
        :param img_size: Size of the input images.
        :param in_channel: Number of input channels.
        :param num_mixes: Number of mixtures to use.
        :param batch_size: Batch size to use.
        :param seed: Seed to use.
        :param device: Device to use.
        :param kwargs: Additional parameters (e.g. depth, num_heads, etc.).
        """

        # Store the parameters
        self.params = {'dataset': dataset, 'data_path': Path(data_path), 'input_path': input_path,
                       'output_path': Path(output_path), 'img_size': img_size, 'in_channel': in_channel,
                       'num_mixes': num_mixes, 'batch_size': batch_size, 'seed': seed, 'device': device,
                       'kwargs': kwargs
                       }

        # Set the seed
        set_seed(seed)
        torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark for the evaluation

        # Create the path to where the output shall be stored
        self.params['output_path'] = Path(self.params['output_path']) / f"{self.params['dataset']}_patch{self.params['kwargs']['patch_size']}_embed{self.params['kwargs']['embedding_dim']}_seed{self.params['seed']}"
        self.params['output_path'].mkdir(parents=True, exist_ok=True)

        # Initialize the dataloader
        print("\tInitializing the dataloader...")
        dataset_loader = DatasetLoader(self.params['dataset'], self.params['data_path'], self.params['img_size'],
                                       self.params['batch_size'], self.params['seed'], train=False)

        self.train_loader, self.val_loader, self.test_loader = (dataset_loader.get_train_loader(),
                                                                dataset_loader.get_val_loader(),
                                                                dataset_loader.get_test_loader())

        # Initialize the model
        print("\tInitializing the model...")
        self.model = SharpMixer(self.params['img_size'], self.params['in_channel'], self.params['num_mixes'], self.params['kwargs'])

        # Load the pretrained model
        print("\tLoad the pretrained model weights...")
        checkpoint_file = Path(self.params['input_path']) / f"ckp_pretrain_{self.params['dataset']}_patch{self.params['kwargs']['patch_size']}_embed{self.params['kwargs']['embedding_dim']}_seed{self.params['seed']}_best.pth"
        checkpoint = torch.load(checkpoint_file, map_location='cpu')

        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.params['device'])
        self.model.requires_grad_(False)

        # Initialize new weights & biases to track the metrics and losses
        print("\tInitialize weights & biases to track the metrics and losses...")
        wandb.init(project="pretrain_evaluation",
                   name=f"{self.params['dataset']}-patch{self.params['kwargs']['patch_size']}-embed{self.params['kwargs']['embedding_dim']}-seed{self.params['seed']}",
                   config=self.params)

    def evaluate(self):
        """
        Run the evaluation of the pretraining.
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

            # Create the output path for the current set
            output_path = self.params['output_path'] / set
            output_path.mkdir(parents=True, exist_ok=True)

            nr_batches = len(set_loader)
            print(f"\t\t\tRun evaluation for {set} for {nr_batches} batches...")

            with torch.no_grad():
                # Set the model into evaluation mode
                self.model.eval()

                # Run the inference
                self.run_evaluation_for_single_set(set_loader, nr_batches, output_path)

            # Stop the time
            end_time_set = time.time()
            hours_set, minutes_set, seconds_set = Misc.calculate_passed_time(start_time_set, end_time_set)

            print("\t\t\tElapsed time for set '{}': {:0>2}:{:0>2}:{:05.2f}".format(set, hours_set, minutes_set,
                                                                                   seconds_set))

            # Stop the time
        end_time = time.time()
        hours, minutes, seconds = Misc.calculate_passed_time(start_time, end_time)

        print("\t\t\tElapsed time for inference: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))

    def run_evaluation_for_single_set(self, set_loader, nr_batches, output_path):
        """
        Run the inference for the given set.

        :param set_loader: Data loader for the current set.
        :param nr_batches: Number of batches to process.
        :param output_path: Where the image shall be stored.
        """

        # Initialize a progress bar
        pbar = tqdm(total=len(set_loader), bar_format='{l_bar}{bar}', ncols=80, initial=0, position=0, leave=False)
        pbar.set_description(f'\t\t\tProgres')

        mse, psnr = 0, 0
        self.model.eval()

        for i, X in enumerate(set_loader):
            # Map the input to the respective device
            X = X.to(self.params['device'])

            # Run the input through the model
            mse_batch, _, _, psnr_batch, X_hat, _ = self.model(X)

            # Append the current loss values to the overall values for the set
            mse += mse_batch.cpu()
            psnr += psnr_batch.cpu()

            # Update the progress bar
            pbar.update(1)

        # Average the metrics over the number of samples for the current set
        mse /= nr_batches
        psnr /= nr_batches

        # Print the loss values and send them to wandb
        print(f"\t\t\tAverage MSE: {mse}")
        print(f"\t\t\tAverage PSNR: {psnr}")

        # Log the metrics' averages
        wandb.log(
            {
                "Average MSE": mse,
                "Average PSNR": psnr,
            })


if __name__ == '__main__':
    # Read out the command line parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str, help="Path to the configuration file to use.")
    parser.add_argument("--dataset", required=False, type=str, help="Which dataset to use.")
    parser.add_argument("--data_path", required=False, type=str, help="Where the dataset is stored.")
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

    # If a seed is specified, overwrite the seed in the config file
    if args.seed:
        config['seed'] = args.seed

    # Initialize the pretraining
    evaluater = EvaluatePretraining(config["dataset"], config["data_path"], config["input_path"],
                                    config["output_path"], config["img_size"], config["in_channel"],
                                    config["num_mixes"], config["batch_size"], config["seed"], config['kwargs'],
                                    config['device'])

    # Run the pretraining
    evaluater.evaluate()
