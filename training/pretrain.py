"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Pretraining.

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
import argparse
import yaml
import torch
import time
import numpy as np

from pathlib import Path
from copy import deepcopy
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import tqdm
from transformers import set_seed
from timm.optim import AdamW
from timm.scheduler import CosineLRScheduler

# Import own scripts
from ..utils import Misc
from ..models.mixer import SharpMixer
from ..data.data_loader import DatasetLoaderPretrain as DatasetLoader


class PreTrainer:
    def __init__(self, dataset: str, data_path: str, output_path: str, resume_training: dict, starting_epoch: int,
                 epochs: int, img_size: int, in_channel: int, num_mixes: int, learning_rate: float, weight_decay: float,
                 seed: int, batch_size: int, kwargs: dict, max_gpu_batch_size=32, device='cuda:0'):
        """
        Initialize the Pretraining.

        :param dataset: Which dataset to use.
        :param data_path: Where the dataset is stored
        :param output_path: Parent directory to where the trained model shall be stored.
        :param resume_training: Information about whether to resume training from a checkpoint.
        :param starting_epoch: Which epoch to start from if resume training is enabled.
        :param epochs: How many epochs to train for.
        :param img_size: Size of the input images.
        :param in_channel: Number of input channels.
        :param num_mixes: Number of mixtures to use.
        :param batch_size: Batch size to use.
        :param max_gpu_batch_size: Maximum batch size that fits on the GPU.
        :param learning_rate: Learning rate to use.
        :param weight_decay: Weight decay to use.
        :param seed: Seed to use.
        :param device: Device to use.
        :param kwargs: Additional parameters (e.g. depth, num_heads, etc.).
        """

        # Store the parameters
        self.params = {'dataset': dataset, 'data_path': Path(data_path), 'output_path': Path(output_path),
                       'resume_training': resume_training, 'starting_epoch': starting_epoch, 'epochs': epochs,
                       'img_size': img_size, 'in_channel': in_channel, 'num_mixes': num_mixes, 'seed': seed,
                       'batch_size': batch_size, 'eff_batch_size': batch_size, 'max_gpu_batch_size': max_gpu_batch_size,
                       'learning_rate': learning_rate, 'weight_decay': weight_decay, 'device': device, 'kwargs': kwargs
                       }

        # Initialize the accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with="wandb")
        self.params['device'] = self.accelerator.device

        # Set the seed
        set_seed(seed)

        # If the batch size is too big we use gradient accumulation
        self.params['gradient_accumulation_steps'] = 1
        if self.params['batch_size'] > self.params['max_gpu_batch_size']:
            self.params['gradient_accumulation_steps'] = self.params['batch_size'] // self.params['max_gpu_batch_size']
            self.params['eff_batch_size'] = self.params['max_gpu_batch_size']
        else:
            self.params['eff_batch_size'] = self.params['batch_size']

        # Create the path to where the output shall be stored and create the checkpoint file name
        self.params['output_path'].mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = Path(self.params['output_path']) / f"ckp_pretrain_{self.params['dataset']}_patch{self.params['kwargs']['patch_size']}_embed{self.params['kwargs']['embedding_dim']}_seed{self.params['seed']}"

        # Initialize the dataloader
        self.accelerator.print("\tInitializing the dataloader...")
        dataset_loader = DatasetLoader(self.params['dataset'], self.params['data_path'], self.params['img_size'],
                                       self.params['eff_batch_size'], self.params['seed'])

        self.train_loader, self.val_loader = dataset_loader.get_train_loader(), dataset_loader.get_val_loader()

        # Initialize the model
        self.accelerator.print("\tInitializing the model...")
        self.model = SharpMixer(self.params['img_size'], self.params['in_channel'], self.params['num_mixes'], self.params['kwargs'])

        # Initialize the optimizer and lr scheduler
        self.accelerator.print("\tInitializing the optimizer and lr scheduler...")
        self.optimizer = AdamW(self.model.parameters(), lr=self.params['learning_rate'], betas=(0.9, 0.999),
                               eps=1e-8, weight_decay=self.params['weight_decay'])
        self.scheduler = CosineLRScheduler(self.optimizer, t_initial=self.params['epochs'])

        # Create variables to store the best performing model
        self.accelerator.print("\tInitializing helper variables...")
        self.params['best_loss'], self.params['best_epoch'] = np.inf, starting_epoch
        self.best_model = deepcopy(self.model)

        # Resume training if desired and initialize weights & biases to track the losses and metrics
        if self.params['resume_training']['resume']:
            # Load the checkpoint
            checkpoint_file = Path(self.params['output_path']) / f"ckp_pretrain_{self.params['dataset']}_patch{self.params['kwargs']['patch_size']}_embed{self.params['kwargs']['embedding_dim']}_seed{self.params['seed']}_epoch{starting_epoch}.pth"
            checkpoint = torch.load(checkpoint_file, map_location='cpu')

            # Load the model, optimizer and scheduler
            self.model.load_state_dict(checkpoint['model'])
            self.best_model = deepcopy(self.model)

            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

            # Adjust the starting epoch for resuming training
            self.params['starting_epoch'] = checkpoint['epoch'] + 1

            # Reload the associated weights& biases run
            self.accelerator.init_trackers(
                project_name="pretrain",
                config=self.params,
                init_kwargs={
                    "wandb": {
                        "entity": "xxx",
                        "id": self.params['resume_training']['wandb_id'],
                        "resume": "allow",
                    }
                }
            )
        else:
            # Initialize a new weights & biases run
            name = f"{self.params['dataset']}-patch{self.params['kwargs']['patch_size']}-embed{self.params['kwargs']['embedding_dim']}-seed{self.params['seed']}"

            self.accelerator.init_trackers(
                project_name="pretrain",
                config=self.params,
                init_kwargs={
                    "wandb": {
                        "entity": "xxx",
                        "name": name
                    }
                }
            )

        # Prepare the data_loader, optimizer and model for distributed training
        self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader = self.accelerator.prepare(self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader)

    def pretrain(self):
        """
        Run the pretraining.
        """

        # Start code
        start_time = time.time()

        # Start the training
        self.accelerator.print(f"\t\tRun the training for {self.params['epochs'] - self.params['starting_epoch']} epochs...")
        for epoch in range(self.params['starting_epoch'], self.params['epochs']):
            # Stop time
            start_time_epoch = time.time()
            self.accelerator.print(f"\t\t\tEpoch {epoch} of {self.params['epochs']}:")

            # Run the training
            _ = self.run_iteration('Train', self.train_loader, epoch)

            # Run the validation
            with torch.no_grad():
                loss = self.run_iteration('Val', self.val_loader, epoch)

            # Check if the current model is the best one
            if loss < self.params['best_loss']:
                self.params['best_loss'] = loss
                self.params['best_epoch'] = epoch
                self.best_model = deepcopy(self.accelerator.unwrap_model(self.model))

            self.accelerator.print(f"\t\t\tCurrent best Total Loss: {self.params['best_loss']}")
            self.accelerator.print(f"\t\t\tCurrent best Epoch: {self.params['best_epoch']}")

            # Save the current model
            self.accelerator.print(f"\t\t\t\tSaving the checkpoint...")
            Misc.save_model_parallel(self.checkpoint_path, epoch, self.params['epochs'], self.model, self.best_model,
                                     self.optimizer, self.scheduler, self.accelerator)

            # Stop the time for the epoch
            end_time_epoch = time.time()
            hours_epoch, minutes_epoch, seconds_epoch = Misc.calculate_passed_time(start_time_epoch, end_time_epoch)
            self.accelerator.print("\t\t\tElapsed time for epoch: {:0>2}:{:0>2}:{:05.2f}".format(hours_epoch, minutes_epoch, seconds_epoch))

        # Stop the time
        end_time = time.time()
        hours, minutes, seconds = Misc.calculate_passed_time(start_time, end_time)
        self.accelerator.print("\tElapsed time: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))

    def run_iteration(self, mode, data_loader, epoch):
        """
        Run one epoch for the current model.

        :param mode: Train or validation mode.
        :param data_loader: Data loader.
        :param epoch: Current epoch.
        """

        # Initialize a progress bar
        pbar = tqdm(total=len(data_loader), bar_format='{l_bar}{bar}', ncols=80, initial=0, position=0, leave=False)
        pbar.set_description(f'\t\t\t{mode}')

        # Initialize the loss values
        metrics = {metric: 0 for metric in ['total loss', 'consistency loss', 'consistency loss anatomy',
                                            'consistency loss characteristics', 'reconstruction loss', 'psnr']}

        # Run the synthetic image batches through the forward pass of the model
        if mode == 'Train':
            self.model.train()
        else:
            self.model.eval()

        # Iterate through all samples in the data loader
        for i, X in enumerate(data_loader):
            # Run the input through the model to create synthetic images
            loss_recon, loss_consistency_anatomy, loss_consistency_characteristics, psnr, _, _ = self.model(X)

            # Calculate the total loss
            loss = loss_recon + loss_consistency_anatomy + loss_consistency_characteristics
            loss = loss / self.params['gradient_accumulation_steps']

            # Run the backward pass and update the weights
            if mode == 'Train':
                self.accelerator.backward(loss)

                if i % self.params['gradient_accumulation_steps'] == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Append the current loss values to the overall values for the current epoch
            metrics_batch = {'total loss': loss, 'consistency loss': loss_consistency_anatomy + loss_consistency_characteristics,
                             'consistency loss anatomy': loss_consistency_anatomy, 'consistency loss characteristics': loss_consistency_characteristics,
                             'reconstruction loss': loss_recon, 'psnr': psnr}

            for key in metrics.keys():
                metrics[key] += metrics_batch[key].detach().cpu()

            # Update the progress bar
            pbar.update(1)

        # Update the learning rate scheduler
        if mode == 'Train':
            self.scheduler.step(epoch)

        # Calculate the average metric values per batch and log the results
        for key, value in metrics.items():
            metrics[key] /= len(data_loader)

            self.accelerator.print(f"\t\t\t\t{mode} {key}: {value}")

        self.accelerator.log({f"{mode} {key}": value for key, value in metrics.items()}, step=epoch)

        # Return the total loss
        return metrics['total loss']


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
    pretrainer = PreTrainer(config["dataset"], config["data_path"], config["output_path"], config['resume_training'],
                            config["starting_epoch"], config["epochs"], config["img_size"], config["in_channel"],
                            config["num_mixes"], config["learning_rate"], config["weight_decay"], config["seed"],
                            config["batch_size"], config['kwargs'], config["max_gpu_batch_size"], config['device'])

    # Run the pretraining
    pretrainer.pretrain()
