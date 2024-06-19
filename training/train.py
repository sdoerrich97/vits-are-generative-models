"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Training.

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
import argparse
import yaml
import torch
import time
import torch.nn as nn

from pathlib import Path
from copy import deepcopy
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import tqdm
from transformers import set_seed
from timm.optim import AdamW
from timm.scheduler import CosineLRScheduler
from sklearn.metrics import accuracy_score

# Import own scripts
from code.utils import Misc
from code.models.mixer import SharpMixer
from code.models.classifier import SharpClassifier
from code.data.data_loader import DatasetLoaderTrain as DatasetLoader


class Trainer:
    def __init__(self, dataset: str, data_path: str, input_path: str, output_path: str, backbone_classifier: str,
                 resume_training: dict, starting_epoch: int, epochs: int, img_size: int, in_channel: int,
                 num_mixes: int, num_classes: int, learning_rate: float, weight_decay: float, seed: int,
                 batch_size: int, kwargs: dict, max_gpu_batch_size=32, device='cuda:0'):
        """
        Initialize the Training.

        :param dataset: Which dataset to use.
        :param data_path: Where the dataset is stored
        :param input_path: Parent directory to where the pre-trained model is stored.
        :param output_path: Parent directory to where the trained model shall be stored.
        :param backbone_classifier: Which backbone to use for the classifier.
        :param resume_training: Information about whether to resume training from a checkpoint.
        :param starting_epoch: Which epoch to start from if resume training is enabled.
        :param epochs: How many epochs to train for.
        :param img_size: Size of the input images.
        :param in_channel: Number of input channels.
        :param num_classes: Number of classes to predict.
        :param batch_size: Batch size to use.
        :param max_gpu_batch_size: Maximum batch size that fits on the GPU.
        :param learning_rate: Learning rate to use.
        :param weight_decay: Weight decay to use.
        :param seed: Seed to use.
        :param device: Device to use.
        :param kwargs: Additional parameters (e.g. depth, num_heads, etc.).
        """

        # Store the parameters
        self.params = {'dataset': dataset, 'data_path': Path(data_path), 'input_path': Path(input_path),
                       'output_path': Path(output_path), 'backbone_classifier': backbone_classifier,
                       'resume_training': resume_training, 'starting_epoch': starting_epoch, 'epochs': epochs,
                       'img_size': img_size, 'in_channel': in_channel, 'num_mixes': num_mixes,
                       'num_classes': num_classes, 'seed': seed, 'batch_size': batch_size, 'eff_batch_size': batch_size,
                       'max_gpu_batch_size': max_gpu_batch_size, 'learning_rate': learning_rate,
                       'weight_decay': weight_decay, 'device': device, 'kwargs': kwargs
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
        self.checkpoint_path = Path(self.params['output_path']) / f"ckp_train_{self.params['dataset']}_{self.params['backbone_classifier']}_patch{self.params['kwargs']['patch_size']}_embed{self.params['kwargs']['embedding_dim']}_seed{self.params['seed']}"

        # Initialize the dataloader
        self.accelerator.print("\tInitializing the dataloader...")
        dataset_loader = DatasetLoader(self.params['dataset'], self.params['data_path'], self.params['img_size'],
                                       self.params['eff_batch_size'], self.params['seed'])

        self.train_loader, self.val_loader = dataset_loader.get_train_loader(), dataset_loader.get_val_loader()

        # Initialize the pretrained encoder
        self.accelerator.print("\tInitializing the pretrained encoder...")
        self.encoder = SharpMixer(self.params['img_size'], self.params['in_channel'], self.params['num_mixes'], kwargs)

        ckp_pretraining_path = Path(self.params['input_path']) / f"ckp_pretrain_{self.params['dataset']}_patch{self.params['kwargs']['patch_size']}_embed{self.params['kwargs']['embedding_dim']}_seed{self.params['seed']}_best.pth"
        checkpoint = torch.load(ckp_pretraining_path, map_location='cpu')

        self.encoder.load_state_dict(checkpoint)

        # Initialize the classifier
        self.accelerator.print("\tInitializing the classifier...")
        self.classifier = SharpClassifier(backbone=self.params['backbone_classifier'],
                                          num_classes=self.params['num_classes'])

        # Initialize the optimizer and lr scheduler for the classifier
        self.accelerator.print("\tInitializing the optimizer and lr scheduler for the classifier...")
        self.optimizer = AdamW(self.classifier.parameters(), lr=self.params['learning_rate'], betas=(0.9, 0.999),
                               eps=1e-8, weight_decay=self.params['weight_decay'])
        self.scheduler = CosineLRScheduler(self.optimizer, t_initial=self.params['epochs'])

        # Resume training if desired and initialize weights & biases to track the losses and metrics
        if self.params['resume_training']['resume']:
            self.accelerator.print("\tLoading training checkpoint...")
            # Load the checkpoint
            checkpoint_file = Path(self.params['output_path']) / f"ckp_train_{self.params['dataset']}_{self.params['backbone_classifier']}_patch{self.params['kwargs']['patch_size']}_embed{self.params['kwargs']['embedding_dim']}_seed{self.params['seed']}_epoch{starting_epoch}.pth"
            checkpoint = torch.load(checkpoint_file, map_location='cpu')

            # Load the model, optimizer and scheduler
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.classifier.load_state_dict(checkpoint['classifier'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

            # Adjust the starting epoch for resuming training
            self.params['starting_epoch'] = checkpoint['epoch'] + 1

        # Freeze the encoder and map the model to the GPU
        self.encoder.to(self.params['device'])
        self.classifier.to(self.params['device'])
        self.encoder.requires_grad_(False)
        self.classifier.requires_grad_(True)

        # Create loss criterion and helper variables to store the best performing model based on the training scheme
        self.accelerator.print("\tInitializing loss criterion and helper variables...")
        self.loss_criterion = nn.CrossEntropyLoss()
        self.best_classifier = deepcopy(self.classifier)

        # Initialize new weights & biases to track the metrics and losses
        self.accelerator.print("\tInitializing weights & biases to track the metrics and losses...")
        if self.params['resume_training']['resume']:
            # Resume the weights & biases project for the specified training run
            self.accelerator.init_trackers(
                project_name="train",
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
            # Initialize a weights & biases project for a training run with the given training configuration
            name = f"{self.params['dataset']}-{self.params['backbone_classifier']}-patch{self.params['kwargs']['patch_size']}-embed{self.params['kwargs']['embedding_dim']}-seed{self.params['seed']}"
            
            self.accelerator.init_trackers(
                project_name="train",
                config=self.params,
                init_kwargs={
                    "wandb": {
                        "entity": "xxx",
                        "name": name
                    }
                }
            )

        # Prepare the data_loader, optimizer and model for distributed training
        self.encoder, self.classifier, self.best_classifier, self.optimizer, self.scheduler, self.train_loader, self.val_loader = self.accelerator.prepare(self.encoder, self.classifier, self.best_classifier, self.optimizer, self.scheduler, self.train_loader, self.val_loader)

    def train(self):
        """
        Run the training.
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

            if loss < self.params['best_loss']:
                self.params['best_loss'] = loss
                self.params['best_epoch'] = epoch
                self.best_classifier = deepcopy(self.classifier)

            self.accelerator.print(f"\t\t\tCurrent best Loss: {self.params['best_loss']}, Current best Epoch: {self.params['best_epoch']}")

            # Save the current model
            self.accelerator.print(f"\t\t\tSaving the checkpoint...")
            Misc.save_training(self.checkpoint_path, epoch, self.params['epochs'], self.encoder, self.classifier,
                               self.best_classifier, self.optimizer, self.scheduler, self.accelerator)

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
        Run one epoch for the current model and end-to-end training.

        :param mode: Train or validation mode.
        :param data_loader: Data loader.
        :param epoch: Current epoch.
        """

        # Initialize a progress bar
        pbar = tqdm(total=len(data_loader), bar_format='{l_bar}{bar}', ncols=80, initial=0, position=0, leave=False)
        pbar.set_description(f'\t\t\t{mode}')

        # Run the synthetic image batches through the forward pass of the model
        if mode == 'Train':
            self.classifier.train()
        else:
            self.classifier.eval()

        # Initialize the metrics
        total_loss = 0
        Y_true, Y_pred = torch.tensor([]), torch.tensor([])

        # Iterate through all samples in the data loader
        for i, (X, Y) in enumerate(data_loader):
            # Run the input images through the encoder
            with torch.no_grad():
                S = self.encoder.create_synthetic_images(X)

            # Stack all the images and the labels
            if mode == 'Train':
                X_total = torch.vstack([X, S])
                Y_total = torch.cat([Y] * (self.params['num_mixes'] + 1), dim=0)

            else:
                X_total, Y_total = X, Y

            # Run all images through the classifier to get the outputs
            outputs = self.classifier(X_total)

            # Calculate the loss
            loss = self.loss_criterion(outputs, Y_total)
            loss = loss / self.params['gradient_accumulation_steps']

            # Run the backward pass and update the weights
            if mode == 'Train':
                self.accelerator.backward(loss)

                if i % self.params['gradient_accumulation_steps'] == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Update the total loss
            total_loss += loss.item()

            # Create the predictions and append the current predictions and targets to the overall predictions and targets
            pred = torch.argmax(outputs, dim=1, keepdim=True)
            Y_true = torch.cat((Y_true, deepcopy(Y_total).cpu()), dim=0)
            Y_pred = torch.cat((Y_pred, deepcopy(pred).cpu()), dim=0)

            # Update the progress bar
            pbar.update(1)

        # Update the learning rate scheduler
        if mode == 'Train':
            self.scheduler.step(epoch)

        # Average the loss value across all batches and compute the performance metrics
        total_loss /= len(data_loader)

        # Calculate the accuracy
        ACC = accuracy_score(Y_true.numpy(), Y_pred.numpy())

        self.accelerator.print(f"\t\t\t\t{mode} Loss: {total_loss}")
        self.accelerator.print(f"\t\t\t\t{mode} Accuracy: {ACC}")

        self.accelerator.log(
            {
                f"{mode} Loss": total_loss,
                f"{mode} Accuracy": ACC
            }, step=epoch)

        # Return the total loss
        return total_loss


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

    # Initialize the pretraining
    trainer = Trainer(config["dataset"], config["data_path"], config["input_path"], config["output_path"],
                      config["backbone_classifier"], config['resume_training'], config["starting_epoch"],
                      config["epochs"], config["img_size"], config["in_channel"], config["num_mixes"],
                      config["num_classes"], config["learning_rate"], config["weight_decay"], config["seed"],
                      config["batch_size"], config['kwargs'], config["max_gpu_batch_size"], config['device'])

    # Run the pretraining
    trainer.train()
