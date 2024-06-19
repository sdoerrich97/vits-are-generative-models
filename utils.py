"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Helper functions.

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
import torch
import torch.nn.functional as F


class Misc:
    @staticmethod
    def save_training(output_path, epoch, max_epoch, encoder, classifier, best_classifier, optimizer, scheduler,
                      accelerator, save_idx=50):
        """
        Save the model.
        :param output_path: Output path.
        :param epoch: Which epoch.
        :param max_epoch: maximum number of epochs.
        :param encoder: Encoder model to save.
        :param classifier: Classifier model to save.
        :param best_classifier: Best performing classifier model at current epoch.
        :param optimizer: Optimizer of current model.
        :param scheduler: Scheduler of current model.
        :param accelerator: Accelerator used for Multi-GPU processing.
        :param save_idx: How often to save an intermediate model.
        """

        # Let all processes finish before saving the model
        accelerator.wait_for_everyone()

        # Create the checkpoint
        checkpoint = {
            'encoder': accelerator.unwrap_model(encoder).state_dict(),
            'classifier': accelerator.unwrap_model(classifier).state_dict(),
            'optimizer': optimizer.optimizer.state_dict(),  # optimizer is an AcceleratedOptimizer object
            'scheduler': scheduler.state_dict()
        }

        # Save the checkpoint
        accelerator.save(checkpoint, str(output_path) + "_latest.pth")

        if epoch % save_idx == 0:
            # Let all processes finish before saving the model
            accelerator.wait_for_everyone()

            # Create the checkpoint
            checkpoint = {
                'encoder': accelerator.unwrap_model(encoder).state_dict(),
                'classifier': accelerator.unwrap_model(classifier).state_dict(),
                'optimizer': optimizer.optimizer.state_dict(),  # optimizer is an AcceleratedOptimizer object
                'scheduler': scheduler.state_dict()
            }

            # Save the checkpoint
            accelerator.save(checkpoint, str(output_path) + f"_{epoch}.pth")

            # Unwrap the models
            unwrapped_model_final = accelerator.unwrap_model(classifier)
            unwrapped_model_best = accelerator.unwrap_model(best_classifier)
            accelerator.save(unwrapped_model_final.state_dict(), str(output_path) + f"_final.pth")
            accelerator.save(unwrapped_model_best.state_dict(), str(output_path) + f"_best.pth")

        if epoch == max_epoch - 1:
            # Let all processes finish before saving the model
            accelerator.wait_for_everyone()

            # Unwrap the models
            unwrapped_model_final = accelerator.unwrap_model(classifier)
            unwrapped_model_best = accelerator.unwrap_model(best_classifier)

            # Save the models
            accelerator.save(unwrapped_model_final.state_dict(), str(output_path) + f"_final.pth")
            accelerator.save(unwrapped_model_best.state_dict(), str(output_path) + f"_best.pth")

    @staticmethod
    def save_model_parallel(output_path, epoch, max_epoch, model, best_model, optimizer, scheduler, accelerator, save_idx=50):
        """
        Save the model.
        :param output_path: Output path.
        :param epoch: Which epoch.
        :param max_epoch: maximum number of epochs.
        :param model: Model to save.
        :param best_model: Best performing model at current epoch.
        :param optimizer: Optimizer of current model.
        :param scheduler: Scheduler of current model.
        :param accelerator: Accelerator used for Multi-GPU processing.
        :param save_idx: How often to save an intermediate model.
        """

        # Let all processes finish before saving the model
        accelerator.wait_for_everyone()

        # Create the checkpoint
        checkpoint = {
            'model': accelerator.unwrap_model(model).state_dict(),
            'optimizer': optimizer.optimizer.state_dict(),  # optimizer is an AcceleratedOptimizer object
            'scheduler': scheduler.state_dict()
        }

        # Save the checkpoint
        accelerator.save(checkpoint, str(output_path) + "_latest.pth")

        if epoch % save_idx == 0:
            # Let all processes finish before saving the model
            accelerator.wait_for_everyone()

            # Create the checkpoint
            checkpoint = {
                'model': accelerator.unwrap_model(model).state_dict(),
                'optimizer': optimizer.optimizer.state_dict(),  # optimizer is an AcceleratedOptimizer object
                'scheduler': scheduler.state_dict()
            }

            # Save the checkpoint
            accelerator.save(checkpoint, str(output_path) + f"_{epoch}.pth")

            # Unwrap the models
            unwrapped_model_final = accelerator.unwrap_model(model)
            unwrapped_model_best = accelerator.unwrap_model(best_model)
            accelerator.save(unwrapped_model_final.state_dict(), str(output_path) + f"_final.pth")
            accelerator.save(unwrapped_model_best.state_dict(), str(output_path) + f"_best.pth")

        if epoch == max_epoch - 1:
            # Let all processes finish before saving the model
            accelerator.wait_for_everyone()

            # Unwrap the models
            unwrapped_model_final = accelerator.unwrap_model(model)
            unwrapped_model_best = accelerator.unwrap_model(best_model)

            # Save the models
            accelerator.save(unwrapped_model_final.state_dict(), str(output_path) + f"_final.pth")
            accelerator.save(unwrapped_model_best.state_dict(), str(output_path) + f"_best.pth")

    """ Helper functions for training and validation. """
    @staticmethod
    def calculate_passed_time(start_time, end_time):
        """
        Calculate the time needed for running the code

        :param: start_time: Start time.
        :param: end_time: End time.
        :return: Duration in hh:mm:ss.ss
        """

        # Calculate the duration
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        # Return the duration in hours, minutes and seconds
        return int(hours), int(minutes), seconds


class Loss:
    """ Helper functions for loss calculation."""

    @staticmethod
    def calculate_mse(tensor_1: torch.tensor, tensor_2: torch.tensor):
        """
        Calculate the mean-squared-error (MSE) for the given input tensors.

        :param tensor_1: First Tensor.
        :param tensor_2: Second Tensor.
        :return: MSE.
        """

        return F.mse_loss(tensor_1, tensor_2)

    @staticmethod
    def calculate_psnr(original: torch.tensor, reconstruction: torch.tensor):
        """
        Calculate the peak-signal-to-noise-ratio (PSNR) for the given original image and its reconstruction.

        :param original: Original input image.
        :param reconstruction: Reconstructed image.
        :return: PSNR.
        """

        # Normalize the images based on the original image's min and max values
        original_min, original_max = original.min(), original.max()
        original = (original - original_min) / (original_max - original_min)
        reconstruction = (reconstruction - original_min) / (original_max - original_min)

        # Calculate the mse value
        mse = Loss.calculate_mse(original, reconstruction)

        # Calculate the psnr value
        return 20 * torch.log10(torch.tensor(1).to(original.device)) - 10 * torch.log10(mse)
