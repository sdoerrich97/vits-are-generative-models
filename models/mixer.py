"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Implementation of the unsupervised feature orthogonalization with intermixing

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
import torch
import torch.nn as nn

# Import own packages
from blocks.vit import EncoderViT
from ..utils import Loss
from .utils import Mixing


class SharpMixer(nn.Module):
    def __init__(self, img_size: int, in_channel: int, num_mixes: int = 4, kwargs: dict = None):
        """
        Constructor of the autoencoder.
        :param img_size: Width and height of the input image.
        :param in_channel: Channel dimension of the input.
        :param num_mixes: Number of characteristic feature embeddings to mix with each anatomical feature embedding.
        :param kwargs: Additional backbone specific arguments.
        """

        # Initialize the parent constructor
        super().__init__()
        self.img_size = img_size
        self.in_channel = in_channel
        self.num_mixes = num_mixes

        # Create the encoder
        self.encoder = EncoderViT(img_size, in_channel, patch_size=kwargs['patch_size'],
                                  embed_dim=kwargs['embedding_dim'], depth=kwargs['depth'],
                                  num_heads=kwargs['num_heads'], mlp_ratio=kwargs['mlp_ratio'],
                                  norm_layer=kwargs['norm_layer'])

        self.patch_size = kwargs['patch_size']

    def forward(self, X: torch.Tensor):
        """
        Forward pass of the model.

        :param X: Batch of input images [B, C, H, W].

        :return: Losses and reconstructions
        """

        # Create the target indices for the mixing
        target_sample_idxs = torch.randint(0, X.size(0), size=(self.num_mixes,))
        target_patch_idxs = torch.randint(0, X.size(1), size=(self.num_mixes,))

        # Run the original input batch through the encoder and split the latent representations into anatomical and
        # characteristic features
        Z_orig = self.encoder(X)  # ([B, P + 1, E] for ViT)

        Z_orig_a = Z_orig[:, :, :Z_orig.size(-1) // 2]  # Anatomical Features
        Z_orig_c = Z_orig[:, :, Z_orig.size(-1) // 2:]  # Characteristic Features

        # Reconstruct the original images and calculate the reconstruction loss
        X_recon = Mixing.patch_wise_image_synthesis(Z_orig_a, Z_orig_c, in_channel=self.in_channel,
                                                    patch_size=self.patch_size)

        loss_reconstructions = Loss.calculate_mse(X, X_recon)
        psnr_reconstructions = Loss.calculate_psnr(X, X_recon)

        # Run the reconstructed input through the encoder and split the latent representations into anatomical and
        # characteristic features
        Z_recon = self.encoder(X_recon)  # ([B, P + 1, E] for ViT)
        Z_recon_a = Z_recon[:, :, :Z_recon.size(-1) // 2]  # Anatomical Features
        Z_recon_c = Z_recon[:, :, Z_recon.size(-1) // 2:]  # Characteristic Features

        # Calculate the consistency losses for the original batch and its reconstructions
        loss_consistency_anatomy = Loss.calculate_mse(Z_recon_a, Z_orig_a)  # Anatomical consistency losses
        loss_consistency_characteristics = Loss.calculate_mse(Z_recon_c, Z_orig_c)  # Characteristics consistency losses

        # Iterate over the target indices and mix the original batch with the target batches
        synthetic_images = []
        for mix_idx in range(self.num_mixes):
            # Get the target characteristics for the current mixing
            Z_target_c = Z_orig_c[target_sample_idxs[mix_idx]][target_patch_idxs[mix_idx]].view(1, 1, -1).expand(Z_orig_c.size(0), Z_orig_c.size(1), -1)

            # Create the synthetic images, pass them through the encoder and split the latent representations
            X_synth = Mixing.patch_wise_image_synthesis(Z_orig_a, Z_target_c, in_channel=self.in_channel,
                                                        patch_size=self.patch_size)
            synthetic_images.append(X_synth)

            Z_synth = self.encoder(X_synth)
            Z_synth_a = Z_synth[:, :, :Z_synth.size(-1) // 2]
            Z_synth_c = Z_synth[:, :, Z_synth.size(-1) // 2:]

            # Calculate the consistency losses for the current mix
            loss_consistency_anatomy += Loss.calculate_mse(Z_synth_a, Z_orig_a)
            loss_consistency_characteristics += Loss.calculate_mse(Z_synth_c, Z_target_c)

        # Stack the synthetic images
        if len(synthetic_images) > 0:
            synthetic_images = torch.vstack(synthetic_images)

        # return the losses and the reconstructions
        return loss_reconstructions, loss_consistency_anatomy, loss_consistency_characteristics, psnr_reconstructions, X_recon, synthetic_images

    def create_synthetic_images(self, X: torch.Tensor):
        """
        Forward pass of the model.

        :param X: Batch of input images [B, C, H, W].

        :return: Losses and reconstructions
        """

        # Create the target indices for the mixing
        target_sample_idxs = torch.randint(0, X.size(0), size=(self.num_mixes,))
        target_patch_idxs = torch.randint(0, X.size(1), size=(self.num_mixes,))

        # Run the original input batch through the encoder and split the latent representations into anatomical and
        # characteristic features
        Z_orig = self.encoder(X)  # ([B, P + 1, E] for ViT)
        Z_orig_a = Z_orig[:, :, :Z_orig.size(-1) // 2]  # Anatomical Features
        Z_orig_c = Z_orig[:, :, Z_orig.size(-1) // 2:]  # Characteristic Features

        # Iterate over the target indices and mix the original batch with the target batches
        synthetic_images = []
        for mix_idx in range(self.num_mixes):
            # Get the target characteristics for the current mixing
            Z_target_c = Z_orig_c[target_sample_idxs[mix_idx]][target_patch_idxs[mix_idx]].view(1, 1, -1).expand(Z_orig_c.size(0), Z_orig_c.size(1), -1)

            # Create the synthetic images, pass them through the encoder and split the latent representations
            X_synth = Mixing.patch_wise_image_synthesis(Z_orig_a, Z_target_c, in_channel=self.in_channel,
                                                        patch_size=self.patch_size)
            synthetic_images.append(X_synth)

        # Stack the synthetic images
        if len(synthetic_images) > 0:
            synthetic_images = torch.vstack(synthetic_images)

        # return the losses and the reconstructions
        return synthetic_images

    def mix(self, X: torch.Tensor, target_indices: str):
        """
        Create synthetic images by intermixing all images within the given batch

        :param X: Batch of input images [B, C, H, W].
        :param target_indices: Patch indices to be used for the mixing.
        :return: Synthetic images through intermixing anatomy and characteristics.
        """

        # Run the original input batch through the encoder and split the latent representations into anatomical and
        # characteristic features
        Z = self.encoder(X)  # ([B, P + 1, E] for ViT)
        Z_a = Z[:, :, :Z.size(-1) // 2]  # Anatomical Features
        Z_c = Z[:, :, Z.size(-1) // 2:]  # Characteristic Features

        # Mix the anatomical and characteristic features and return the resulting synthetic images
        return Mixing.mix(Z_a, Z_c, in_channel=self.in_channel, patch_size=self.patch_size)

