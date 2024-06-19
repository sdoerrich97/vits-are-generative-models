"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Helpers for the model architectures.

@references:
mae: https://github.com/facebookresearch/mae

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

import torch
import numpy as np


# ======================================================================================================================
# General Model Helpers
# ======================================================================================================================
class Patches:
    """
    Patches utils.
    """

    @staticmethod
    def patchify(X, in_channel, patch_size):
        """
        Split each image of the input batch into patches.

        :param in_channel: Channel dimension of the input.
        :param X: Input batch of images of shape: (N, 3, H, W).
        :param patch_size: Patch size.

        :return: Patchified batch of images of shape: (N, P, patch_size**2 *3).
        """

        # Assert that the width and height of the input images is the same and the image can be split into
        # non-overlapping patches
        assert X.shape[2] == X.shape[3] and X.shape[2] % patch_size == 0

        # Split the input images into equally sized patches
        h = w = X.shape[2] // patch_size
        X = X.reshape(shape=(X.shape[0], in_channel, h, patch_size, w, patch_size))

        X = torch.einsum('nchpwq->nhwpqc', X)
        X = X.reshape(shape=(X.shape[0], h * w, patch_size ** 2 * in_channel))  # [B, P, PS * PS * C]

        # Return the patchified batch
        return X

    @staticmethod
    def unpatchify(X, in_channel, patch_size):
        """
        Redo a patched input to obtain the original batch of images.

        :param X: Patchified batch of images of shape: (N, L, patch_size**2 * 3) or (N, L, C, patch_size, patch_size).
        :param in_channel: Channel dimension of the input.
        :param patch_size: Patch size.
        :param X: Patchified batch of images of shape: (N, L, patch_size**2 * 3).

        :return: Original un-patchified input batch of shape: (N, 3, H, W).
        """

        # Assert the correct height and width of the original input size
        h = w = int(X.shape[1] ** .5)
        assert h * w == X.shape[1]

        # Reshape the patchified input batch to the original image shape
        if len(X.shape) == 3:
            X = X.reshape(shape=(X.shape[0], h, w, patch_size, patch_size, in_channel))

        elif len(X.shape) == 5:
            X = X.reshape(shape=(X.shape[0], h, w, X.shape[3], X.shape[4], X.shape[2]))

        X = torch.einsum('nhwpqc->nchpwq', X)
        X = X.reshape(shape=(X.shape[0], in_channel, h * patch_size, h * patch_size))

        # Return the original image-sized batch
        return X


# ======================================================================================================================
# Mixing-related Model Helpers
# ======================================================================================================================
class Mixing:
    """
    Mixing utils.
    """

    @staticmethod
    def patch_wise_image_synthesis(Z_a: torch.Tensor, Z_c: torch.Tensor, in_channel: int = 3, patch_size: int = 16):
        """
        Synthesize an image S from the given anatomical and characteristic feature embeddings via Matrix-Multiplication
        in a patch-wise manner.

        :param Z_a: Anatomical feature embedding [B, P + 1, E].
        :param Z_c: Characteristic feature embedding [B, P + 1, E].
        :param in_channel: Number of input channels.
        :param patch_size: Patch size.
        :return: Synthetic image S with anatomy Z_a and characteristic Z_c.
        """

        # # Remove the CLS token
        Z_a = Z_a[:, 1:, :]
        Z_c = Z_c[:, 1:, :]

        # Combine anatomical and characteristic features for each patch of each sample in the batch using matrix multiplication
        Z_a = Z_a.view(Z_a.size(0), Z_a.size(1), in_channel, patch_size, -1)  # [B, P, C, PS, L]
        Z_c = Z_c.view(Z_c.size(0), Z_c.size(1), in_channel, -1, patch_size)  # [B, P, C, L, PS]

        S = torch.matmul(Z_a, Z_c)  # [B, P, C, PS, PS]

        # Reshape the tensor to get the synthetic image
        S = Patches.unpatchify(S, in_channel, patch_size)  # [B, C, H, W]

        # Return the synthetic image
        return S

    @staticmethod
    def mix(Z_a: torch.Tensor, Z_c: torch.Tensor, in_channel: int = 3, patch_size: int = 16):
        """
        Create synthetic images by intermixing all anatomical embeddings Z_a with all characteristic embeddings Z_c.

        :param Z_a: Anatomical feature embeddings ([B, P + 1, E] for ViT).
        :param Z_c: Characteristic feature embeddings ([B, P + 1, E] for ViT).
        :param in_channel: Number of input channels.
        :param patch_size: Patch size.

        :return: Synthetic images through intermixing anatomy and characteristics.
        """
        # Extract the batch dimension
        B = Z_a.size(0)

        # Reconstruct the original images and calculate the reconstruction loss
        synthetic_images = []
        for b in range(B):
            # Get the target characteristics for the current mixing
            Z_target_c = Z_c[b].unsqueeze(0).expand(Z_c.size(0), Z_c.size(1), -1)

            # Create the synthetic images
            X_synth = Mixing.patch_wise_image_synthesis(Z_a, Z_target_c, in_channel, patch_size)

            # Store the synthetic images
            synthetic_images.append(X_synth)

        # return the synthetic images
        return synthetic_images


# ======================================================================================================================
# ViT Model Helpers
#   - References:
#       - MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# ======================================================================================================================
class PositionEmbedding:
    """
    Position embedding utils.

    References:
        - MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
        - Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
        - MoCo v3: https://github.com/facebookresearch/moco-v3
        - DeiT: https://github.com/facebookresearch/deit
    """

    @staticmethod
    def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token=False):
        """
        2D sine-cosine position embedding.

        :param embed_dim: Embedding dimension.
        :param grid_size: Grid height and width.

        :return: Position embedding [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """

        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = PositionEmbedding.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

        return pos_embed

    @staticmethod
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        """
        Get the 2d sin-cosine position embedding from the provided grid.

        :param embed_dim: Embedding dimension.
        :param grid:

        :return: 2d sin-cosine position embedding
        """

        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = PositionEmbedding.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = PositionEmbedding.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)

        return emb

    @staticmethod
    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        """
        Get the 1d sin-cosine position embedding from the provided grid.

        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)

        :return: 1d sin-cosine position embedding.
        """

        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=float)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out) # (M, D/2)
        emb_cos = np.cos(out) # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    @staticmethod
    def interpolate_pos_embed(model, checkpoint_model):
        """
        Interpolate position embeddings for high-resolution.

        :param model:
        :param checkpoint_model:
        :return:
        """

        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed
