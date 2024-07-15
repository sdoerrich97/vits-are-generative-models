"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Implementation of the Vision-Transformer encoder.

@references:
mae: https://github.com/facebookresearch/mae
timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
DeiT: https://github.com/facebookresearch/deit

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block

# Import own packages
from ..utils import PositionEmbedding, Patches


class EncoderViT(nn.Module):
    """
    Encoder with Vision Transformer backbone.
    """

    def __init__(self, img_size: int, in_channel: int, patch_size: int, embed_dim: int, depth: int, num_heads: int,
                 mlp_ratio: float, norm_layer: str):
        """
        :param img_size: Width and height of the input image.
        :param in_channel: Channel dimension of the input.
        :param patch_size: Width and height of the patches.
        :param embed_dim: Embedding (latent) dimension of each patch.
        :param depth: Depth.
        :param num_heads: Number of attention heads.
        :param mlp_ratio: Ratio of the hidden dimension compared to the input dimension of the Multi-Layer-Perceptron (MLP) - layer.
        :param norm_layer: Normalization layer.
        """

        # Initialize the parent constructor
        super().__init__()

        # Store the parameters
        self.img_size = img_size
        self.in_channel = in_channel
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer

        # Initialize the normalization layer
        if self.norm_layer == 'nn.LayerNorm':
            self.norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Create the patch embedding
        # self.patch_embed = PatchEmbed(img_size, patch_size, in_channel, embed_dim, bias=True)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channel, embed_dim)

        # Get the number of input patches
        self.num_patches = self.patch_embed.num_patches

        # Create the CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)

        # Create the position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)

        # Create the transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=self.norm_layer)
            for i in range(depth)])

        # Create the normalization layer
        self.norm = self.norm_layer(embed_dim)

        # Initialize the encoder's weights
        self.initialize_weights()

    def get_num_patches(self):
        """Get the number of patches."""

        return self.num_patches

    def initialize_weights(self):
        """
        Initialize the weights of the encoder.
        """

        # Initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = PositionEmbedding.get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize nn.Linear and nn.LayerNorm.

        :param m: Module to initialize
        """

        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, X):
        """
        Forward pass of the encoder.

        :param X: Batch of input images

        :return: Encoded input.
        """

        # Embed patches
        X = self.patch_embed(X)

        # Append cls token
        cls_tokens = self.cls_token.expand(X.shape[0], -1, -1)
        X = torch.cat((cls_tokens, X), dim=1)

        # Add the pos embed
        X = X + self.pos_embed

        # Apply Transformer blocks
        for blk in self.blocks:
            X = blk(X)

        # Apply the normalization
        X = self.norm(X)

        return X
