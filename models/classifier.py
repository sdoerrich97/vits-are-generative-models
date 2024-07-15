"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Implementation of a classifier with varying backbones.

@author: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
"""

# Import packages
import timm
import torch.nn as nn

class SharpClassifier(nn.Module):
    """
    unORANIC#'s classifier with a timm backbone.
    """

    def __init__(self, backbone: str = 'densenet121', num_classes: int = 2):
        """
        :param backbone: Name of the backbone architecture to use.
        :param num_classes: Number of classes to predict.
        """

        # Initialize the parent constructor
        super().__init__()

        self.model = timm.create_model(backbone, pretrained=True, num_classes=num_classes)

    def forward(self, X):
        """
        Forward pass of the classifier.

        :param X: Batch of input images X.
        :return: Prediction.
        """

        return self.model(X)