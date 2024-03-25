# network module
# adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch.nn as nn
import torchvision.models as models

class ViT(nn.Module):
    def __init__(self, num_classes):
        """
        Initializes a Vision Transformer (ViT) model for image classification.
        
        Args:
            num_classes (int): The number of output classes. This defines the size of the output layer of the model.
        """
        super(ViT, self).__init__()
        self.model = models.vit_b_32(
            image_size=32,  # 32x32 images for CIFAR-10
            num_classes=num_classes,
            weights=None,  # Do not use pre-trained weights
        )
    
    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): The input tensor containing a batch of images.
        
        Returns:
            The output of the Vision Transformer model for the input batch of images.
        """
        # Pass the input through the Vision Transformer model and return its output.
        return self.model(x)
