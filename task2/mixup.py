import numpy as np
import torch

class MixUp:
    def __init__(self, alpha, sampling_method=1):
        """
        Initialize the MixUp data augmentation class.
        
        Args:
            alpha (float): The alpha parameter for the Beta distribution used to mix samples. 
                           A higher alpha makes the mixup images more like one of the input images.
            sampling_method (int): Determines the method of lambda sampling. 
                                   1 for sampling from a Beta distribution, 
                                   2 for uniform sampling.
        """
        # Store the alpha value and sampling method for later use
        self.alpha = alpha
        self.sampling_method = sampling_method
        
        # Seed the numpy random number generator for reproducible lambda values
        np.random.seed(42)

    def mixup(self, x, y):
        """
        Applies MixUp augmentation to a batch of images and labels.
        
        Args:
            x (torch.Tensor): A batch of images.
            y (torch.Tensor): Corresponding labels for the batch of images.
            
        Returns:
            mixed_x (torch.Tensor): The batch of mixed images.
            mixed_y (torch.Tensor): The batch of mixed labels.
        """
        # Determine lambda value based on the chosen sampling method
        if self.sampling_method == 1:
            # Sampling lambda from a Beta distribution
            lam = np.random.beta(self.alpha, self.alpha)
        elif self.sampling_method == 2:
            # Sampling lambda uniformly
            lam = np.random.uniform(0, 1)

        # Get the size of the batch
        batch_size = x.shape[0]
        
        # Generate a random permutation of the batch indices
        index = torch.randperm(batch_size).to(x.device)

        # Mix images. The images are mixed by a weighted sum determined by lambda
        mixed_x = lam * x + (1 - lam) * x[index, :]
        
        # Mix labels in the same way as images
        mixed_y = lam * y + (1 - lam) * y[index]
        
        # Return the mixed images and labels
        return mixed_x, mixed_y
