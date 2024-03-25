import torch
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split
from PIL import Image
import torch.nn.functional as F

def load_dataset(batch_size=128):
    """ 
    Loads the CIFAR10 dataset and splite based on requirment.

    Args:
        batch_size (int): Batch size to use.
    Returns:
        train_loader: Training data loader.
        test_loader: Test data loader.
        vali_loader: Validation data loader.
    """
    ## cifar-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    
    # Load the dataset and split into train, test, and validation sets
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    n_samples = len(dataset)
    n_dev = int(0.8 * n_samples)
    n_test =int(0.2 * n_samples)
    n_train = int(0.9 * n_dev)
    n_vali = n_dev - n_train
    train_dataset, test_dataset, vali_dataset = random_split(dataset, [n_train, n_test, n_vali])

    # Create data loaders for the train, validation and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    vali_loader = DataLoader(vali_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, test_loader, vali_loader


def soft_cross_entropy(preds, soft_targets):
    """
    Computes the Soft Cross-Entropy Loss between predictions and soft targets.
    
    Args:
        preds (torch.Tensor): The raw predictions from the model (logits).
        soft_targets (torch.Tensor): The soft targets (one-hot encoded or soft labels).
        
    Returns:
        torch.Tensor: The computed loss.
    """
    log_softmax_preds = F.log_softmax(preds, dim=1)

    return torch.mean(torch.sum(- soft_targets * log_softmax_preds, dim=1))
