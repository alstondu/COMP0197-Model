import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from network_pt import ViT
from mixup import MixUp
from PIL import Image
import os

def montage_visualization(images, nrow, filename):
    """
    Visualize a montage of images using Pillow, arranging them in a grid.
    Directly integrates inverse normalization for visualization.
    Args:
        images (torch.Tensor): Tensor of images to visualize.
        nrow (int): Number of images in each row of the grid.
        filename (str): Filename for saving the image.
    """
    # Performe inverse normalization
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))],
        std=[1/s for s in (0.2023, 0.1994, 0.2010)])

    images = torch.stack([inv_normalize(img) for img in images])  # Inverse normalization
    
    # Calculate grid size
    ncol = nrow
    
    # Create the grid canvas
    grid_img = Image.new('RGB', (images.shape[2] * ncol, images.shape[3] * nrow))
    # Paste images into the canvas
    for i in range(nrow):
        for j in range(ncol):
            # get image
            im = images[i*ncol + j, :, :, :]
            # im = im.permute(1, 2, 0)
            im = transforms.ToPILImage()(im)
            grid_img.paste(im, (j * images.shape[2], i * images.shape[3]))
    
    # Save the grid to a file
    file_path = os.path.join(os.getcwd(), filename)
    grid_img.save(file_path)
    print(f"Montage saved to {file_path}")


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


def train(trainloader, method, epoch):
    """ 
    Trains and tests the model.
    
    Args:
        trainloader (torch.utils.data.DataLoader): Training data loader with mixup.
        method (int): Sampling method
        epoch (int): Current epoch.
    """
    # Visualization flag
    visualize_mixup = True
    loss_total = 0

    # Training loop
    net.train()
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = inputs.to(device), labels.to(device)

        # Convert labels to one-hot encoding
        labels_one_hot = F.one_hot(labels, num_classes=10).float().to(device)

        # Apply mixup
        inputs, labels_mixed = mixup.mixup(inputs, labels_one_hot)

        # Visualize MixUp images (once)
        if visualize_mixup and epoch == 0 and i == 0:
            montage_visualization(images=inputs[:16], nrow=4, filename=f"mixup{method}.png")
            visualize_mixup = False  # Disable further visualization to save time

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = net(inputs)

        # Calculate soft cross-entropy loss
        loss = soft_cross_entropy(outputs, labels_mixed)

        # backward + optimize
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    
    loss_average = loss_total / len(trainloader)
        
    return loss_average


def test(net, testloader, device):
    """ 
    Tests the model on the test data.
    
    Args:
        net (torchvision model): Model to test.
        testloader (torch.utils.data.DataLoader): Test data loader.
        device: cpu or cuda
    """
    correct = 0
    total = 0

    # Evaluation loop
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            # Test with the model
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # See correctness
            correct += (predicted == labels).sum().item()
    # Calculate the accuracy for current test epoch
    epoch_accuracy = 100. *(correct / total)
    
    return epoch_accuracy
    
   
if __name__ == '__main__':
    # Check for device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    ## cifar-10 dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=36, shuffle=True, num_workers=8)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=36, shuffle=False, num_workers=8)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Vision transformer
    net = ViT(num_classes=10).to(device)
    # Optimiser
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for method in [1, 2]:
        print("-" * 15 + 'MixUp Method: ', method, "-" * 15)

        # Mixup data augmentation
        mixup = MixUp(alpha=0.4, sampling_method=method)
        for epoch in range(20):
            # Train the model
            loss = train(trainloader=trainloader, method=method, epoch=epoch)
            epoch_accuracy = test(net, testloader, device)
            print(f'Epoch {epoch+1}/{20}, Loss: {loss}')
            print(f'Epoch {epoch+1}/{20}, Accuracy: {epoch_accuracy}')
        # Save the model
        torch.save(net.state_dict(), f'{os.path.dirname(os.path.abspath(__file__))}/vit_mixup_method_{method}.pt')

        # Visualize the result
        images, labels = next(iter(testloader))
        images, labels = images.to(device), labels.to(device)
        images = images[:36]
        outputs = net(images)
        montage_visualization(images, nrow=6, filename=f"result{method}.png")
        _, predicted = torch.max(outputs.data, 1)
        # Print ground-truth and predicted classes for each model
        print(f"\nVisualizing results for method {method}:")
        print(f"Image NO.: Ground-truth / Prediction")
        for i in range(36):
            print(f"Image {i+1}: {classes[labels[i]]} / {classes[predicted[i]]}")