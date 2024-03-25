import torch
import torch.optim as optim
import torch.nn.functional as F
from network_pt import ViT
from mixup import MixUp
from utils import*
import os
import time

def calculate_mcc(tp, tn, fp, fn):
    """
    Calculate the Matthews Correlation Coefficient for multi-class classification using PyTorch tensors.
    
    Args:
        tp (torch.Tensor): Tensor of true positives per class.
        tn (torch.Tensor): Tensor of true negatives per class.
        fp (torch.Tensor): Tensor of false positives per class.
        fn (torch.Tensor): Tensor of false negatives per class.
    
    Returns:
        float: The average MCC across all classes.
    """
    # Ensure no division by zero
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    denominator[denominator == 0] = 1  # Replace 0s with 1 to avoid division by zero
    
    mcc_per_class = (tp * tn - fp * fn) / denominator
    mcc_per_class[torch.isnan(mcc_per_class)] = 0  # Convert NaNs to 0 for undefined MCC values
    
    return mcc_per_class.mean().item()


def vali(net, valiloader, device):
    """ 
    Tests the model on the validation set.
    
    Args:
        net (torchvision model): Model to validate.
        valiloader (torch.utils.data.DataLoader): Validation data loader.
        device: cpu or cuda
    Returns:
        accu_vali (float): validation accuracy
        loss_vali (float): Validation loss.
        MCC_vali (float): Validation MCC score.
        speed_vali (float): Validation speed.
    """
    tp = torch.zeros(10, device=device)
    tn = torch.zeros(10, device=device)
    fp = torch.zeros(10, device=device)
    fn = torch.zeros(10, device=device)
    loss_vali = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    # Start time measurement
    start_time = time.time()

    # Evaluation loop
    net.eval()
    with torch.no_grad():
        for images, labels in valiloader:
            images, labels = images.to(device), labels.to(device)
            # Test with the model
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss_vali += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # See correctness
            correct += (predicted == labels).sum().item()

            # Update true positives, false positives, false negatives, and true negative for MCC score calculation
            for i in range(10):
                tp[i] += ((predicted == i) & (labels == i)).sum().item()
                tn[i] += ((predicted != i) & (labels != i)).sum().item()
                fp[i] += ((predicted == i) & (labels != i)).sum().item()
                fn[i] += ((predicted != i) & (labels == i)).sum().item()
        
        # Calculate elapsed time and training speed
        elapsed_time = time.time() - start_time
        speed_vali = len(trainloader) / elapsed_time  # Batches per second

        MCC_vali = calculate_mcc(tp, tn, fp, fn)
        # Calculate the accuracy for current test epoch
        loss_vali = loss_vali / len(valiloader)
        accu_vali = 100. *(correct / total)
    
    return accu_vali, loss_vali, MCC_vali, speed_vali


def train(trainloader, method, epoch):
    """ 
    Trains and tests the model.
    
    Args:
        trainloader (torch.utils.data.DataLoader): Training data loader with mixup.
        method (int): Sampling method
        epoch (int): Current epoch.
    Returns:
        loss_train (float): Training loss.
        MCC_train (float): Training MCC score.
        speed_train (float): Training speed.
    """
    loss_train = 0
    tp = torch.zeros(10, device=device)
    tn = torch.zeros(10, device=device)
    fp = torch.zeros(10, device=device)
    fn = torch.zeros(10, device=device)

    # Start time measurement
    start_time = time.time()

    # Training loop
    net.train()
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = inputs.to(device), labels.to(device)

        # Convert labels to one-hot encoding
        labels_one_hot = F.one_hot(labels, num_classes=10).float().to(device)

        # Apply mixup
        inputs, labels_mixed = mixup.mixup(inputs, labels_one_hot)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = net(inputs)

        # Calculate soft cross-entropy loss
        loss = soft_cross_entropy(outputs, labels_mixed)
        loss_train += loss.item()

        # backward + optimize
        loss.backward()
        optimizer.step()
        
        # Convert soft labels back to hard labels for MCC calculation
        _, labels_hard = labels_mixed.max(dim=1)
        _, predicted = outputs.max(dim=1)

        # Update TP, TN, FP, FN for each class
        for c in range(10):
            tp[c] += ((predicted == c) & (labels_hard == c)).sum().item()
            tn[c] += ((predicted != c) & (labels_hard != c)).sum().item()
            fp[c] += ((predicted == c) & (labels_hard != c)).sum().item()
            fn[c] += ((predicted != c) & (labels_hard == c)).sum().item()

    # Calculate elapsed time and training speed
    elapsed_time = time.time() - start_time
    speed_train = len(trainloader) / elapsed_time # Batches per second

    MCC_train = calculate_mcc(tp, tn, fp, fn)
    loss_train = loss_train / len(trainloader)
        
    return loss_train, MCC_train, speed_train


def test(net, testloader, device):
    """ 
    Tests the model on the test data.
    
    Args:
        net (torchvision model): Model to test.
        testloader (torch.utils.data.DataLoader): Test data loader.
        device: cpu or cuda
    Returns:
        accu_test (float): test accuracy
        loss_test (float): test loss.
        MCC_test (float): test MCC score.
        speed_test (float): test speed.
    """
    tp = torch.zeros(10, device=device)
    tn = torch.zeros(10, device=device)
    fp = torch.zeros(10, device=device)
    fn = torch.zeros(10, device=device)
    loss_test = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    # Start time measurement
    start_time = time.time()

    # Evaluation loop
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            # Test with the model
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss_test += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # See correctness
            correct += (predicted == labels).sum().item()
            # Update true positives, false positives, false negatives, and true negative for MCC score calculation
            for i in range(10):
                tp[i] += ((predicted == i) & (labels == i)).sum().item()
                tn[i] += ((predicted != i) & (labels != i)).sum().item()
                fp[i] += ((predicted == i) & (labels != i)).sum().item()
                fn[i] += ((predicted != i) & (labels == i)).sum().item()
        
        # Calculate elapsed time and training speed
        elapsed_time = time.time() - start_time
        speed_test = len(trainloader) / elapsed_time  # Batches per second

        MCC_test = calculate_mcc(tp, tn, fp, fn)
        # Calculate the accuracy for current test epoch
        loss_test = loss_test / len(testloader)
        accu_test = 100. *(correct / total)

    return accu_test, loss_test, MCC_test, speed_test
    
   
if __name__ == '__main__':
    # Check for device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    trainloader, testloader, valiloader  = load_dataset()

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Vision transformer
    net = ViT(num_classes=10).to(device)
    # Optimiser
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for method in [1, 2]:
        print("-" * 15 + 'MixUp Method: ', method, "-" * 15)
        print(f'| Epoch \t | Train Loss \t | Train MCC \t | Train Speed \t | Test Acc \t | Test Loss \t | Test MCC \t | Test Speed \t | Vali Acc \t | Vali Loss \t | Vali MCC \t | Vali Speed \t') 
        # Mixup data augmentation
        mixup = MixUp(alpha=0.4, sampling_method=method)
        for epoch in range(20):
            # Train the model
            loss_train, MCC_train, speed_train= train(trainloader=trainloader, method=method, epoch=epoch)
            accu_test, loss_test, MCC_test, speed_test = test(net, testloader, device)
            accu_vali, loss_vali, MCC_vali, speed_vali = vali(net, testloader, device)
            print(f'|\t {epoch+1} \t | {loss_train:.3f} \t | {MCC_train:.3f} \t | {speed_train:.3f} b/s \t | {accu_test:.3f}% \t | {loss_test:.3f} \t | {MCC_test:.3f} \t | {speed_test:.3f} b/s \t | {accu_vali:.3f}% \t | {loss_vali:.3f} \t | {MCC_vali:.3f} \t | {speed_vali:.3f} b/s \t')
        # Save the model
        torch.save(net.state_dict(), f'{os.path.dirname(os.path.abspath(__file__))}/Ablation_method_{method}.pt')

        
        