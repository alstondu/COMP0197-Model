import torch
import torch.nn.functional as F
from task import polynomial_fun
from torch.utils.data import TensorDataset, DataLoader

def fit_polynomial_sgd_M(x, t, M_max, lr, minibatch_size, num_epochs):
    """
    Fits a polynomial of degree less than M_max to N pairs of x and t using the stochastic minibatch gradient descent method.

    Args:
        x (torch array): Input values.
        t (torch array): Target values.
        M_max (int): Maximum degree of the polynomial.
        lr (float): Learning rate.
        minibatch_size (int): Size of the minibatch.
        num_epochs (int): The number of epochs

    Returns:
        w_op (torch array): Optimum weights.
    """
    # Define the report interval
    report_interval = num_epochs/50

    M = torch.FloatTensor([M_max]).requires_grad_(True)
    # Initialize weights
    w = torch.randn(M_max + 1, requires_grad=True)

    # Create a DataLoader for minibatch processing
    dataset = TensorDataset(x, t)
    data_batch = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

    # Use mean squared error loss
    loss_fn = torch.nn.MSELoss()

    # Use stochastic gradient descent optimizer
    optimizer = torch.optim.SGD((w,M), lr=lr)

    loss_op = torch.inf

    # Training loop
    for epoch in range(num_epochs):
        for x_batch, t_batch in data_batch:
            loss_batch = 0 # Clear the loss of current batch

            w_mask = (F.gelu(M - torch.arange(M_max + 1).float())).clamp(0,1)
            w_maksed = w * w_mask

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: compute predicted y by passing x to the model
            y_pred = polynomial_fun(x_batch, w_maksed)

            # Compute loss
            loss = loss_fn(y_pred, t_batch)

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Gradient clipping to prevent the exploding gradient
            torch.nn.utils.clip_grad_norm_(w, max_norm=1)
            torch.nn.utils.clip_grad_norm_(M, max_norm=1)

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

            # Accumulate the loss
            loss_batch += loss.item()

        # Find the optimal weight leading to minimum loss
        if loss_batch < loss_op:
            loss_op = loss_batch
            w_op = w_maksed
    
        # Print loss periodically
        if epoch % report_interval == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Truncate the weight
    w_op = w_op[w_op.abs() > 1e-4]

    return w_op


if __name__ == '__main__':

    #################################################### DATA GENERATION ####################################################
    w_g = torch.tensor([1, 2, 3], dtype=torch.float)
    # Generate the training set
    x_train = torch.linspace(-20, 20, 20, dtype=torch.float)
    y_train = polynomial_fun(x_train, w_g)  
    t_train = y_train + 0.5 * torch.randn_like(y_train)

    # Generate the test set
    x_test = torch.linspace(-20, 20, 10, dtype=torch.float)
    y_test = polynomial_fun(x_test, w_g)

    #################################################### SGD ################################################################
    w_sgd = fit_polynomial_sgd_M(x_train, t_train, M_max=10, lr = 0.01, minibatch_size = 1, num_epochs = 10000)

    # Get predicted target values 
    y_train_pred_sgd = polynomial_fun(x_train, w_sgd)
    y_test_pred_sgd = polynomial_fun(x_test, w_sgd)

    # Compute difference in mean and std
    Diff_std_train, Diff_mean_train = torch.std_mean(y_train_pred_sgd - y_train)
    Diff_std_test, Diff_mean_test = torch.std_mean(y_test_pred_sgd - y_test)

    # Report by printing
    print('weight = ', w_sgd.tolist(),'\
        \nTraning data:\
        \n  Difference in mean: ', Diff_mean_train.item(), '\
        \n  Difference in standard deviation: ', Diff_std_train.item(), '\
        \nTest data:\
        \n  Difference in mean: ', Diff_mean_test.item(), '\
        \n  Difference in standard deviation: ', Diff_std_test.item())
