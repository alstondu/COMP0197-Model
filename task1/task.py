import torch
from torch.utils.data import TensorDataset, DataLoader
import time

def polynomial_fun(x, w):
    """
    Implements a polynomoial function that takes two input arguments, a weight vector w of size M + 1 
    and a scalar x. 
    The function returns the result of the polynomial function. 
    The polynomial function is vectorized for multiple pairs of scalar input and output, with the same w.

    Args:
        w (torch array): Parameters of the polynomial function.
        x (scalar): Input scalar.

    Returns:
        y (torch array): polynomial result.
    """
    powers = torch.arange(len(w)).reshape(-1, 1)  # Exponents of x, reshape for broadcasting
    x_powers = torch.pow(x, powers) # Compute x^m for each m
    y = torch.matmul(w, x_powers) # Perform element-wise multiplication with w and sum

    return y

import torch

def fit_polynomial_ls(x, t, M):
    """
    Fits a polynomial to the data using the least squares method with torch.lstsq.

    Parameters:
    x (tensor): 1D tensor of data points.
    t (tensor): 1D tensor of target values corresponding to the data points.
    M (int): Degree of the polynomial.

    Returns:
    w (tensor): Optimal weight vector.
    """
    # Create the Vandermonde matrix for x
    X = torch.vander(x, M + 1, increasing=True)
    
    # Solve for the coefficients using the least squares method
    w_opt = torch.linalg.lstsq(X, t).solution[:M+1]

    return w_opt


def fit_polynomial_sgd(x, t, M, lr, minibatch_size, num_epochs):
    """
    Fits a polynomial of degree M to N pairs of x and t using the stochastic minibatch gradient descent method.

    Args:
        x (torch array): Input values.
        t (torch array): Target values.
        M (int): Degree of the polynomial.
        lr (float): Learning rate.
        minibatch_size (int): Size of the minibatch.

    Returns:
        w (torch array): Optimum weights.
    """
    # Define the report interval
    report_interval = num_epochs/50

    # Initialize weights
    w = torch.randn(M+1, requires_grad=True)

    # Create a DataLoader for minibatch processing
    dataset = TensorDataset(x, t)
    data_batch = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

    # Use mean squared error loss
    loss_fn = torch.nn.MSELoss()

    # Use stochastic gradient descent optimizer
    optimizer = torch.optim.SGD([w], lr=lr)

    loss_op = torch.inf

    # Training loop
    for epoch in range(num_epochs):
        for x_batch, t_batch in data_batch:
            loss_batch = 0 # Clear the loss of current batch
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: compute predicted y by passing x to the model
            y_pred = polynomial_fun(x_batch, w) 

            # Compute loss
            loss = loss_fn(y_pred, t_batch)

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

            # Accumulate the loss
            loss_batch += loss.item()

        # Find the optimal weight leading to minimum loss
        if loss_batch < loss_op:
            loss_op = loss_batch
            w_op = w

        # Print loss periodically
        if epoch % report_interval == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    return w_op

def calculate_rmse(true, pred):
    """
    Calculates the root mean square error between the true and predicted values.
    
    Args:
        true (torch.Tensor): The true values.
        pred (torch.Tensor): The predicted values.
    
    Returns:
        rmse (float): The RMSE value.
    """
    power = torch.pow((true - pred), 2)
    mse = torch.mean(power)
    rmse = torch.sqrt(mse)

    return rmse.item()


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
    t_test = y_test + 0.5 * torch.randn_like(y_test)

    #################################################### OBSERVED VS TRUE ####################################################
    print("-" * 15 + 'Observed data' + "-" * 15)
    Diff_std_ob_t, Diff_mean_ob_t = torch.std_mean(t_train - y_train)
    print('Difference in mean: ', Diff_mean_ob_t, '\
         \nDifference in standard deviation: ', Diff_std_ob_t)

    #################################################### LEAST SQUARE #######################################################
    print("-" * 15 + 'Least Square Fitting' + "-" * 15)

    W_ls = [] # Store the weights from least square
    T_ls = [] # Store the least square fitting time

    # Iterate through differen M
    for M in [2,3,4]:
        t_s = time.time() # Record the time before fitting
        w_ls = fit_polynomial_ls(x_train, t_train, M = M) # Weight fitting with ls
        t_e = time.time() # Record the time after fitting
        T_ls.append(t_e - t_s) # Store the fitting time
        W_ls.append(w_ls) # Store the weight

        # Get predicted target values 
        y_train_pred_ls = polynomial_fun(x_train, w_ls)
        y_test_pred_ls = polynomial_fun(x_test, w_ls)

        # Compute difference in mean and std
        Diff_std_train, Diff_mean_train = torch.std_mean(y_train_pred_ls - y_train)
        Diff_std_test, Diff_mean_test = torch.std_mean(y_test_pred_ls - y_test)

        # Report by printing
        print('M = ', M, ':\
              \nTraning data:\
              \n  Difference in mean: ', Diff_mean_train.item(), '\
              \n  Difference in standard deviation: ', Diff_std_train.item(), '\
              \nTest data:\
              \n  Difference in mean: ', Diff_mean_test.item(), '\
              \n  Difference in standard deviation: ', Diff_std_test.item())

    #################################################### SGD ################################################################
    print("-" * 15 + 'SGD Fitting' + "-" * 15)

    W_sgd = [] # Store the weights from SGD
    T_sgd = [] # Store the SGD fitting time

    # Define training parameters
    parameters = {
    2: {'lr': 1e-5, 'mini_batch_size': 4, 'num_epochs': 1000},
    3: {'lr': 1e-8, 'mini_batch_size': 4, 'num_epochs': 2000},
    4: {'lr': 1e-10, 'mini_batch_size': 4, 'num_epochs': 3000}}

    # Iterate through differen M
    for M in [2, 3, 4]:
        # Extract defined parameters
        lr = parameters[M]['lr']
        mini_batch_size = parameters[M]['mini_batch_size']
        num_epochs = parameters[M]['num_epochs']
        t_s = time.time() # Record the time before fitting
        # Weight fitting with SGD
        w_sgd = fit_polynomial_sgd(x_train, t_train, M = M, lr = lr, minibatch_size = mini_batch_size, num_epochs = num_epochs)
        t_e = time.time() # Record the time after fitting
        T_sgd.append(t_e - t_s) # Store the fitting time
        W_sgd.append(w_sgd) # Store the weight
               
        # Get predicted target values 
        y_train_pred_sgd = polynomial_fun(x_train, w_sgd)
        y_test_pred_sgd = polynomial_fun(x_test, w_sgd)

        # Compute difference in mean and std
        Diff_std_train, Diff_mean_train = torch.std_mean(y_train_pred_sgd - y_train)
        Diff_std_test, Diff_mean_test = torch.std_mean(y_test_pred_sgd - y_test)

        # Report by printing
        print('M = ', M, ':\
              \nTraning data:\
              \n  Difference in mean: ', Diff_mean_train.item(), '\
              \n  Difference in standard deviation: ', Diff_std_train.item(), '\
              \nTest data:\
              \n  Difference in mean: ', Diff_mean_test.item(), '\
              \n  Difference in standard deviation: ', Diff_std_test.item())
        
    #################################################### ACCURACY COMPARISON ################################################
    print("-" * 15 + 'Accuracy Comparison' + "-" * 15)

    # The ground truth weights
    W_true = [
    torch.tensor([1, 2, 3], dtype=torch.float),
    torch.tensor([1, 2, 3, 0], dtype=torch.float),
    torch.tensor([1, 2, 3, 0, 0], dtype=torch.float)]
    
    for i, (M, w_ls, w_sgd) in enumerate(zip([2, 3, 4], W_ls, W_sgd)):
        # Extract current ground truth weight
        w_true = W_true[i]

        # Calculate RMSE for weights
        rmse_w_ls = calculate_rmse(w_true, w_ls)
        rmse_w_sgd = calculate_rmse(w_true, w_sgd)

        # Get the predicted test results from both methods
        y_test_pred_ls = polynomial_fun(x_test, w_ls)
        y_test_pred_sgd = polynomial_fun(x_test, w_sgd)

        # Calculate RMSE for the test sets
        rmse_y_ls = calculate_rmse(y_test, y_test_pred_ls)
        rmse_y_sgd = calculate_rmse(y_test, y_test_pred_sgd)

        # Reporting the accuracy comparison
        print(f"M = {M}:")
        print(f"  LS Weight RMSE: {rmse_w_ls}, LS Output RMSE: {rmse_y_ls}")
        print(f"  SGD Weight RMSE: {rmse_w_sgd}, SGD Output RMSE: {rmse_y_sgd}")

    #################################################### SPEED COMPARISON ##################################################
    print("-" * 15 + 'Speed Comparison' + "-" * 15)
    # Reporting the time comparison
    for M, t_ls, t_sgd in zip([2, 3, 4], T_ls, T_sgd):
        print(f"M = {M}:")
        print(f"  Least Squares Fitting Time: {t_ls:.8f} seconds")
        print(f"  SGD Fitting Time: {t_sgd:.8f} seconds")
