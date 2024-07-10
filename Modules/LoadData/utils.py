from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from scipy.integrate import odeint

class MyDataset(Dataset):
    def __init__(self, data, id, group):
        self.data = data
        self.id = id
        self.group = group

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id = self.id[idx]
        data_value = self.data[idx]
        data = torch.tensor(data_value, dtype=torch.float32)
        group = self.group[idx]
        return data, id, group

def time_split(T, step=10):
    # 对长序列进行截断以训练模型
    # T: [num_node, num_time, num_feature]
    start = 0
    end = step
    samples = []
    while end <= T.shape[1]:
        samples.append(T[:, start:end, :])
        start += 1
        end += 1
    return samples

def simulate_var(p, T, lag, sparsity=0.2, beta_value=1.0, sd=0.1, seed=0):
    '''
    Simulate data from a VAR model.
    
    Parameters:
    p (int): Number of variables.
    T (int): Length of time series.
    lag (int): Number of lags in the VAR model.
    sparsity (float): Sparsity level of the coefficient matrix.
    beta_value (float): Value of non-zero coefficients in the coefficient matrix.
    sd (float): Standard deviation of the errors.
    seed (int): Seed for the random number generator.
    
    Returns:
    numpy.ndarray: Simulated data.
    numpy.ndarray: Coefficient matrix of the VAR model.
    numpy.ndarray: Ground truth of Granger causality.
    '''
    if seed is not None:
        np.random.seed(seed)
    
    # Set up coefficients and Granger causality ground truth.
    GC = np.eye(p, dtype=int)
    beta = np.eye(p) * beta_value
    num_nonzero = int(p * sparsity) - 1
    
    # Randomly assign non-zero coefficients and Granger causality
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        GC[i, choice] = 1
    
    # Expand the coefficient matrix to include lags
    beta = np.hstack([beta for _ in range(lag)])
    
    # Make the VAR model stationary
    beta = make_var_stationary(beta)
    
    # Generate data.
    burn_in = 100
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))
    X = np.zeros((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]
    
    for t in range(lag, T + burn_in):
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        X[:, t] += errors[:, t-1]
    
    return X.T[burn_in:], beta, GC

def make_var_stationary(beta, radius=0.97):
    '''
    Rescale coefficients of VAR model to make it stable.
    
    Parameters:
    beta (numpy.ndarray): Coefficient matrix of VAR model.
    radius (float): Maximum eigenvalue radius for stability.
    
    Returns:
    numpy.ndarray: Rescaled coefficient matrix.
    '''
    p = beta.shape[0]
    lag = beta.shape[1] // p
    
    # Construct bottom part of coefficient matrix
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    
    # Calculate eigenvalues of the coefficient matrix
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    
    # Check if the VAR model is non-stationary
    nonstationary = max_eig > radius
    
    if nonstationary:
        # Recursively rescale the coefficient matrix
        return make_var_stationary(0.95 * beta, radius)
    else:
        return beta

def lorenz(x, t, F):
    '''Partial derivatives for Lorenz-96 ODE.'''
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i+1) % p] - x[(i-2) % p]) * x[(i-1) % p] - x[i] + F

    return dxdt


def simulate_lorenz_96(p, T, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000,
                       seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(lorenz, x0, t, args=(F,))
    X += np.random.normal(scale=sd, size=(T + burn_in, p))

    # Set up Granger causality ground truth.
    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:], GC
