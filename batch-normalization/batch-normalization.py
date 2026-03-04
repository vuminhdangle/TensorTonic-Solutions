import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).

    x: input array
    gamma: scale parameter
    beta: shift parameter
    """

    # Make sure all the inputs are numpy arrays
    x = np.asarray(x, dtype = float)
    gamma = np.asarray(gamma, dtype = float)
    beta = np.asarray(beta, dtype = float)
    
    if x.ndim == 2:
        # (N, D) => normalize over axis = 0
        mean = np.mean(x, axis = 0)
        var = np.var(x, axis = 0)

        x_hat = (x - mean) / np.sqrt(var + eps)

        out = gamma * x_hat + beta
    else:
        # (N, C, H, W) => normalize over axes (0, 2, 3)
        mean = np.mean(x, axis = (0, 2, 3), keepdims = True)
        var = np.var(x, axis = (0, 2, 3), keepdims = True)

        x_hat = (x - mean) / np.sqrt(var + eps)

        # Reshape gamma, beta before broadcasting

        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)

        out = gamma * x_hat + beta

    return out