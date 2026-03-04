import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.

    Forward: p = _sigmoi(z) with z = X @ w + b

    Gradient: 
    + The gradient of L w.r.t w = 1/N * X^T * (p-y)
    + The gradient of L w.r.t b = 1/N * sum(p-y)
    
    Return (w, b).
    """
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    N, D = np.shape(X)

    # Initialize parameters
    w = np.zeros(D)
    b = 0.0

    for _ in range(steps):

        # Forward pass
        z = X @ w + b # (N,)
        p = _sigmoid(z) 

        # Compute gradients
        dz = p - y # (N,)
        dw = (1 / N) * (X.T @ dz) # (D,)
        db = (1 / N) * np.sum(dz) # scalar

        # Update parameters
        w = w - lr * dw
        b = b - lr * db

    b = float(b)

    return w, b