import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """

    A = np.array(A)
    
    M, N = np.shape(A)

    A_transpose = np.zeros((N,M))

    for i in range(M):
        for j in range(N):
            A_transpose[j,i] = A[i,j]

    return A_transpose
