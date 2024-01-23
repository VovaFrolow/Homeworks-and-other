import numpy as np


def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    # YOUR CODE HERE
    x = np.ones(data.shape[1])
    v = np.ones(data.shape[0]) / np.sqrt(data.shape[1])
    eig_value = v.dot(data.dot(v))
    for i in range(num_steps):
        Av = data.dot(v)
        v_new = Av / np.linalg.norm(Av)
        eigv_new = v_new.dot(data.dot(v_new))
        if np.abs(eig_value - eigv_new) < 1e-9:
            break

        eig_value = eigv_new
        v = v_new

    return float(eigv_new), v_new
