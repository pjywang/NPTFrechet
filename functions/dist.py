import numpy as np

def Bures_Wasserstein(A, B):
    """
    Compute the squared Bures-Wasserstein distance between two positive semi-definite matrices A and B.

    Parameters:
    A (numpy.ndarray): First positive semi-definite matrix.
    B (numpy.ndarray): Second positive semi-definite matrix.

    Returns:
    float: The squared Bures-Wasserstein distance between A and B.
    """
    # Ensure symmetry (mutate if not)
    A = (A + A.T) / 2
    B = (B + B.T) / 2

    # Perform closed-form computation if both are 2D correlation matrices
    if A.shape[0] == 2 and B.shape[0] == 2 and np.allclose(np.diag(A), 1) and np.allclose(np.diag(B), 1):
        a = A[0, 1]
        b = B[0, 1]
        return 4 - 2 * np.sqrt((1 + a) * (1 + b)) - 2 * np.sqrt((1 - a) * (1 - b))

    # Compute the square root of A
    sqrt_A = positive_sqrtm(A)

    # Compute the product of the square root of A and B
    product = sqrt_A @ B @ sqrt_A

    # Compute the square root of the product
    sqrt_product = positive_sqrtm(product)

    # Compute the Bures-Wasserstein distance
    bw_distance = np.trace(A) + np.trace(B) - 2 * np.trace(sqrt_product)

    return bw_distance


def positive_sqrtm(A, eps=1e-15):
    """
    Compute the positive square root of a positive semi-definite matrix A.

    Parameters
    ----------
    A : np.ndarray
        A positive semi-definite matrix.

    Returns
    ----------
    np.ndarray: The positive square root of A.
    """
    # Ensure symmetry (mutate; can help with numerical stability)
    A = (A + A.T) / 2

    d, v = np.linalg.eigh(A)
    # Filter small negative eigenvalues that may occur due to floating point errors
    d_sqrt = np.sqrt(np.maximum(d, eps))

    return v @ np.diag(d_sqrt) @ v.T


def sqrt_prod_cov(A, B):
    """
    Computes (AB)^{1/2} using the formula:
        (AB)^{1/2} = A^{1/2}(A^{1/2}BA^{1/2})^{1/2}A^{-1/2}
    This is used to compute the geodesic between A and B.
        
    Input: Two symmetric positive definite matrices A and B
    """
    A_sqrt = positive_sqrtm(A)
    return A_sqrt @ positive_sqrtm(A_sqrt @ B @ A_sqrt) @ np.linalg.inv(A_sqrt)


def opt_transport(A, B):
    """
    Optimal transport from A to B, which is the geometric mean of A^{-1} and B
    """
    A_sqrt = positive_sqrtm(A)
    A_sqrt_inv = np.linalg.inv(A_sqrt)
    return A_sqrt_inv @ positive_sqrtm(A_sqrt @ B @ A_sqrt) @ A_sqrt_inv
