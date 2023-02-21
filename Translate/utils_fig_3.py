import numpy as np
import cvxpy as cp
from scipy.linalg import dft, eigh, norm
from numpy.fft import fft, ifft



def compute_error(x_est, x_true):
    """
    this function computes the relative error with respect to dihedral group
    (shifts + reflection) times Z_2 (sign change).
    """

    X_true = fft(x_true, axis=0)
    X_est = fft(x_est, axis=0)
    a1 = np.abs(ifft(X_true * X_est, axis=0))  # the abs values takes also the sign change into account
    a2 = np.abs(ifft(X_true * X_est.conj(), axis=0))  # the reflected signal
    max_correlation = np.max([a1, a2])
    err = norm(x_est) ** 2 + norm(x_true) ** 2 - 2 * max_correlation
    err = err / norm(x_true) ** 2  # relative error

    return err


def patao_assert(x1, x2):
    assert np.all(np.round(x1, 5) == np.round(x2, 5))
