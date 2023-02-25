"""
Translated From Matlab Scripts at: https://github.com/TamirBendory/sparseMRA
Translater: Jonathan Patao
Date: 18 Feb 2023
"""

import numpy as np



def generate_sparse_signal(L, p):
    rand_vec = np.random.rand(L, 1)
    x = np.array(rand_vec > (1 - p), dtype=float)

    return x


def generate_observations(x, M, sigma, noisetype='Gaussian'):
    """
    Given a signal x of length N, generates a matrix X of size N x M such
    that each column of X is a randomly, circularly shifted version of x with
    i.i.d. Gaussian noise of variance sigma^2 added on top. If x is complex,
    the noise is also complex.
    If noisetype is set to 'uniform' instead of 'Gaussian' (or instead of
    being omitted), the noise is uniformly distributed in a centered interval
    such that the variance is sigma^2. (For real signals only.)
    The optional second output, shifts, contains the true shifts that affect
    the individual columns of X: M integers between 0 and N-1.
    https://arxiv.org/abs/1705.00641
    https://github.com/NicolasBoumal/MRA
    """

    x = x.reshape(-1)
    N = len(x)

    X = np.zeros((N, M), dtype=int)
    shifts = np.random.randint(N, size=(M, 1))
    for m in range(M):
        X[:, m] = np.roll(x, shifts[m])

    if noisetype.lower() == 'gaussian':
        if np.any(np.iscomplex(x)):
            X = X + sigma * (np.random.randn(N, M) + 1j * np.random.randn(N, M)) / np.sqrt(2)
        else:
            X = X + sigma * np.random.randn(N, M)
    elif noisetype.lower() == 'uniform':
        if np.any(np.iscomplex(x)):
            raise ValueError('Uniform complex noise not supported yet.')
        else:
            X = X + sigma * (np.random.randn(N, M) - 0.5) * np.sqrt(12)
    else:
        raise TypeError("Noise type can be 'Gaussian' or 'uniform'.")

    return X, shifts

def relative_error(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    L = len(x)
    L_errs = np.zeros((L, ))
    for l in range(L):
        x_est_shifted = np.roll(x_hat, l)
        L_errs[l] = np.linalg.norm(x_est_shifted - x, ord=2)
    mse_error = np.min(L_errs) / np.linalg.norm(x, ord=2)
    return mse_error

def EM_iteration(x, fftX, sqnormX, sigma, p):
    """
    Execute one iteration of EM with current estimate of the DFT of the
    signal given by fftx, and DFT's of the observations stored in fftX, and
    squared 2-norms of the observations stored in sqnormX, and noise level
    sigma.
    """
    fftx = np.fft.fft(x)
    C = np.real(
        np.fft.ifft(
            fftx.reshpe(-1, 1).conj() * fftX, axis=0))
    T = (2 * C - sqnormX) / (2 * (sigma ** 2))
    L, n = fftX
    S = x.sum()
    post_value = np.sum(np.log(np.sum(np.exp(T), axis=0))) + S * np.log(p) + (L - S) * np.log10(1 - p)

    T = T - np.max(T, axis=0)
    W = np.exp(T)
    W = W / np.sum(W, axis=0)
    fftx_new = np.mean(np.fft.fft(W, axis=0).conj() * fftX, axis=1)
    x_new = np.real(np.fft.ifft(fftx_new)) + 2 * np.log(p / (1 - p)) * (sigma ** 2) / n

    return x_new, W, post_value


def MRA_EM(X: np.ndarray, sigma, max_iter, p, x=None, tol=1e-5):
    """
    Expectation maximization algorithm for multireference alignment.
    X: data (each column is an observation)
    sigma: noise standard deviation affecting measurements
    x: initial guess for the signal (optional)
    tol: EM stops iterating if two subsequent iterations are closer than tol
         in 2-norm, up to circular shift (default: 1e-5).
    batch_niter: number of batch iterations to perform before doing full data
                 iterations if there are more than 3000 observations
                 (default: 3000.)
        May 2017
    https://arxiv.org/abs/1705.00641
    https://github.com/NicolasBoumal/MRA
    """

    N, M = X.shape

    if x is None:
        if np.any(np.iscomplex(X)):
            x = np.random.randn(N, 1) + 1j*np.random.randn(N, 1)
        else:
            x = np.random.randn(N, 1)

    x = x.flatten()
    assert len(x) == N, 'Initial guess x must have length N.'

    x_est = x

    # Precomputations on the observations
    fftX = np.fft.fft(X, axis=0)
    sqnormX = np.tile(np.sum(np.absolute(X) ** 2, axis=0), (N, 1))

    # In any case, finish with full passes on the data
    full_niter = max_iter
    post_value = np.zeros((max_iter, 1))
    EM_time = None

    for _iter in range(full_niter):
        x_new, W, post_value[_iter] = EM_iteration(x_est, fftX, sqnormX, sigma, p)

        if relative_error(x_new, x_est) < tol:
            break

        x_est = x_new

    EM_iter = _iter
    post_value = post_value[:EM_iter]
    x = x_est

    return x, W, EM_iter, EM_time, post_value




