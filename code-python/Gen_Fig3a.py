import numpy as np
import cvxpy as cp
from scipy.linalg import dft, eig, norm
from numpy.fft import fft, ifft
import warnings
import time

warnings.filterwarnings("ignore")

def compute_error(x_est, x_true):
    """
    This function computes the relative error with respect to dihedral group
    (shifts + reflection) times Z_2 (sign change).
    """
    X_true = fft(x_true, axis=0)
    X_est = fft(x_est, axis=0)
    a1 = np.abs(ifft(X_true * X_est, axis=0))  # the abs values takes also the sign change into account
    a2 = np.abs(ifft(X_true * X_est.conj(), axis=0))  # the reflected signal
    max_correlation = np.max([a1, a2])
    err = norm(x_est) ** 2 + norm(x_true) ** 2 - 2 * max_correlation
    err = np.abs(err / norm(x_true) ** 2)  # relative error

    return err

rand_seed = 100
np.random.seed(rand_seed)

# Parameters
L = 20  # signal's length
Ms = np.arange(5, 7) # sparsity
max_iter = 10 # number of iterations

errs = np.zeros((len(Ms), max_iter))
computation_times = np.zeros_like(errs)

F = dft(L)

for iM, M in enumerate(Ms):
    for iter_num in range(max_iter):
        start_time = time.time()
        # generating a binary signal
        ind_true = np.random.permutation(L)

        ind_true = ind_true[:M]
        ind_true = np.mod(ind_true - ind_true[0], L)  # + 1  # the first entry is forced to be one
        x_true = np.zeros((L, ))
        x_true[ind_true] = 1

        # measurements
        y = np.abs(F @ x_true) ** 2

        # Solving the SDP
        X = cp.Variable((L, L), symmetric=True)

        Constraints = [
            X >> 0,
            X[ :, :] >= 0,
            cp.trace(X) == M,
            cp.diag(X) == X[:, 0],
            X[0, 0] == 1]
        for i in range(L):
            f = F[i, :].reshape(1, -1)
            Constraints += [cp.trace((np.conj(f).T @ f) @ X) == y[i]]

        R = np.random.randn(L, L)
        Obj = cp.Minimize(cp.trace(X @ R))
        Problem = cp.Problem(objective=Obj, constraints=Constraints)
        Problem.solve(verbose=False)

        # extracting the leading eigenvector:
        eig_val, eig_vec = eig(X.value)
        ind_max = np.abs(eig_val).argmax()
        x_est = np.real(eig_vec[:, ind_max])
        x_est = np.sqrt(eig_val[ind_max].real) * x_est

        x_est = np.abs(np.round(x_est))
        errs[iM, iter_num] = compute_error(x_est, x_true)
        end_time = time.time()
        computation_times[iM, iter_num] = end_time - start_time

a = 0


