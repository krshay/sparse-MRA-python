import numpy as np
import cvxpy as cp
from scipy.linalg import dft, eig, norm
from numpy.fft import fft, ifft
import warnings
import time
import matplotlib.pyplot as plt
from scipy.io import savemat

import scienceplots
plt.style.use('science')

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

rand_seed = 1
np.random.seed(rand_seed)

Ls = [20, 30, 40, 50, 60]  # signal's length
Ms = np.arange(2, 13) # sparsity
max_iter = 20 # number of iterations

errs = np.zeros((len(Ls), len(Ms), max_iter))
computation_times = np.zeros_like(errs)

for iL, L in enumerate(Ls):
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
            errs[iL, iM, iter_num] = compute_error(x_est, x_true)
            end_time = time.time()
            computation_times[iL, iM, iter_num] = end_time - start_time

mean_errs = np.mean(errs, 2)
mean_computation_times = np.mean(computation_times, 2)
fig = plt.figure()
plt.plot(Ms, mean_errs.T)
plt.ylabel('error')
plt.xlabel('M')
plt.xticks(Ms)
plt.legend(['L = 20', 'L = 30', 'L = 40', 'L = 50', 'L = 60'])
fig.tight_layout()
plt.savefig('SPD_errs.pdf')

fig = plt.figure()
plt.plot(Ms, mean_computation_times.T)
plt.ylabel('Computation time [secs]')
plt.xlabel('M')
plt.xticks(Ms)
plt.legend(['L = 20', 'L = 30', 'L = 40', 'L = 50', 'L = 60'])
fig.tight_layout()
plt.savefig('SPD_time.pdf')

savemat('mean_errs.mat', {'mean_errs': mean_errs})
savemat('mean_computation_times.mat', {'mean_computation_times': mean_computation_times})
