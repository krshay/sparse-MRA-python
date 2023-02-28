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

Ls = [25]  # signals' length
Ms = np.arange(2, 18, 2) # sparsity
max_iter = 20 # number of iterations

errs_MRA = np.zeros((len(Ls), len(Ms), max_iter))
computation_times_MRA = np.zeros_like(errs_MRA)

errs_MTD = np.zeros((len(Ls), len(Ms), max_iter))
computation_times_MTD = np.zeros_like(errs_MTD)

for iL, L in enumerate(Ls):
    F1 = dft(L)
    F2 = dft(2 * L)
    for iM, M in enumerate(Ms):
        for iter_num in range(max_iter):

            # generating a binary signal
            ind_true = np.random.permutation(L)

            ind_true = ind_true[:M]
            ind_true = np.mod(ind_true - ind_true[0], L)  # + 1  # the first entry is forced to be one
            x_true = np.zeros((L, ))
            x_true[ind_true] = 1

            # measurements
            y1 = np.abs(F1 @ x_true) ** 2
            y2 = np.abs(F2 @ np.pad(x_true, (0, L))) ** 2

            start_time = time.time()
            # Solving the SDP
            X1 = cp.Variable((L, L), symmetric=True)

            Constraints = [
                X1 >> 0,
                X1[:, :] >= 0,
                cp.trace(X1) == M,
                cp.diag(X1) == X1[:, 0],
                X1[0, 0] == 1]
            for i in range(L):
                f = F1[i, :].reshape(1, -1)
                Constraints += [cp.trace((np.conj(f).T @ f) @ X1) == y1[i]]

            R1 = np.random.randn(L, L)
            Obj = cp.Minimize(cp.trace(X1 @ R1))
            Problem = cp.Problem(objective=Obj, constraints=Constraints)
            Problem.solve(verbose=False)

            # extracting the leading eigenvector:
            eig_val, eig_vec = eig(X1.value)
            ind_max = np.abs(eig_val).argmax()
            x_est_MRA = np.real(eig_vec[:, ind_max])
            x_est_MRA = np.sqrt(eig_val[ind_max].real) * x_est_MRA

            x_est_MRA = np.abs(np.round(x_est_MRA))
            errs_MRA[iL, iM, iter_num] = compute_error(x_est_MRA, x_true)
            end_time = time.time()
            computation_times_MRA[iL, iM, iter_num] = end_time - start_time

            start_time = time.time()
            # Solving the SDP
            X2 = cp.Variable((2 * L, 2 * L), symmetric=True)

            Constraints = [
                X2 >> 0,
                X2[:, :] >= 0,
                cp.trace(X2) == M,
                cp.diag(X2) == X2[:, 0],
                X2[0, 0] == 1]
            for i in range(L):
                f = F2[i, :].reshape(1, -1)
                Constraints += [cp.trace((np.conj(f).T @ f) @ X2) == y2[i]]

            R2 = np.random.randn(2 * L, 2 * L)
            Obj = cp.Minimize(cp.trace(X2 @ R2))
            Problem = cp.Problem(objective=Obj, constraints=Constraints)
            Problem.solve(verbose=False)

            # extracting the leading eigenvector:
            eig_val, eig_vec = eig(X2.value)
            ind_max = np.abs(eig_val).argmax()
            x_est_MTD = np.real(eig_vec[:, ind_max])
            x_est_MTD = np.sqrt(eig_val[ind_max].real) * x_est_MTD

            x_est_MTD = np.abs(np.round(x_est_MTD))
            errs_MTD[iL, iM, iter_num] = compute_error(x_est_MTD, np.pad(x_true, (0, L)))
            end_time = time.time()
            computation_times_MTD[iL, iM, iter_num] = end_time - start_time

mean_errs_MRA = np.mean(errs_MRA, 2)
mean_computation_times_MRA = np.mean(computation_times_MRA, 2)
mean_errs_MTD = np.mean(errs_MTD, 2)
mean_computation_times_MTD = np.mean(computation_times_MTD, 2)
savemat('mean_errs_MRA.mat', {'mean_errs_MRA': mean_errs_MRA})
savemat('mean_computation_times_MRA.mat', {'mean_computation_times_MRA': mean_computation_times_MRA})
savemat('mean_errs_MTD.mat', {'mean_errs_MTD': mean_errs_MTD})
savemat('mean_computation_times_MTD.mat', {'mean_computation_times_MTD': mean_computation_times_MTD})
fig = plt.figure()
plt.plot(Ms, mean_errs_MRA.T)
plt.plot(Ms, mean_errs_MTD.T)
plt.ylabel('error')
plt.xlabel('M')
plt.xticks(Ms)
plt.legend(['sparse-MRA', 'sparse-MTD'])
fig.tight_layout()
plt.savefig('SPD_errs_MTD.pdf')

fig = plt.figure()
plt.plot(Ms, mean_computation_times_MRA.T)
plt.plot(Ms, mean_computation_times_MTD.T)
plt.ylabel('Computation time [secs]')
plt.xlabel('M')
plt.xticks(Ms)
plt.legend(['sparse-MRA', 'sparse-MTD'])
fig.tight_layout()
plt.savefig('SPD_time_MTD.pdf')
