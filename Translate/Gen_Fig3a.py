import numpy as np
import cvxpy as cp
from scipy.linalg import dft, eigh
from utils_fig_3 import compute_error
# scipy.eigs.sparse.linalg.eigsh

rand_seed = np.random.randint(10000)
print(f'random seed: {rand_seed}')
np.random.seed(rand_seed)

# Parameters
N_vec = np.arange(start=10, stop=81, step=5)  # signal's length
K_vec = np.arange(start=2, stop=21, step=2)  # sparsity
max_iter = 10  # number of trials per (N,K)

# Main Loop
sdp_err = np.zeros((len(N_vec), len(K_vec), max_iter))
X_rank = np.zeros((len(N_vec), len(K_vec), max_iter))

for n in range(len(N_vec)):
    N = N_vec[n]
    F = dft(N)

    for K in range(len(K_vec)):
        for _iter in range(max_iter):
            # geenrating a binary signal
            ind_true_1 = np.random.permutation(N)
            ind_true_2 = ind_true_1[:K]
            ind_true = np.mod(ind_true_2 - ind_true_2(0), N) + 1  # the first entry is forced to be one
            x_true = np.zeros((N, 1))
            x_true[ind_true] = 1

            # measurements
            y = np.abs(np.fft.fft(x_true), axis=0) ** 2

            # Solving the SDP using cvx http://cvxr.com/
            R = np.random.randn(N, N)

            X = cp.Variable((N, N), symmetric=True)

            Constraints = [
                X >= 0,
                cp.trace(X) == K,
                cp.diag(X) == X[:, 0],
                X[0, 0] == 1]
            Constraints += [
                cp.trace((F[i, :].reshape(1, -1).T @ F[i, :].reshape(1, -1)) @ X) == y[i] for i in range(N)
            ]

            Obj = cp.Minimize(cp.trace(X @ R))

            Problem = cp.Problem(objective=Obj, constraints=Constraints)

            # extracting the leading eigenvector:
            eig_val, eig_vec = eigh(X)
            x_est = eig_vec[:, np.abs(eig_val).argmax()]
            x_est = np.sqrt(np.abs(eig_val).max()) * x_est

            X_rank[n, K, _iter] = np.linalg.matrix_rank(X, 1e-4)

            x_est = np.round(x_est)
            sdp_err[n, K, _iter] = compute_error(x_est, x_true)


