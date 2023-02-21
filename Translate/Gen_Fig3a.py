import numpy as np
import cvxpy as cp
from scipy.linalg import dft, eig
from scipy.io import savemat
from utils_fig_3 import compute_error
import warnings

warnings.filterwarnings("ignore")

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

for n in np.arange(len(N_vec)):
    N = N_vec[n]
    F = dft(N)

    for K in np.arange(1, len(K_vec) + 1):
        for _iter in np.arange(max_iter):
            # geenrating a binary signal
            ind_true = np.random.permutation(N)

            # ind_true_1 = first_iter['ind_true_1']

            ind_true = ind_true[:K]
            ind_true = np.mod(ind_true - ind_true[0], N)  # + 1  # the first entry is forced to be one
            x_true = np.zeros((N, 1))
            x_true[ind_true] = 1

            # measurements
            y = np.abs(np.fft.fft(x_true, axis=0)) ** 2

            # Solving the SDP using cvx http://cvxr.com/
            R = np.random.randn(N, N)
            # R = first_iter['R']
            #
            # patao_assert(y, first_iter['y'])
            # patao_assert(R, first_iter['R'])
            # patao_assert(F, first_iter['F'])
            # patao_assert(K, first_iter['K'][0])

            X = cp.Variable((N, N), PSD=True)

            Constraints = [
                X >= 0,
                cp.trace(X) == K,
                cp.diag(X) == X[:, 0],
                X[0, 0] == 1]

            for i in range(N):
                f = F[i, :].reshape(1, -1)
                Constraints += [cp.trace((f.T @ f) @ X) == y[i]]

            Obj = cp.Minimize(cp.trace(X @ R))

            Problem = cp.Problem(objective=Obj, constraints=Constraints)

            for i in np.arange(3):
                Problem.solve(verbose=False)
                if not Problem.solution.status == 'infeasible':
                    break
            if Problem.solution.status == 'infeasible':
                sdp_err[n, K - 1, _iter] = np.nan

            else:
                # extracting the leading eigenvector:
                eig_val, eig_vec = eig(X.value)
                ind_max = np.abs(eig_val).argmax()
                x_est = eig_vec[:, ind_max].real
                x_est = np.sqrt(eig_val[ind_max].real) * x_est

                X_rank[n, K - 1, _iter] = np.linalg.matrix_rank(X, 1e-4)

                x_est = np.round(x_est)
                sdp_err[n, K - 1, _iter] = compute_error(x_est, x_true)

            print(f"N = {N}, K = {K}, iter = {_iter}")
            print(f"error = {sdp_err[n, K - 1, _iter]:.3e}")
            savemat('sdp_err.mat', {'sdp_err': sdp_err})
