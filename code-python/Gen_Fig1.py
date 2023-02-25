import numpy as np
import matplotlib.pyplot as plt
from RRR import RRR
import scienceplots

plt.style.use('science')

rand_seed = 1
np.random.seed(rand_seed)


def RRR_loop(N):
    M_vec = np.arange(5, 18)
    num_iter = 500

    rrr_data = np.zeros((len(M_vec), num_iter))

    parameters = {
        'beta': 1/2,
        'max_iter': int(1e6),
        'verbosity': 0,
        'th': 1e-5
    }

    for m in range(len(M_vec)):
        M = M_vec[m]
        print(f'M = {M:g} \n')

        for _iter in range(num_iter):
            ind_true = np.random.permutation(N)
            ind_true = ind_true[1:M]
            x_true = np.zeros((N, 1), dtype=int)
            x_true[ind_true] = 1

            y = np.abs(np.fft.fft(x_true, axis=0)) ** 2

            x_init = np.random.randn(N, 1)
            x_est, error, diff, rrr_data[m, _iter] = RRR(np.sqrt(y), x_init, M, parameters)

        print(f'M = {M:g} \n')

    return M_vec, rrr_data


if __name__ == '__main__':
    M_vec, rrr_80 = RRR_loop(N=80)
    _, rrr_100 = RRR_loop(N=100)
    _, rrr_120 = RRR_loop(N=120)
    _, rrr_140 = RRR_loop(N=140)

    med_rrr_80 = np.median(rrr_80, axis=1)
    med_rrr_100 = np.median(rrr_100, axis=1)
    med_rrr_120 = np.median(rrr_120, axis=1)
    med_rrr_140 = np.median(rrr_140, axis=1)

    fig = plt.figure()
    plt.semilogy(M_vec, med_rrr_80)
    plt.semilogy(M_vec, med_rrr_100)
    plt.semilogy(M_vec, med_rrr_120)
    plt.semilogy(M_vec, med_rrr_140)
    plt.ylabel('iterations')
    plt.xlabel('M')
    plt.xticks(M_vec)
    fig.tight_layout()
    plt.legend(['L = 80', 'L = 100', 'L = 120', 'L = 140'])

    plt.savefig('RRR_iterations.pdf')
