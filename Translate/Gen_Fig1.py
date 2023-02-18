import numpy as np
from matplotlib import pyplot as plt
from RRR import RRR
import os, scipy

rand_seed = np.random.randint(10000)
print(f'random seed: {rand_seed}')
np.random.seed(rand_seed)


def generate_last_itervec(N=80):
    K_vec = np.arange(5, 18)
    num_iter = 500

    last_iter_rrr = np.zeros((len(K_vec), num_iter))

    parameters = {
        'beta': 1/2,
        'max_iter': int(1e6),
        'verbosity': 0,
        'th': 1e-5
    }

    for k in range(len(K_vec)):
        K = K_vec[k]
        print(f'K = {K:g} \n')

        for _iter in range(num_iter):
            ind_true = np.random.permutation(N)
            ind_true = ind_true[1:K]
            x_true = np.zeros((N, 1), dtype=int)
            x_true[ind_true] = 1

            y = np.abs(np.fft.fft(x_true, axis=0)) ** 2

            x_init = np.random.randn(N, 1)
            x_est, error, diff, last_iter_rrr[k, _iter] = RRR(np.sqrt(y), x_init, K, parameters)

        print(f'K = {K:g} \n')

    return K_vec, last_iter_rrr


if __name__ == '__main__':
    K_vec, last_iter_rrr_80 = generate_last_itervec(N=80)
    _, last_iter_rrr_120 = generate_last_itervec(N=120)

    med_last_iter_rrr_80 = np.median(last_iter_rrr_80, axis=1)
    med_last_iter_rrr_120 = np.median(last_iter_rrr_120, axis=1)

    fig = plt.figure(figsize=[6.4 * 1.5, 4.8 * 1.5])
    plt.plot(K_vec, med_last_iter_rrr_80)
    plt.plot(K_vec, med_last_iter_rrr_120)
    plt.yscale('log')
    plt.ylabel('iterations')
    plt.xlabel('K')
    fig.tight_layout()
    plt.legend(['L=80', 'L=120'])

    plt.savefig('Fig1.png')
