import numpy as np
import os, scipy


def P1(x: np.ndarray, K):
    x1 = np.zeros_like(x)
    ind = np.argpartition(x, -K, axis=0)[-K:]

    x1[ind] = x[ind]
    return x1


def P2(x: np.ndarray, y: np.ndarray):
    x_fft = np.fft.fft(x, axis=0)
    sign_fft = x_fft / np.absolute(x_fft)
    y_sign = y * sign_fft
    x2 = np.real(np.fft.ifft(y_sign, axis=0))
    return x2


def RRR(y: np.ndarray,
        x_init: np.ndarray,
        K: np.ndarray,
        parameters: dict):

    beta = parameters['beta']
    max_iter = parameters['max_iter']
    verbosity = parameters['verbosity']
    th = parameters['th']

    x_est = x_init
    diff = np.zeros((max_iter, 1))
    error = np.zeros((max_iter, 1))
    last_iter = max_iter

    for _iter in range(max_iter):

        x1 = P1(x_est, K)
        x2 = P2(2 * x1 - x_est, y)
        x_est = x_est + beta * (x2 - x1)

        x_proj_new = P1(x_est, K)
        diff[_iter] = np.linalg.norm(y - np.abs(np.fft.fft(x_proj_new, axis=0))) / np.linalg.norm(y)
        x_proj = x_proj_new

        if ((_iter % int(max_iter / 100)) == 0) and verbosity:
            msg_status = f'iter = {_iter:g}, ' \
                         f'eta = {float(diff[_iter]):.4g}, '
            print(msg_status)

        if diff[_iter] < th:
            last_iter = _iter
            diff = diff[1:last_iter]
            break

    return x_est, error, diff, last_iter
