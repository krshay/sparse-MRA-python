function [x, W, EM_iter, EM_time, post_value] = MRA_EM(X, sigma, x, max_iter, tol, p)

% Expectation maximization algorithm for multireference alignment.
% X: data (each column is an observation)
% sigma: noise standard deviation affecting measurements
% x: initial guess for the signal (optional)
% tol: EM stops iterating if two subsequent iterations are closer than tol
%      in 2-norm, up to circular shift (default: 1e-5).
% batch_niter: number of batch iterations to perform before doing full data
%              iterations if there are more than 3000 observations
%              (default: 3000.)
%
% May 2017
% https://arxiv.org/abs/1705.00641
% https://github.com/NicolasBoumal/MRA

% X contains M observations, each of length N
[N, M] = size(X);

% Initial guess of the signal
if ~exist('x', 'var') || isempty(x)
    if isreal(X)
        x = randn(N, 1);
    else
        x = randn(N, 1) + 1i*randn(N, 1);
    end
end
x = x(:);
assert(length(x) == N, 'Initial guess x must have length N.');

% Tolerance to declare convergence
if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-5;
end


% In practice, we iterate with the DFT of the signal x
%fftx = fft(x);
x_est = x;
% Precomputations on the observations
fftX = fft(X);
sqnormX = repmat(sum(abs(X).^2, 1), N, 1);
tic_full = tic();

% In any case, finish with full passes on the data
full_niter = max_iter;
post_value = zeros(max_iter, 1);

for iter = 1 : full_niter
    
    [x_new, W, post_value(iter)] = EM_iteration(x_est, fftX, sqnormX, sigma, p);
    
    if relative_error(x_new, x_est) < tol
        break;
    end
    
    x_est = x_new;
    
end

%fprintf('\t\tEM: %d full iterations, %.2g [s]\n', iter, toc(tic_full));

EM_iter = iter;
post_value = post_value(1:EM_iter);
EM_time = toc(tic_full);
%x = ifft(fftx);
x = x_est;

end


% Execute one iteration of EM with current estimate of the DFT of the
% signal given by fftx, and DFT's of the observations stored in fftX, and
% squared 2-norms of the observations stored in sqnormX, and noise level
% sigma.
function [x_new, W, post_value] = EM_iteration(x, fftX, sqnormX, sigma, p)
fftx = fft(x);
C = ifft(bsxfun(@times, conj(fftx), fftX));
T = (2*C - sqnormX)/(2*sigma^2);
L = size(fftX, 1);
n = size(fftX, 2);
S = sum(x);
post_value = sum(log(sum(exp(T),1))) + S*log(p) + (L-S)*log(1-p);  % posterior function
T = bsxfun(@minus, T, max(T, [], 1));
W = exp(T);
W = bsxfun(@times, W, 1./sum(W, 1));
fftx_new = mean(conj(fft(W)).*fftX, 2);
x_new = real(ifft(fftx_new))+ 2*sigma^2/n*log(p/(1-p));

end




