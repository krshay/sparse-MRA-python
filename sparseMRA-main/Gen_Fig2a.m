% This script generates Figure 2a of the paper "SPARSE MULTI-REFERENCE
% ALIGNMENT: SAMPLE COMPLEXITY AND COMPUTATIONAL HARDNESS" by Tamir Bendory, Oscar Mickelin, and Amit Singer
%
% Last update: September 20, 2021
% Tamir Bendory

clear;
close all;
clc;

%% parameters

rng(9876); %seed

N_vec = 10:5:80; % signal's length
K_vec = 2:2:20; % sparsity
max_iter = 10; % number of trials per (N,K)

%% Main loop

sdp_err = zeros(length(N_vec), length(K_vec), max_iter);
X_rank = zeros(length(N_vec), length(K_vec), max_iter);

for n = 1:length(N_vec)
    N = N_vec(n);
    F = dftmtx(N);
    for K = 1:length(K_vec)
        for iter = 1:max_iter
            % geenrating a binary signal
            ind_true = randperm(N);
            ind_true = ind_true(1:K);
            ind_true = mod(ind_true - ind_true(1),N)+1; % the first entry is forced to be one
            x_true = zeros(N,1);
            x_true (ind_true) = 1;
            % measurements
            y = abs(fft(x_true)).^2;
            
            %% Solving the SDP using cvx http://cvxr.com/
            
            warning('off');
            R = randn(N);
            
            cvx_begin sdp quiet
            variable X(N,N) symmetric
            minimize trace(X*R);
            subject to
            X >= 0;
            trace(X) == K;
            X(:) >= 0 ;
            diag(X) == X(:,1);
            X(1,1) == 1;
            for i = 1:N
                f = F(i, :);
                trace((f'*f)*X) == y(i);
            end
            cvx_end
            
            % extracting the leading eigenvector
            [x_est, val]= eigs(X, 1);
            x_est = sqrt(val)*x_est;
            X_rank(n, K,iter) = rank(X, 1e-4);
            %fprintf(' rank = %g\n', X_rank);
            %figure; plot(eig(X))
            x_est = round(x_est);
            sdp_err(n, K, iter) = compute_error(x_est, x_true);
            
            fprintf('N = %g, K = %g, iter = %g\n', N, K, iter);
            fprintf('error = %g\n', sdp_err(n, K,iter));
            save('sdp_err','sdp_err');
        end
    end
end

%% plotting

figure;
imagesc(K_vec, N_vec, mean(sdp_err,3))
ylabel('L');
xlabel('M');
colorbar
%df_print_code(gcf, 'SDP_xp.pdf', 11)


