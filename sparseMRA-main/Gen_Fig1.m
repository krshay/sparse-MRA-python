% This script generates Figure 1 of the paper "SPARSE MULTI-REFERENCE
% ALIGNMENT: SAMPLE COMPLEXITY AND COMPUTATIONAL HARDNESS" by Tamir Bendory, Oscar Mickelin, and Amit Singer
%
% Last update: September 20, 2021
% Tamir Bendory

clear;
close all;
clc;

%% parameters

rng(987); % seed

N = 80; % signal's length
% N = 120;
K_vec = 5:17;  % sparsity level
num_iter = 500;
% last_iter_rrr_120 = zeros(length(K_vec), num_iter);
last_iter_rrr_80 = zeros(length(K_vec), num_iter);

%% RRR parameters

parameters.beta = 1/2;
parameters.max_iter = 10^6;
parameters.verbosity = 1;
parameters.th = 1e-5;

%% Main loop

for k = 1:length(K_vec)
    K = K_vec(k);
    %%
    for iter = 1:num_iter
        
        %% generating a binary signal with random support
        ind_true = randperm(N);
        ind_true = ind_true(1:K);
        x_true = zeros(N,1);
        x_true (ind_true) = 1;
        
        %% measurements
        
        y = abs(fft(x_true)).^2;
        
        %% RRR
        x_init = randn(N, 1);
        tic
        [x_est, error, diff, last_iter_rrr_80(k,iter)] = RRR(sqrt(y), x_init, K, parameters);
    end
    fprintf('K = %g\n', K)
%     save('last_iter_rrr_120','last_iter_rrr_120');
end

%% plotting

fig = figure;
hold on;
% plot(K_vec, mean(last_iter_rrr, 2));
plot(K_vec, median(last_iter_rrr_80, 2), 'linewidth',1);
plot(K_vec, median(last_iter_rrr_120, 2), 'linewidth',1);
legend({'L=80', 'L=120'})
set(gca, 'YScale', 'log')
ylabel('iterations')
xlabel('K')
axis tight;
%pdf_print_code(gcf, 'RRR_iterations_80.pdf', 11)
saveas(fig,'Fig1.png')


%%

% 
%     load('last_iter_rrr_80.mat');
%     load('last_iter_rrr_120.mat');
%     K_vec = 5:17;  % sparsity level
%     
%     figure;
%     hold on;
%     plot(K_vec, median(last_iter_rrr_80, 2), 'linewidth',1);
%     plot(K_vec, median(last_iter_rrr_120, 2), 'linewidth',1);
%     set(gca, 'YScale', 'log')
%     ylabel('iterations (log scale)')
%     xlabel('M')
%     axis tight;
%     legend('L=80', 'L=120', 'Location', 'northwest');
%     pdf_print_code(gcf, 'RRR_XP.pdf', 11)
%     
    