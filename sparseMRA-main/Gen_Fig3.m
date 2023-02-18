% This script generates Figure 3 of the paper "SPARSE MULTI-REFERENCE
% ALIGNMENT: SAMPLE COMPLEXITY AND COMPUTATIONAL HARDNESS" by Tamir Bendory, Oscar Mickelin, and Amit Singer
%
% Last update: September 20, 2021
% Tamir Bendory

clear;
close all;
clc;

%% Problem setup

% generating signal
rng(1234); % seed
L = 60; %signal's length
%p = 0.5;
p = 0.2;
% generate a Bernoulli signal with parameter p
x_true = generate_sparse_signal(L, p);
%x_true = double(x_true);

% Number of measurements
n= 5e3;

% number of trials
num_trials = 100;

% Noise level
sigma_len = 20;
sigma_vec = logspace(-3, .7, sigma_len);

% EM paramters
max_iter = 1000;
tol_em = 1e-7;

% bispectrum inversion parameters
opts = struct();
opts.maxiter = 200;
opts.tolgradnorm = 1e-7;
opts.tolcost = 1e-18;
opts.verbosity = 0;

%% Main loop
error_em_02 = zeros(sigma_len, num_trials);
error_bis_02 = zeros(sigma_len, num_trials);

for s = 1:sigma_len
    sigma = sigma_vec(s);
    for iter = 1:num_trials
        [data, shifts] = generate_observations(x_true, n, sigma);
        x0 = generate_sparse_signal(L, p);
        
        % EM
        [x_em, W, EM_iter, EM_time, post_value] = MRA_EM(data, sigma, x0, max_iter, tol_em, p);
        [x_em, ~] = align_to_reference(x_em, x_true);
        error_em_02(s,iter) = norm(x_em - x_true)/norm(x_true);
        
        % bispectrum inversion; this algorithm needs Manopt https://www.manopt.org/
        [x_bis, p_est, problem] = MRA_het_mixed_invariants_free_p(data, sigma, 1, [], [], opts);
        [x_bis, ~] = align_to_reference(x_bis, x_true);
        error_bis_02(s,iter) = norm(x_bis - x_true)/norm(x_true);
        
        fprintf('sigma = %g, iter = %g\n', sigma, iter);
        fprintf('EM error = %g, Bis error = %g\n', error_em_02(s,iter), error_bis_02(s,iter));
        save('error_bis_02','error_bis_02');
        save('error_em_02','error_em_02');
    end
end

%% plotting

figure;
hold on;
plot(sigma_vec, mean(error_em_02, 2),'linewidth',1);
plot(sigma_vec, mean(error_bis_02, 2),'linewidth',1);
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
legend('EM','Bispectrum','Location','northwest')
axis tight
%title('l2 error');
hold off;
filename = strcat('error_p_',num2str(p));
%pdf_print_code(gcf, strcat(filename,'.pdf'), 11);
saveas(gcf, strcat(filename, '.fig'))
saveas(gcf, strcat(filename, '.jpg'))

%%

% load('error_bis_02.mat')
% load('error_bis_05.mat')
% load('error_em_02.mat')
% load('error_em_05.mat')
%
% figure;
% hold on;
% plot(sigma_vec, mean(error_em_02, 2),'b','linewidth',1);
% plot(sigma_vec, mean(error_bis_02, 2),'r','linewidth',1);
% plot(sigma_vec, mean(error_em_05, 2),'--b','linewidth',1);
% plot(sigma_vec, mean(error_bis_05, 2),'--r','linewidth',1);
%
% set(gca, 'YScale', 'log')
% set(gca, 'XScale', 'log')
% legend('EM, q=0.2','Bispectrum , q=0.2','EM, q=0.5','Bispectrum , q=0.5','Location','northwest')
% axis tight
% xlabel('\sigma')
% ylabel('relative error')
% %title('l2 error');
% hold off;
%
% pdf_print_code(gcf, 'EM_bispectrum.pdf', 11)
