% This script generates Figure 3b of the paper "SPARSE MULTI-REFERENCE
% ALIGNMENT: SAMPLE COMPLEXITY AND COMPUTATIONAL HARDNESS" by Tamir Bendory, Oscar Mickelin, and Amit Singer
%
% Last update: September 20, 2021
% Oscar Mickelin

clc; clf; clear; close all

Ls = 2:2:26;                                %signal length
Ms = 1:1:max(Ls);                           %sparsity level
repeats = 10;                               %number of trials
inner_rep = 10;                             %number of random R to try
errs = Inf*ones(length(Ls), length(Ms));    %average residual

%main loop
for Lind = 1:length(Ls)
    L = Ls(Lind);

    for Mind = 1:length(Ms)
        M = Ms(Mind);
        if M < L/2
            for iter = 1:repeats
                disp([L,M,iter])
                if iter == 1
                    errs(Lind,Mind) = solve_binary_sos_phase_inner(L,M,inner_rep)/repeats;
                else
                    errs(Lind,Mind) = errs(Lind,Mind) + solve_binary_sos_phase_inner(L,M,inner_rep)/repeats;
                end

            end
        end
    end
    
    %plot during loop
    imagesc(errs)
    caxis([0,1])
    colorbar
    pause(0.01)
end


%save data
fname = sprintf('./binary_sosN%drepeats%dinner_rep%d.mat', L, repeats,inner_rep);
save(fname)

%plot after loop
imagesc(errs)
caxis([0,1])
colorbar


function err = solve_binary_sos_phase_inner(L,M,inner_rep)
%%% Generates one instance of the binary MRA

%generate a binary signal
ind_true = randperm(L);
ind_true = ind_true(1:M);
if sum(ind_true==1)==0
    ind_true(1) = 1;
end
x_true = zeros(L,1);
x_true (ind_true) = 1;

% generate R
R = randn(L);
R = R + R';
R = R/norm(R);

%run SoS
[err,x_est] = solve_binary_sos_phase(L,M,x_true,R);
for i = 1:inner_rep-1
    if norm(x_est) <= 1e-3
        %if SoS failed, rerun a maximum of inner_rep times
        [err,x_est] = solve_binary_sos_phase(L,M,x_true,R);
    end
end


end


function [err,x_est,x_true,y] = solve_binary_sos_phase(L,M,x_true,R)
%%% Solves one instance of the binary MRA
yalmip('clear')
mset clear
mset('yalmip',true)
mset(sdpsettings('solver', 'mosek','cachesolvers',1,'verbose',0))

%variable
mpol('x',L,1)

% Generate ground-truth, if not provided
if nargin <= 2
    ind_true = randperm(L);
    ind_true = ind_true(1:M);
    if sum(ind_true==1)==0
        ind_true(1) = 1;
    end
    x_true = zeros(L,1);
    x_true (ind_true) = 1;
end


% measurements

y = abs(fft(x_true)).^2;

% SoS
% preparation 
F = dftmtx(L);
if nargin <= 3
    R = randn(L);
    R = R+R';
    R = R/norm(R);
end

obj = x'*R*x;
constr = [sum(x.^2) == M, x'.^2 == x', x(1) == 1];
 
% measurement constraints
for i = 1:L
    f = F(i, :);
    a = real(f);
    b = imag(f);
    constr = [constr, (a*x)^2 + (b*x)^2 == y(i)];
end


%%
degree = 2; %second-order sum-of-squares relaxation
disp('Solving the problem')
try
    P = msdp(min(obj), constr, degree);
    [status,~] = msol(P);

    if status == 0
        %if not solved, use x = 0 as result
         x_est = zeros(L,1);
         err = compute_error(x_est, x_true);
         return
     end

    x_est =  double(x);

    err = compute_error(x_est, x_true);
catch
     x_est = zeros(L,1);
     err = compute_error(x_est, x_true);
end


end
