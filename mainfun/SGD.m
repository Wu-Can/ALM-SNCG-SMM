%%******************************************************************************************************
%% SGD:
%% Semismooth Newton-CG based augmented Lagrangian method for solving the general support
%% matrix machine (SMM) model:
%%
%% (P) minimize_{W,b}  0.5*||W||^2_F + tau*||W||_* + C sum^n_{i=1}max{1-y_i[tr(W^TX_i)+b], 0}
%%
%% [obj,W,b,runhist,info] = SGD(Ainput,y,OPTIONS)
%%
%% where:
%% {X_i,y_i}^n_{i=1} = training samples
%% W in R^{p*q}, b in R = unkonwn variables
%% [obj,W,b,runhist,info] = SGD(Ainput,y,OPTIONS)
%%
%% Input:
%% Ainput = matrix in R^{(p*q)*n} with i-th column X_i(:) for i=1,...,n
%% y0 = class labels in {-1,1}^n
%% OPTIONS.test_svd = 1, if test time of SVD
%%                    0, otherwise
%% OPTIONS.flag_proj = 1, if use projection step
%%                   = 0, otherwise
%% OPTIONS.tol = solution accuracy tolerance of (P)
%% OPTIONS.tau = parameter tau in (P)
%% OPTIONS.C = parameter C in (P)
%% OPTIONS.optval = objective value from ALM-SNCG with relkkt < 1e-8
%% OPTIONS.maxiter = maximum iteration numbers
%%
%% Output:
%% obj = primal objective value
%% (W, b) = output primal solution for (P)
%% runhist = a structure containing the run history
%% info.relobj = relative objective value
%% info.totaltime = total running time
%% info.iter = total number of ALM iterations
%% info.num_svd = number of SVD
%% info.time_svd = spent time on SVD
%%
%% Copyright (c) 2024 by Can Wu, Donghui Li, Defeng Sun
%% For more details, please see Sections 2-3 of the paper:
%% Support matrix machine: exploring sample sparsity, low rank,
%% and adaptive sieving in high-performance computing
%%******************************************************************************************************
function [obj,W,b,runhist,info] = SGD(Ainput,y,OPTIONS)
%%
%% Input parameters
%%
tol = 1e-4;
tau = 1;
C = 1;
maxiter = 100000;
maxtime = 7200;
test_svd = 1;  % Test time of SVD
flag_proj = 1; % Use projection step

if isfield(OPTIONS,'n'), n = OPTIONS.n; end
if isfield(OPTIONS,'p'), p = OPTIONS.p; end
if isfield(OPTIONS,'q'), q = OPTIONS.q; end
if isfield(OPTIONS,'test_svd'), test_svd = OPTIONS.test_svd; end
if isfield(OPTIONS,'flag_proj'), flag_proj = OPTIONS.flag_proj; end
if isfield(OPTIONS,'tol'), tol = OPTIONS.tol; end
if isfield(OPTIONS,'tau'), tau = OPTIONS.tau; end
if isfield(OPTIONS,'C'), C = OPTIONS.C; end
if isfield(OPTIONS,'optval'), optval = OPTIONS.optval; end
if isfield(OPTIONS,'maxiter'), maxiter = OPTIONS.maxiter; end
%%
%% Initial point
tstart = clock;
tstart_cpu = cputime;

W_old = zeros(p, q);
b_old = 0;
en = ones(n,1);

if test_svd
    num_svd = 0; time_svd = 0;
    tic; [U, S, V] = svd(W_old, 'econ'); time_svd_tmp = toc;
    num_svd = num_svd + 1; time_svd = time_svd + time_svd_tmp;
else
    [U, S, V] = svd(W_old, 'econ');
end
s = diag(S);
tolerance = max(size(W_old)) * eps(norm(s));
rank_W = sum(s > tolerance);
if rank_W < min(p,q)
    U = U(:, 1:rank_W);
    V = V(:, 1:rank_W);
end
%%
%% Print the initial information
%%
fprintf('\n *********************************************************');
fprintf('*********************************************************');
fprintf('\n\t\t SGD for solving SMM with tau = %6.3f and C = %6.3f', tau, C);
fprintf('\n *********************************************************');
fprintf('********************************************************* \n');
fprintf('\n problem size: p = %3.0f, q = %3.0f, n = %3.0f', p, q, n);
fprintf('\n ----------------------------------------------------------')
fprintf('\n  iter  |     pobj    |  time | steplength |  relobj');
%%
%% Main code: SGD
%%
cntAY = 0;
for iter = 1:maxiter

    % Choose k from [n] uniformly at random
    k = randi(n);
    y_k = y(k);
    X_k = reshape(Ainput(:, k),[p,q]);

    runhist.index_k(iter) = k;
    % Set steplength
    eta = n*C/iter;

    % Update W and b
    WX = sum(sum(W_old.*X_k));
    if y_k*(WX + b_old) < 1
        W_new = (1-(1/iter))*W_old - (tau/iter)*(U*V') + (eta*y_k)*X_k;
        b_new = b_old + eta*y_k;
    else
        W_new = (1-(1/iter))*W_old - (tau/iter)*(U*V');
        b_new = b_old;
    end

    if flag_proj
        W_new = min(1, sqrt(n*C)/norm(W_new, 'fro'))*W_new;
        b_new = min(1, sqrt(n*C)/abs(b_new))*b_new;
    end
    if test_svd
        tic; [U, S, V] = svd(W_new, 'econ'); time_svd_tmp = toc;
        num_svd = num_svd + 1; time_svd = time_svd + time_svd_tmp;
    else
        [U, S, V] = svd(W_new, 'econ');
    end
    s = diag(S);
    tolerance = max(size(W_new)) * eps(norm(s));
    rank_W = sum(s > tolerance);
    if rank_W < min(p,q)
        U = U(:, 1:rank_W);
        V = V(:, 1:rank_W);
    end

    AWbye_new = AYfun(Ainput,y,W_new) + b_new*y - en; cntAY = cntAY + 1;
    obj = 0.5*norm(W_new, 'fro')^2 + tau*sum(abs(s)) + C*sum(max(-AWbye_new,0));
    relobj = abs(obj - optval)/(1 + abs(optval));

    % Print results of each iteration
    ttime = etime(clock,tstart);
    fprintf('\n %5.1d  | %- 5.4e | %5.1f |  %3.2e  | %3.2e ',iter, obj, ttime, eta, relobj);

    if relobj < tol
        msg = 'relobj converged';
        fprintf('\n  %s, relobj = %3.2e', msg, relobj);
        break;
    end

    if iter == maxiter
        msg = 'maximum iteration reached';
        fprintf('\n %s, relobj = %3.2e', msg, relobj);
    elseif  (ttime > maxtime)
        msg = 'maximum time reached';
        fprintf('\n %s, relobj = %3.2e', msg, relobj);
    end

    W_old = W_new;
    b_old = b_new;
end

ttime = etime(clock,tstart);
ttime_cpu = cputime - tstart_cpu;

%%
%% Print results
%%
W = W_new; b = b_new;
info.relobj = relobj;
info.totaltime = ttime;
info.totaltime_cpu = ttime_cpu;
info.iter = iter;
if test_svd
    info.num_svd = num_svd;
    info.time_svd = time_svd;
end


fprintf('\n----------------------------------------------------------');
fprintf('------------------------------');
fprintf('\n number iter = %2.0f',iter);
fprintf('\n time = %3.2f',ttime);
fprintf('\n time per iter = %5.4f',ttime/iter);
fprintf('\n cputime = %3.2f',ttime_cpu);
fprintf('\n cntAY = %2.0d', cntAY);
if test_svd
    fprintf('\n cntSVD = %2.0d, time = %3.2f', num_svd, time_svd);
end
fprintf('\n objective value = %9.8e', obj);
fprintf('\n relative objective residual = %3.2e',relobj);
fprintf('\n----------------------------------------------------------');
fprintf('------------------------------');
end