function [w_k, b, rk, stop_iter,info] = fastADMM (X, y, p, q, C, tau, max_iter, inner_iter, eps, stop, optval, rho, eta)
if (~exist('max_iter', 'var'))
    max_iter = 500;
end

if (~exist('stop', 'var'))
    stop = 0;
end

if (~exist('optval', 'var'))
    optval = [];
end

if (~exist('eps', 'var'))
    eps = 1e-8;
end

if (~exist('rho', 'var'))
    rho = 10;
end

if (~exist('eta', 'var'))
    eta = 0.999;
end

max_time = 7200;
tstart = clock;
K = ((X*X') .* (y*y')) / (rho + 1);
%t = 1 / norm(K, 2);
n = size(X, 1);
d = size(X, 2);

s_km1 = zeros(d, 1); 
s_hatk = s_km1;
lambda_km1 = zeros(d, 1); %ones(d, 1);
lambda_hatk = lambda_km1;
t_k = 1;
c_km1 = 0;

recent_number = 50;
recent_idx = 0;
obj_recent = zeros(recent_number, 1);

fprintf('\n *********************************************************');
fprintf('*********************************************************');
fprintf('\n\t\t Fast ADMM for solving SMM with tau = %6.3f and C = %6.3f', tau, C);
fprintf('\n *********************************************************');
fprintf('********************************************************* \n');
fprintf('\n problem size: p = %3.0f, q = %3.0f, n = %3.0f', p, q, n);
fprintf('\n tolerance = %3.2e, rho = %3.2f, eta = %3.2f', eps, rho, eta);
fprintf('\n ----------------------------------------------------------\n')

for k=1: max_iter
    f = 1 - ((X * (lambda_hatk + rho * s_hatk)) .* y) / (rho + 1);
    %[alpha_1, inten_k_1, ~] = steepestDescentQP(K, -b, C, 100, eps);
    %[alpha, inten_k_2] = nestrovQP(K, -b, C, alpha_1, t, 10000, eps/100000);
    
    opt = struct('TolKKT', eps/100, 'MaxIter', inner_iter, 'verb', 0);
    LB = zeros(n,1);
    UB = C * ones(n,1);
    [alpha,~] = libqp_gsmo(K, -f, y', 0, LB, UB, [], opt);
    w_k = (lambda_hatk + rho * s_hatk + X'*(alpha.*y)) / (rho + 1);
    sel = (alpha > 0) & (alpha < C);
    b = sel' * (y - X * w_k) / sum(sel);
    
    W_k = reshape(w_k, p, q);
    Lambda_k = reshape(lambda_hatk, p, q);
    S = shrinkage(rho*W_k - Lambda_k, tau) / rho;
    s_k = reshape(S, d, 1);
    
    lambda_k = lambda_hatk - rho * (w_k - s_k);
    
    c_k = (lambda_k - lambda_hatk)' * (lambda_k - lambda_hatk) / rho + rho * (s_k - s_hatk)' * (s_k - s_hatk);
    
    if (c_k < eta * c_km1)
        t_kp1 = 0.5 * (1 + sqrt(1 + 4*t_k*t_k));
        s_hatkp1 = s_k + (t_k-1) / t_kp1 * (s_k - s_km1);
        lambda_hatkp1 = lambda_k + (t_k-1) / t_kp1 * (lambda_k - lambda_km1);
        restart = false;
    else
        t_kp1 = 1;
        s_hatkp1 = s_km1;
        lambda_hatkp1 = lambda_km1;
        c_k = c_km1 / eta;
        restart = true;
    end
    
    s_hatk = s_hatkp1;
    lambda_hatk = lambda_hatkp1;
    c_km1 = c_k;
    s_km1 = s_k;
    lambda_km1 = lambda_k;
    t_k = t_kp1;
    
    obj_k = objective_value(w_k, p, q, b, X, y, C, tau);
    relobj = abs(obj_k - optval)/(1 + abs(optval));
    recent_idx = recent_idx + 1;
    obj_recent(recent_idx) = obj_k;
    if (recent_idx == recent_number)
        recent_idx = 0;
    end
    ttime = etime(clock,tstart);
    if mod(k, 1) == 0   %mod(k, 50) == 0 %mod(k, 500) == 0
        rk = sum(svd(reshape(w_k, p, q))>1e-6);
        fprintf('k=%d, obj=%5.4e, restart=%d, rank=%d, relobj=%3.2e, time=%5.1f, norm(alpha)=%5.1f\n', k, obj_k, restart, rk, relobj, ttime, norm(alpha));
    end
    
    if  stop
        if (relobj < eps) || ttime > max_time
            if relobj < eps
                fprintf('\n relobj converge!');
            else
                fprintf('\n maximum time reached!');
            end
            break;
        end
    else
        if (abs(obj_k - mean(obj_recent)) / abs(mean(obj_recent)) < eps && k > recent_number)
            break;
        end
    end
end

if k == max_iter
   fprintf('\n maximum iteration reached!');  
end
stop_iter = k;
rk = sum(svd(reshape(w_k, p, q))>1e-6);
info.time = etime(clock,tstart);
info.obj = obj_k;
info.relobj = relobj;
    function obj = objective_value(w, p, q, b, X, y, C, tau)
        obj = 0.5 * (w') * w + C * sum(max(0, 1 - y .* (X * w + b))) + tau * sum(abs(svd(reshape(w,p,q))));
    end
end