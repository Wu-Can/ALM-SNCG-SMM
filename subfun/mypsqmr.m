%%===========================================================================================
%% mypsqmr:
%% preconditioned symmetric QMR (PSQMR) for solving the Newton linear 
%% system in Algorithm 2.2
%% (E)                Ax = b
%%
%% [x,resnrm,solve_ok] = mypsqmr(AJ,y0J,yJ,AJTy,par,parPX,b,tol,maxit,printlevel)
%%
%% Input: 
%% AJ = A(:,indexJ)
%% y0J = y0(indexJ)
%% yJ = y(indexJ)
%% AJTy = reshape(AJ*(y0J.*yJ),p,q)
%% par.p,par.q = dimensions of feature matrix
%% par.precond = 0, does not use preconditioner; otherwise, use it
%% par.x0 = initial point x0 
%% par.stagnate_check_psqmr = minimum iterations to check for stagnation
%% par.minitpsqmr = minimum iterations for PSQMR
%% parPX.U = left singular vectors of X_k(W)
%% parPX.V = right singular vectors of X_k(W)
%% b = right-hand vector in (E)
%% tol = tolerance for stopping criterion of (E)
%% maxit = maximum iterations
%% printlevel = 0, does not print the status of the solution;
%%            = 1, otherwise
%% Output:
%% x = output solution of (E)
%% resnrm = vector of residual norms ||b-Ax|| for each PSQMR iteration
%% solve_ok = finial status information for the solution of (E)
%%===========================================================================================
function [x,resnrm,solve_ok] = mypsqmr(AJ,y0J,yJ,AJTy,par,parPX,b,tol,maxit,printlevel)

m = par.p; n = par.q;
if ~exist('maxit')
    maxit = max(50,sqrt(par.p*par.q));%maxit = max(50,sqrt(length(b)));
end
if ~exist('printlevel')
    printlevel = 1;
end
if ~exist('tol')
    tol = 1e-6*norm(b,'fro');
end
if ~exist('L')
    par.precond = 0;
    L = [];
end

x0 = sparse(m,n);
if isfield(par,'x0')
    x0 = par.x0;
end
solve_ok = 1;
stagnate_check = 20;
miniter = 5;
if isfield(par,'stagnate_check_psqmr')
    stagnate_check = par.stagnate_check_psqmr;
end
if isfield(par,'minitpsqmr')
    miniter = par.minitpsqmr;
end
%%
x = x0;
if (norm(full(x),'fro') > 0)
    Aq = matvec_VdW(AJ,y0J,yJ,x,AJTy,par,parPX);
    r = b - Aq;
else
    r = b;
end
err = norm(r,'fro');
resnrm(1) = err;
minres = err;
%%
if par.precond == 0
    q = r;
else
    q = precondfun(par,L,r);
end
tau_old  = norm(q,'fro');
rho_old = sum(sum(r.*q)); %rho_old  = r'*q; 
theta_old = 0;
d = sparse(m,n); %d = zeros(N,1);
res = r;
Ad =  sparse(m,n);%Ad = zeros(N,1);
%%
%% main loop
%%
tiny = 1e-30;
for iter = 1:maxit
    Aq = matvec_VdW(AJ,y0J,yJ,q,AJTy,par,parPX);
    sigma = sum(sum(q.*Aq)); %sigma = q'*Aq;
    if (abs(sigma) < tiny)
        solve_ok = 2;
        if (printlevel); fprintf('s1'); end
        break;
    else
        alpha = rho_old/sigma;
        r = r - alpha*Aq;
    end
    if par.precond == 0
        u = r;
    else
        u = precondfun(par,L,r);
    end
    %%
    theta = norm(u,'fro')/tau_old; c = 1/sqrt(1+theta^2);
    tau = tau_old*theta*c;
    gam = (c^2*theta_old^2); eta = (c^2*alpha);
    d = gam*d + eta*q;
    x = x + d;
    %%----- stopping conditions ----
    Ad = gam*Ad + eta*Aq;
    res = res - Ad;
    err = norm(res,'fro');
    resnrm(iter+1) = err;
    if (err < minres); minres = err; end
    if (err < tol) && (iter > miniter)
        %solve_ok = 3;
        break;
    end
    if (iter > stagnate_check) && (iter > 10)
        ratio = resnrm(iter-9:iter+1)./resnrm(iter-10:iter);
        if (min(ratio) > 0.997) && (max(ratio) < 1.003)
            solve_ok = -1;
            if (printlevel);  fprintf('s'); end
            break;
        end
    end
    %%-----------------------------
    if (abs(rho_old) < tiny)
        solve_ok = 2;
        if (printlevel); fprintf('s2'); end
        break;
    else
        rho = sum(sum(r.*u));%rho  = r'*u; 
        beta = rho/rho_old;
        q = u + beta*q;
    end
    rho_old = rho;
    tau_old = tau;
    theta_old = theta;
end
if (iter == maxit)
    solve_ok = -2;
end
if (solve_ok ~= -1)
    if (printlevel)
        fprintf('ss');
    end
end
%%************************************************************************
%%************************************************************************
function  q = precondfun(par,L,r)

precond = 0;
if isfield(par,'precond')
    precond = par.precond;
end

if (precond == 0)
    q = r;
elseif (precond == 1)
    q = L.invdiagM.*r;
end
%%************************************************************************
