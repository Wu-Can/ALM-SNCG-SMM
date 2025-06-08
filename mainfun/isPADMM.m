%%*************************************************************************************************
%% isPADMM:
%% an inexact semi-proximal ADMM for solving the support matrix machine (SMM) model:
%%
%%  (P) minimize_{W,b,U}   0.5*||W||^2_F + tau*||U||_* + delta^*_S(en - AW - b*y)
%%      subject to         W - U = 0,
%% where W in R^{p*q}, b in R, U in R^{p*q}, and S = { x in R^n: 0 <= x <= C}.
%% {X_i,y_i}^n_{i=1} is given training samples.
%%
%% [obj,W,b,runhist,info] = isPADMM_scaling_relobj_1(Ainput,y,OPTIONS)
%%
%% Input:
%% Ainput = matrix in R^{(p*q)*n} with i-th column as the vector X_i(:) for i=1,...,n
%% y = class label vector in R^n
%% OPTIONS.tol = accuracy tolerance for the solution of (P)
%% OPTIONS.C = parameter C in (P)
%% OPTIONS.tau = parameter tau in (P)
%% OPTIONS.sigma = initial sigma value in isPADMM
%% OPTIONS.steplen = step length in isPADMM
%% OPTIONS.fixsigma = 1, fix sigma as a constant in isPADMM
%%                    0, otherwise
%% (OPTIONS.W0, OPTIONS.b0, OPTIONS.Lam0) = initial point for isPADMM
%% OPTIONS.maxiter = maximum number of isPADMM iterations
%% OPTIONS.ifrandom = 1, test isPADMM on random data
%%                    0, test isPADMM on real data
%% OPTIONS.optval = objective value from ALM-SNCG with relkkt < 1e-8
%% [OPTIONS.sigscale1,OPTIONS.sigscale2,OPTIONS.sigscale3,OPTIONS.sigma_mul,
%% OPTIONS.sigma_iter,OPTIONS.sigma_siter,OPTIONS.sigma_giter] = posotive 
%% factors for updating sigma in isPADMM
%% [OPTIONS.sigalm_scale1,OPTIONS.sigalm_scale2,OPTIONS.sigalm_scale3] = 
%% posotive factors for updating sigma in ALM
%%
%% Output:
%% obj = primal objective value of (P) from isPADMM 
%% (W, b, info.U) = output primal solution for (P)
%% info.iter = total number of isPADMM iterations
%% info.totaltime = total running time
%% info.totaltime_cpu = total running CPU time
%% info.primfeas = relative primal infeasibility
%% info.dualfeas = relative dual infeasibility
%% info.maxfeas = maximum of relative primal and dual infeasibilities
%% info.relobj = relative objective value
%% info.sigma_final = final sigma value in isPADMM
%% info.totle_iter_alm = total number of ALM iterations
%% info.totle_numSSNCG = total number of SNCG iterations
%%
%% Copyright (c) 2024 by Can Wu, Donghui Li, Defeng Sun
%% For more details, please see Appendix D of the paper: 
%% Support matrix machine: exploring sample sparsity, low rank,
%% and adaptive sieving in high-performance computing
%%*************************************************************************************************

function [obj,W,b,runhist,info] = isPADMM(Ainput,y,OPTIONS)
%%
%% parameter setting
%%
tol = 1e-6;
C = 1;
tau = 1;
sigma = 1;
steplen = 1.618;

maxiter = 30000;
maxtime = 7200;
printyes = 1; % print the initial information
breakyes = 0;
msg = [];
flag_initial0 = 1; % 0 as the initial point

testsigma = 1;
fixsigma = 0;
sigalm_scale1 = 5;
sigalm_scale2 = 10;
sigalm_scale3 = 35;

if isfield(OPTIONS,'tol'), tol = OPTIONS.tol; end
if isfield(OPTIONS,'C'), C = OPTIONS.C; end
if isfield(OPTIONS,'tau'), tau = OPTIONS.tau; end
if isfield(OPTIONS,'sigma'), sigma = OPTIONS.sigma; end
if isfield(OPTIONS,'n'), n = OPTIONS.n; else, n = length(y); end
if isfield(OPTIONS,'p'), p = OPTIONS.p; end
if isfield(OPTIONS,'q'), q = OPTIONS.q; end
if isfield(OPTIONS,'steplen'), steplen = OPTIONS.steplen; end
if isfield(OPTIONS,'fixsigma'), fixsigma = OPTIONS.fixsigma; end
if isfield(OPTIONS,'W0'), W0 = OPTIONS.W0; flag_initial0 = 0; end
if isfield(OPTIONS,'b0'), b0 = OPTIONS.b0; end
if isfield(OPTIONS,'Lam0'), Lam0 = OPTIONS.Lam0; end
if isfield(OPTIONS,'maxiter'), maxiter = OPTIONS.maxiter; end
if isfield(OPTIONS,'ifrandom'), ifrandom = OPTIONS.ifrandom; end
if isfield(OPTIONS,'optval'), optval = OPTIONS.optval; end
if (fixsigma == 0) || (testsigma == 1)
    sigma_iter = 100;
    sigma_siter = 50;
    sigma_giter = 500;
    sigma_mul0 = 2;
    sigscale1 = 1.8;
    sigscale2 = 1.9;
    sigscale3 = 2.5;
    if isfield(OPTIONS,'sigscale1');  sigscale1 = OPTIONS.sigscale1; end
    if isfield(OPTIONS,'sigscale2');  sigscale2 = OPTIONS.sigscale2; end
    if isfield(OPTIONS,'sigscale3');  sigscale3 = OPTIONS.sigscale3; end
    if isfield(OPTIONS,'sigma_mul');  sigma_mul0 = OPTIONS.sigma_mul; end
    if isfield(OPTIONS,'sigma_iter');  sigma_iter = OPTIONS.sigma_iter; end
    if isfield(OPTIONS,'sigma_siter');  sigma_siter = OPTIONS.sigma_siter; end
    if isfield(OPTIONS,'sigma_giter');  sigma_giter = OPTIONS.sigma_giter; end
end
if isfield(OPTIONS,'sigalm_scale1');  sigalm_scale1 = OPTIONS.sigalm_scale1; end
if isfield(OPTIONS,'sigalm_scale2');  sigalm_scale2 = OPTIONS.sigalm_scale2; end
if isfield(OPTIONS,'sigalm_scale3');  sigalm_scale3 = OPTIONS.sigalm_scale3; end

par.tol = tol;
par.C = C;
par.tau = tau;
par.sigma = sigma;
par.n = n;
par.p = p;
par.q = q;
par.steplen = steplen;

par.minitpsqmr = 3;
cntATz = 0;
cntAY = 0;
totle_iter_alm = 0;
totle_numSSNCG = 0;
totaltime_ALM = 0;
%%
%% Amap and ATmap
%%
sparsity = 1 - nnz(Ainput)/(n*p*q);
if sparsity > 0.1  % adjusting
    A0 = sparse(Ainput);
else
    A0 = Ainput;
end
%%
%% initiallization
%%
tstart = clock;
tstart_cpu = cputime;
en = ones(par.n,1);
if flag_initial0 == 1
    Z0pq = zeros(par.p,par.q);
    z0n = zeros(par.n,1);
    W0 = Z0pq; b0 = 0;
    Lam0 = Z0pq; U0 = W0;
else
    U0 = W0;
end
%%
%% print initial information
%%
if printyes
    fprintf('\n *******************************************************');
    fprintf('******************************************');
    fprintf('\n \t\t   two-block isPADMM  for solving SMM with tau = %6.3f and  C = %6.3f ',par.tau, par.C);
    if (testsigma == 1) && ( fixsigma == 0 )
        fprintf('\n sigscale1 = %6.3f, sigscale2 = %6.3f, sigscale3 = %6.3f', sigscale1, sigscale2, sigscale3);
        fprintf('\n sigma_mul = %6.3f, sigma_siter = %6.3f, sigma_giter = %6.3f',...
            sigma_mul0, sigma_siter, sigma_giter);
        fprintf('\n sigma_iter1 = %6.3f, \n sigma0 = %6.3f',sigma_iter, sigma);
    end
    fprintf('\n ******************************************************');
    fprintf('*******************************************\n');
    fprintf('\n problem size: n = %3.0f, p = %3.0f, q = %3.0f',n, p, q);
    fprintf('\n ---------------------------------------------------');
    fprintf('\n  iter|  pinf       dinf   |    pobj     |');
    fprintf('  time  |  sigma   |  step  ');
    fprintf('|   relobj   |');
    fprintf('iterA  iterN  timeA| sig0_alm   kktA     gapA    |');
    if (testsigma == 1) && (fixsigma == 0)
        fprintf('   [mfsr      max(mfsr)]');
    end
end
%%
%% initial primal and dual infeasibilities
%%
W_new = W0; b_new = b0; U_new = U0; Lam_new = Lam0;
if flag_initial0 == 1
    AW_new = z0n; W_U_new = Z0pq;
    normW = 0; normU = 0; normLam = 0;
else
    AW_new = AYfun(A0,y,W_new); cntAY = cntAY + 1;
    W_U_new = W_new - U_new;
    normW = norm(W_new,'fro'); normU = norm(U_new,'fro');
    normLam = norm(Lam_new,'fro');
end

% primal feasibility, dual feasibility, duality gap
if flag_initial0 == 1
    primfeas = 0; dualfeas = 0;
    primobj =  par.C*par.n;
    AW_by_en = - en;
else
    primfeas = norm(W_U_new,'fro')/(1 + normW + normU);
    dualfeas = norm(Lam_new - projspball(Lam_new,par.tau),'fro')/(1 + normLam);
    
    AW_by_en = AW_new + b_new*y - en;
    primobj = 0.5*norm(W_new,'fro')^2 + par.tau*sum(svd(full(W_new))) + par.C*sum(max(-AW_by_en,0));
end


max_pdfeas = max(dualfeas,primfeas);
feasratio = primfeas/max_pdfeas;
if feasratio == 0
    maxfeasratio = inf;
else
    maxfeasratio = max(feasratio,1/feasratio);
end
iter = 0; ttime = etime(clock,tstart);
relobj = abs(primobj - optval)/(1 + abs(optval));
if (printyes)
    fprintf('\n %5.0f| %3.2e  %3.2e | %- 5.4e |',iter,primfeas,dualfeas,primobj);
    fprintf('  %5.1f | %3.2e | %2.4f |  %3.2e',ttime, par.sigma, par.steplen, relobj);
    if (testsigma == 1) && (fixsigma == 0)
        fprintf('                                                        [%3.2e        %3.2e]',feasratio,maxfeasratio);
    end
end


%%
%% main Loop
%%
for iter = 1:maxiter
    %% Compute W_new and b_new
    delt = 1e-6; 
    par_alm.tau = 0;
    par_alm.delt = delt/(1 + par.sigma);
    par_alm.C = par.C/(1 + par.sigma);
    B = (par.sigma*U_new - Lam_new)/(1 + par.sigma);
    a = b_new;
    
    par_alm.n = par.n;
    par_alm.p = par.p;
    par_alm.q = par.q;
    if iter > 1
        par_alm.tol = max(min(primfeas/100,1/(iter^2)), par.tol/100);
    else
        par_alm.tol = 0.1;
    end
    par_alm.ifrandom = ifrandom;
    par_alm.flag_scaling = 0;
    par_alm.stop = 0; % 0, if relkkt <= tol; 1, if relobj <= tol; 2, if relgap <= tol
    par_alm.warm = 0;
    
    
    par_alm.W0 = W_new;
    par_alm.b0 = b_new;
    par_alm.lam0 = zeros(par_alm.n,1);
    index_v = (AW_by_en < -1e-12);
    par_alm.lam0(index_v) = -par_alm.C;
    par_alm.sigmaiter = 2;
    par_alm.sigmascale = 1.2;
    par_alm.Test_sigma = 1;
    
    if par_alm.ifrandom
        if (iter == 1) || (primfeas >= 1e-1)
            par_alm.sigma = min(sigalm_scale1*par.sigma,5e3); 
        elseif (primfeas >= 1e-2)
            par_alm.sigma = min(sigalm_scale2*par.sigma,5e3);
        else
            par_alm.sigma = min(sigalm_scale3*par.sigma,5e3);
        end
    else
        if  (primfeas > 1e-3) || (iter < 10)
            if (par.n < 10000)
                par_alm.sigma = min(par.C, 0.1);
            else
                par_alm.sigma = par.C;
            end
        else
            if (par.C == 1) && (par.n < 10000)
                par_alm.sigma = max(1e-3, min(par.sigma*par.C/1000, 1e3));
            else
                par_alm.sigma = max(1e-3, min(par.sigma*par.C/10, 1e3));
            end
        end
    end
    
    par_alm.printyes = 0;
    par_alm.printlevel_ALM = 0;
    par_alm.printlevel_SSN = 0;
    
    [~,W_new,b_new,~,info_alm] = ALMSNCG(A0,y,par_alm,B,a);
    
    AW_by_en = info_alm.AWbye_new_org;
    
    iter_alm = info_alm.iter;
    iter_ssn = info_alm.numSSNCG;
    time_alm = info_alm.totaltime;
    relkkt_alm = info_alm.res_kkt_final;
    relgap_alm = info_alm.res_gap_final;
    
    totle_iter_alm = totle_iter_alm + iter_alm;
    totle_numSSNCG = totle_numSSNCG + iter_ssn;
    totaltime_ALM = totaltime_ALM + time_alm;
    
    %% Compute U_new
    tmpU = Lam_new + par.sigma*W_new;
    U_new = (1/par.sigma)*(tmpU - projspball(tmpU,tau));
    
    %% Update mutiplier Lam_new
    W_U_new = W_new - U_new;
    Lam_new = Lam_new + (par.steplen*par.sigma)*W_U_new;
    
    %% check for termination
    ttime = etime(clock,tstart);
    
    primobj = 0.5*norm(W_new,'fro')^2 + par.tau*sum(svd(full(W_new))) + par.C*sum(max(-AW_by_en,0));
    relobj = abs(primobj - optval)/(1 + abs(optval));
    if relobj < tol
        breakyes = 1;
        msg = 'relobj converged';
        info.termcode = 1;
    end
    runhist.relobj(iter) = relobj;
    
    %%--------------------------------------------------------
    %% print results
    %%--------------------------------------------------------
    normW = norm(W_new,'fro'); normU = norm(U_new,'fro'); normLam = norm(Lam_new);
    primfeas = norm(W_U_new,'fro')/(1 + normW + normU);
    dualfeas = norm(Lam_new - projspball(Lam_new,par.tau),'fro')/(1 + normLam);
    
    if (fixsigma == 0)
        feasratio = primfeas/dualfeas;
        maxfeasratio = max(feasratio,1/feasratio);
        runhist.maxfeasratio(iter) = maxfeasratio;
    end
    
    if (printyes)
        fprintf('\n %5.0d| %3.2e  %3.2e | %- 5.4e | ',...
            iter,primfeas,dualfeas,primobj);
        fprintf(' %5.1f | %3.2e | %2.4f ',ttime, par.sigma,par.steplen);
        fprintf('|  %3.2e  |', relobj);
        fprintf(' %3.0f    %3.0f   %3.2f |  %3.3f    %3.2e   %3.2e |',iter_alm,iter_ssn,...
            time_alm,par_alm.sigma,relkkt_alm,relgap_alm);
        if (testsigma == 1) && (fixsigma == 0)
            fprintf('   [%3.2e  %3.2e]',feasratio,maxfeasratio);
        end
    end
    
    runhist.primobj(iter) = primobj;
    runhist.time(iter) = ttime;
    runhist.primfeas(iter) = primfeas;
    runhist.dualfeas(iter) = dualfeas;
    
    ttime = etime(clock,tstart);
    if (ttime > maxtime)
        msg = 'maximum time reached';
        breakyes = 10;
        info.termcode = 2;
    elseif (iter == maxiter)
        msg = 'maximum iteration reached';
        breakyes = 100;
        info.termcode = 3;
    end
    
    if (breakyes > 0)
        fprintf('\n  breakyes = %3.1f, %s, relobj = %3.2e',breakyes,msg,relobj);
        break;
    end
    
    %%-----------------------------------------------------------
    %% update penalty parameter sigma
    %%-----------------------------------------------------------
    if (fixsigma == 0) && (rem(iter,sigma_iter) == 0) 
        if maxfeasratio <= sigma_siter
            sigmascale = sigscale1;
        elseif maxfeasratio > sigma_giter
            sigmascale = sigscale3;
        else
            sigmascale = sigscale2;
        end
        
        if feasratio > sigma_mul0
            par.sigma = par.sigma*sigmascale;
        elseif feasratio < 1/sigma_mul0
            par.sigma = par.sigma/sigmascale;
        end
        par.sigma = min(max(par.sigma,par.tol),1e6);
        runhist.feasratio(iter) = feasratio;
        runhist.maxfeasratio(iter) = maxfeasratio;
    end
    
end
%------------------------------End dADMM main loop-------------------------
%%-------------------------------------------------------------------------
%% Print results
%%-------------------------------------------------------------------------
ttime = etime(clock,tstart);
ttime_cpu = cputime - tstart_cpu;

if (testsigma == 1) && (fixsigma == 0)
    if iter == 1
        info.max_maxfeasratio = inf;
        info.min_maxfeasratio = inf;
    else
        info.max_maxfeasratio = max(runhist.maxfeasratio(2:end));
        info.min_maxfeasratio = min(runhist.maxfeasratio(2:end));
    end
end

if (printyes)
    if ~isempty(msg); fprintf('\n %s',msg); end
    fprintf('\n--------------------------------------------------------------');
    fprintf('------------------');
    fprintf('\n  number iter = %2.0d',iter);
    fprintf('\n  time = %3.2f',ttime);
    fprintf('\n  time per iter = %5.4f',ttime/iter);
    fprintf('\n  cputime = %3.2f', ttime_cpu);
    fprintf('\n  primobj = %9.8e',primobj);
    fprintf('\n  primfeas = %3.2e, dualfeas = %3.2e',primfeas, dualfeas);
    fprintf('\n number of iter_alm = %3.0d, number of numSSNCG = %3.0d, time_alm = %3.2f',...
        totle_iter_alm, totle_numSSNCG, totaltime_ALM); %?
    fprintf('\n number of ATz = %3.0d, number of AY = %3.0d', cntATz, cntAY); %?
    if (testsigma == 1) && (fixsigma == 0)
        fprintf('\n  max(maxfeasratio) = %3.2e', info.max_maxfeasratio);
        fprintf('\n  min(maxfeasratio) = %3.2e', info.min_maxfeasratio);
    end
    fprintf('\n relative objective residual = %3.2e',relobj);
    fprintf('\n--------------------------------------------------------------');
    fprintf('------------------\n');
end

info.iter = iter;
info.totaltime = ttime;
info.totaltime_cpu = ttime_cpu;
info.primfeas = primfeas;
info.dualfeas = dualfeas;
info.maxfeas = max(primfeas,dualfeas);
info.primobj = primobj;

W = W_new; b = b_new;
U = U_new; Lam = Lam_new;
obj = primobj;
info.W = W; info.b = b;
info.U = U; info.Lam = Lam;
info.relobj = relobj;
info.sigma_final = par.sigma;
info.totle_iter_alm = totle_iter_alm;
info.totle_numSSNCG = totle_numSSNCG;
info.totaltime_ALM = totaltime_ALM;
info.AW_by_en = AW_by_en;
end