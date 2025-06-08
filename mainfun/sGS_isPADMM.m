%%*************************************************************************************************
%% sGS-isPADMM:
%% A symmetric Gauss-Seidel based inexact semi-proximal ADMM for solving the support matrix machine
%% (SMM) model:
%%
%%    minimize_{W,b,v,U}  0.5*||W||^2_F + tau*||U||_* + delta^*_S(v)
%%    subject to          AW + b*y + v = en
%%                        W - U = 0
%%
%% where W in R^{p*q}, b in R, v in R^n, U in R^{p*q}, S = { x in R^n: 0 <= x <= C},
%% and {X_i,y_i}^n_{i=1} is given training samples.
%%
%% [obj,W,b,runhist,info] = sGS_isPADMM(Ainput,y0,OPTIONS)
%%
%% Input:
%% Ainput = matrix in R^{(p*q)*n} with i-th column as the vector X_i(:) for i=1,...,n
%% y0 = class label vector in R^n
%% OPTIONS.C = parameter C in (P)
%% OPTIONS.tau = parameter tau in (P)
%% OPTIONS.tol = accuracy tolerance for the solution of (P)
%% sGS-isPADMM stopping criterion:
%% OPTIONS.stop = 1, use relkkt < tol 
%%              = 0, use relative objective values
%% OPTIONS.computeW = 1,  update W by direct method   
%%                    0,  update W by CG method
%% OPTIONS.largenum_k = number of largest eigenvalues of A^*A+I
%% OPTIONS.flag_scaling = 1, scale matrix Ainput
%%                      = 0, otherwise
%% OPTIONS.sigma = initial sigma value in sGS-isPADMM
%% OPTIONS.fixsigma = 1, fix the value of sigma
%%                    0, otherwise
%% OPTIONS.ifrandom = 1, test sGS-isPADMM on random data
%%                    0, test sGS-isPADMM on real data
%% OPTIONS.steplen = step length in sGS-isPADMM   
%% [OPTIONS.sigscale1,OPTIONS.sigscale2,OPTIONS.sigscale3,OPTIONS.sigma_mul,
%% OPTIONS.sigma_iter,OPTIONS.sigma_siter,OPTIONS.sigma_giter] = posotive 
%% factors for updating sigma in sGS-isPADMM
%% OPTIONS.optval = objective value from ALM-SNCG with relkkt < 1e-8
%%
%% Output:
%% obj = [primal objective value, dual objective value]
%% (W, b, info.v, info.U, info.lam, info.Lam) = output KKT solution for (P)
%% runhist = a structure containing the run history
%% info.relobj = relative objective value
%% info.iter = total number of sGS-isPADMM iterations
%% info.totaltime = total running time
%% info.totaltime_cpu = total running CPU time
%% info.maxfeas = maximum of relative primal and dual infeasibilities
%% info.sigma_final = final sigma value in sGS-isPADMM
%% 
%% Copyright (c) 2024 by Can Wu, Donghui Li, Defeng Sun
%% For more details, please see Appendix E of the paper: 
%% Support matrix machine: exploring sample sparsity, low rank,
%% and adaptive sieving in high-performance computing
%%*************************************************************************************************
function [obj,W,b,runhist,info] = sGS_isPADMM(Ainput,y0,OPTIONS)
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
maxitpsqmr = 100;
printyes = 1; % print the initial information
print_psqmr = 0;
breakyes = 0;
msg = [];
flag_initial0 = 1; % 0 as the initial point
flag_scaling = 0;

testsigma = 1;
fixsigma = 0;
stop_criterion = 0;
computeW = 1; % 1, direct method; 0, CG
k = 25;
warm = 0;

if isfield(OPTIONS,'tol'), tol = OPTIONS.tol; end
if isfield(OPTIONS,'C'), C = OPTIONS.C; end
if isfield(OPTIONS,'tau'), tau = OPTIONS.tau; end
if isfield(OPTIONS,'sigma'), sigma = OPTIONS.sigma; end
if isfield(OPTIONS,'n'), n = OPTIONS.n; else, n = length(y); end
if isfield(OPTIONS,'p'), p = OPTIONS.p; end
if isfield(OPTIONS,'q'), q = OPTIONS.q; end
if isfield(OPTIONS,'maxitpsqmr'), maxitpsqmr = OPTIONS.maxitpsqmr; end
if isfield(OPTIONS,'steplen'), steplen = OPTIONS.steplen; end
if isfield(OPTIONS,'fixsigma'), fixsigma = OPTIONS.fixsigma; end
if isfield(OPTIONS,'flag_scaling'), flag_scaling = OPTIONS.flag_scaling; end
if isfield(OPTIONS,'W0'), W0 = OPTIONS.W0; flag_initial0 = 0; end
if isfield(OPTIONS,'b0'), b0 = OPTIONS.b0; end
if isfield(OPTIONS,'lam0'), lam0 = OPTIONS.lam0; end
if isfield(OPTIONS,'Lam0'), Lam0 = OPTIONS.Lam0; end
if isfield(OPTIONS,'lam0'), lam0 = OPTIONS.lam0; end
if isfield(OPTIONS,'maxiter'), maxiter = OPTIONS.maxiter; end
if isfield(OPTIONS,'stop'), stop_criterion = OPTIONS.stop;
    if stop_criterion, optval = OPTIONS.optval; end
end
if isfield(OPTIONS,'computeW'), computeW = OPTIONS.computeW; end
if isfield(OPTIONS,'largenum_k'), k = OPTIONS.largenum_k; end
if isfield(OPTIONS,'warm'), warm = OPTIONS.warm; end
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

par.tol = tol;
par.C = C;
par.tau = tau;
par.sigma = sigma;
par.n = n;
par.p = p;
par.q = q;
par.steplen = steplen;
par.computeW = computeW;

par.minitpsqmr = 3;
tol_maxfeas = 5*tol;
num_projsp = 0;
cntATz = 0;
cntAY = 0;
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
const_n = 1+sqrt(n);
if flag_initial0 == 1
    Z0pq = zeros(par.p,par.q);
    z0n = zeros(par.n,1);
    W0 = Z0pq; b0 = 0;
    lam0 = z0n; Lam0 = Z0pq;
    U0 = W0; v0 = en;
else
    U0 = W0;
    v0 = en - AYfun(A0,y0,W0) - b0*y0; cntAY = cntAY + 1;
end

%%
%% Scaling
%%
if flag_scaling == 1
    d_scale = full(max(max(abs(A0))',1));
    invd_scale = 1./d_scale;
    invD = sparse(1:par.n,1:par.n,invd_scale);
    
    A = A0*invD;
    y = invd_scale.*y0;
    en = invd_scale;
    v0 = en;
    lam0 = d_scale.*lam0;
    Cd = C*d_scale;
    yTy = y'*y;
else
    d_scale = 1;
    invd_scale = d_scale;
    A = A0; y = y0;
    Cd = C;
    yTy = n;
end
normd_scale = norm(d_scale,'inf');
%% Compute the k largest eigenvalues and their corresponding
%% eigenvectors of A^*A+I
Afun = @(x) A*(x'*A)' + x;
if computeW == 1
    %k = 25;
    options.issym = 1;
    options.isreal = 1;
    
    [Vtmp, Dtmp] = eigs(Afun, p*q, k+1,'largestabs',options);
    dtmp = diag(Dtmp);
    dtmp_smal = dtmp(end);
    dtmp = dtmp(1:end-1);
    Vtmp = Vtmp(:,1:end-1);
    
    par.dtmp = dtmp;
    par.dtmp_smal = dtmp_smal;
    par.Vtmp = Vtmp;
    par.k = k;
end
par.A = A; par.y0 = y0;
%%
%% print initial information
%%
if printyes
    fprintf('\n *******************************************************');
    fprintf('******************************************');
    fprintf('\n \t\t   sGS-isPADMM  for solving SMM with tau = %6.3f and  C = %6.3f ',par.tau, par.C);
    if (testsigma == 1) && ( fixsigma == 0 )
        fprintf('\n sigscale1 = %6.3f, sigscale2 = %6.3f, sigscale3 = %6.3f', sigscale1, sigscale2, sigscale3);
        fprintf('\n sigma_mul = %6.3f, sigma_siter = %6.3f, sigma_giter = %6.3f',...
            sigma_mul0, sigma_siter, sigma_giter);
        fprintf('\n sigma_iter1 = %6.3f, \n sigma0 = %6.3f',sigma_iter, sigma);
    end
    fprintf('\n ******************************************************');
    fprintf('*******************************************\n');
    fprintf('\n problem size: n = %3.0f, p = %3.0f, q = %3.0f, normd_scale = %3.2e',n, p, q, normd_scale);
    if computeW == 1
        fprintf(', largesteigk = %3.0f', par.k);
    end
    fprintf('\n ---------------------------------------------------');
    fprintf('\n  iter|  pinforg   dinforg   comporg   relgap  |    pobj       dobj      |');
    fprintf('  time |  sigma   |  step  ');
    if computeW == 0
        fprintf('  |  tolCG |  [res1  iter1 ok1]');
    end
    fprintf('   relkkt ');
    %----------test----------------
    if (testsigma == 1)
        fprintf(' [pfeas     dfeascomp]');
        fprintf(' [mfsr     max(mfsr)]');
    end
    if stop_criterion
        fprintf('   relobj');
    end
    %------------------------------
end
%%
%% initial primal and dual infeasibilities
%%
W_new = W0; b_new = b0; v_new = v0; U_new = U0; Lam_new = Lam0; lam_new = lam0;
invd_lam = invd_scale.*lam_new; dv = d_scale.*v_new;
if flag_initial0 == 1
    ATlam = Z0pq;
    normATlam = 0;
    ATlam_Lam = ATlam;
    AW_new = z0n; W_U_new = Z0pq;
    normW = 0; normU = 0; normlam_org = 0; normLam = 0;
    normlam = 0;
    %    nunormU = 0;
else
    ATlam = ATzfun1(A,y0,lam_new,par.p,par.q); cntATz = cntATz + 1;
    normATlam = norm(full(ATlam),'fro');
    ATlam_Lam = ATlam + Lam_new;
    AW_new = AYfun(A,y0,W_new); cntAY = cntAY + 1;
    W_U_new = zeros(p,q);
    normW = norm(W_new,'fro'); normU = norm(U_new,'fro'); normlam = norm(lam_new);
    normlam_org = norm(invd_lam); normLam = norm(Lam_new,'fro');
end
par.AW_new = AW_new;

% primal feasibility, dual feasibility, duality gap
if flag_initial0 == 1
    primfeas_org = 0; dualfeas_org = 0;
    primobj =  par.C*sum(max(d_scale.*v_new,0));
    dualobj = 0;
    res_gap = abs(primobj)/(1 + primobj);
    
    primfeas = 0; dualfeas = 0;
else
    AW_by_en = AW_new + b_new*y - en;
    primfeas1_org = norm(d_scale.*(AW_by_en + v_new))/const_n;
    primfeas2_org = norm(W_U_new,'fro')/(1 + normW + normU);
    primfeas_org = max(primfeas1_org,primfeas2_org);
    
    dualfeas1_org = abs(y'*lam_new)/const_n;
    dualfeas2_org = norm(invd_lam + projBox(-invd_lam,par.C))/(1 + normlam_org);
    dualfeas3_org = norm(Lam_new - projspball(Lam_new,par.tau),'fro')/(1 + normLam);
    num_projsp = num_projsp + 1;
    dualfeas_org = max([dualfeas1_org,dualfeas2_org,dualfeas3_org]);
    
    AW_by_en = AW_new + b_new*y - en;
    primobj = 0.5*norm(W_new,'fro')^2 + par.tau*sum(svd(full(W_new))) + par.C*sum(max(-d_scale.*AW_by_en,0));
    dualobj = -0.5*norm(ATlam_Lam,'fro')^2 - sum(invd_lam);
    res_gap = abs(primobj - dualobj)/(1 + primobj + abs(dualobj));
    
    primfeas1 = norm(AW_new + b_new*y + v_new - en)/const_n;
    primfeas2 = primfeas2_org; primfeas = max(primfeas1,primfeas2);
    
    dualfeas1 = dualfeas1_org; dualfeas3 = dualfeas3_org;
    dualfeas2 = norm(lam_new + projBox(-lam_new,par.C))/(1 + normlam);
    dualfeas = max([dualfeas1,dualfeas2,dualfeas3]);
end
maxfeas = max(primfeas,dualfeas);
% relative KKT residual
if flag_initial0 == 1
    eta_W_org = 0; eta_b_org = 0;
    eta_v_org = norm(invd_lam + projBox(dv-invd_lam,par.C))/(1 + normlam_org + norm(dv));
    eta_U_org = 0; eta_lam_org = 0; eta_Lam_org = 0;
    res_kkt = eta_v_org;
else
    eta_W_org = norm(W_new + ATlam_Lam,'fro')/(1 + normW + normATlam + normLam);
    eta_b_org = dualfeas1_org;
    eta_v_org = norm(invd_lam + projBox(dv-invd_lam,par.C))/(1 + normlam + norm(dv));
    eta_U_org = norm(Lam_new - projspball(U_new + Lam_new,par.tau),'fro')/(1 + normLam + normU);
    num_projsp = num_projsp + 1;
    eta_lam_org = primfeas1_org;
    eta_Lam_org = primfeas2_org;
    
    res_kkt = max([eta_W_org, eta_b_org, eta_v_org, eta_U_org, eta_lam_org, eta_Lam_org]);
end
rel_comp_org = max([eta_W_org,eta_v_org,eta_U_org]);
eta_W = eta_W_org; eta_U = eta_U_org;
eta_v = norm(lam_new + projBox(v_new-lam_new,par.C))/(1 + normlam + norm(v_new));
rel_comp = max([eta_W,eta_v,eta_U]);

max_dfeas_comp = max(dualfeas,rel_comp);
feasratio = primfeas/max_dfeas_comp;
if feasratio == 0
    maxfeasratio = inf;
else
    maxfeasratio = max(feasratio,1/feasratio);
end
iter = 0; ttime = etime(clock,tstart);
if (printyes)
    fprintf('\n %5.0f| %3.2e  %3.2e  %3.2e %- 3.2e | %- 5.4e %- 5.4e |',...
        iter,primfeas_org,dualfeas_org,rel_comp_org,res_gap,primobj,dualobj);
    fprintf(' %5.1f | %3.2e | %2.4f ',ttime, par.sigma, par.steplen);
    if computeW == 1
        fprintf('  %3.2e',res_kkt);
    else
        fprintf('                              %3.2e',res_kkt);
    end
    %----------test----------------
    fprintf(' [%3.2e   %3.2e] [%3.2e  %3.2e]',primfeas,max_dfeas_comp,feasratio,maxfeasratio);
    if stop_criterion
        relobj = abs(primobj - optval)/(1 + abs(optval));
        fprintf('   %3.2e', relobj);
    end
    %------------------------------
end


%%
%% main Loop
%%
for iter = 1:maxiter
    %% Compute b_bar
    tmpb = v_new - en + lam_new/par.sigma;
    yTtmpb = y'*tmpb/yTy;
    b_bar = -y'*AW_new/yTy - yTtmpb;
    %% Compute W_new
    tmpW = b_bar*y + tmpb;
    if computeW  % Method I: direct method
        AT_tmp_AW = ATzfun1(A,y0,tmpW + AW_new,par.p,par.q); cntATz = cntATz + 1;
        rhsW_bar = par.sigma*(U_new - AT_tmp_AW - W_new) - Lam_new;
        rhsW_bar  = rhsW_bar + par.sigma*(dtmp_smal*W_new + reshape(Vtmp*spdiags(dtmp-dtmp_smal,0,k,k)*(W_new(:)'*Vtmp)',p,q));
        d_par = 1./(1+par.sigma*dtmp) - 1/(1+par.sigma*dtmp_smal);
        W_new = rhsW_bar/(1+par.sigma*dtmp_smal) + reshape(Vtmp*spdiags(d_par,0,k,k)*(rhsW_bar(:)'*Vtmp)',par.p,par.q);
    else % Method II: psqmr
        ATw_tmp_bar = ATzfun1(A,y0,tmpW,par.p,par.q); cntATz = cntATz + 1;
        rhsW_bar = par.sigma*(U_new - ATw_tmp_bar) - Lam_new;
        tol_sGS = 3*max(0.9*tol,min(1/iter^1.1,0.9*maxfeas)); %tol_sGS = min(1e-2,1/iter^1.1);
        if (primfeas > 1e-3) || (iter <= 5)
            maxitpsqmr = max(maxitpsqmr,200);
        elseif (primfeas > 1e-4)
            maxitpsqmr = max(maxitpsqmr,300);
        elseif (primfeas > 1e-5)
            maxitpsqmr = max(maxitpsqmr,400);
        elseif (primfeas > 5e-6)
            maxitpsqmr = max(maxitpsqmr,500);
        end
        par.minitpsqmr = 0;
        par.x0 = W_new; par.normx0 = normW;
        par.precond = 0;
        
        [W_new,resnrm,solve_ok] = mypsqmr_sGS_isPADMM(A,y0,par,rhsW_bar,tol_sGS,maxitpsqmr,print_psqmr,Afun);
        psqmriter = length(resnrm) - 1;
        runhist.psqmriter(iter) = psqmriter;
        runhist.resnrm(iter) = resnrm(end);
    end
    
    %% Compute b_new
    AW_new = AYfun(A,y0,W_new); cntAY = cntAY + 1; par.AW_new = AW_new;
    b_new = -y'*AW_new/yTy - yTtmpb;
    
    %% Compute v_new and U_new
    AW_by_en = AW_new + b_new*y - en;
    tmpv = -lam_new - par.sigma*(AW_new + b_new*y - en);
    v_new = (1/par.sigma)*(tmpv - projBox(tmpv,Cd));
    
    tmpU = Lam_new + par.sigma*W_new;
    U_new = (1/par.sigma)*(tmpU - projspball(tmpU,tau));
    num_projsp = num_projsp + 1;
    
    %% update mutilpliers lam and Lam
    W_U_new = W_new - U_new; AW_by_v_en = AW_by_en + v_new;
    lam_new = lam_new + (par.steplen*par.sigma)*(AW_by_v_en);
    Lam_new = Lam_new + (par.steplen*par.sigma)*W_U_new;

    %%---------------------------------------------------------
    %% check for termination
    %%---------------------------------------------------------
    ttime = etime(clock,tstart); computekkt = 0; computeobj = 0;

    if stop_criterion
        dv = d_scale.*v_new;
        primobj = 0.5*norm(W_new,'fro')^2 + par.tau*sum(svd(full(W_new))) + par.C*sum(max(-d_scale.*AW_by_en,0));
        relobj = abs(primobj - optval)/(1 + abs(optval));
        if relobj < tol
            breakyes = 1;
            msg = 'relobj converged';
            info.termcode = 1;
            computeobj = 1;
        end
    end

    %%------- primal and dual infeasibility under scaling----------
    if (iter <= 200)
        print_iter = sigma_iter;
    elseif (iter <= 2000)
        print_iter = sigma_iter;
    else
        print_iter = sigma_iter;
    end
    if  (stop_criterion == 0) || (computeW == 0) || iter == maxiter || (breakyes) || ((fixsigma == 0) && (rem(iter,sigma_iter) == 0)) %|| rem(iter,print_iter)==0
        normW = norm(W_new,'fro'); normU = norm(U_new,'fro');
        normLam = norm(Lam_new,'fro'); normlam = norm(lam_new);
        primfeas1 = norm(AW_by_v_en)/const_n;
        primfeas2 = norm(W_U_new,'fro')/(1 + normW + normU);
        primfeas = max(primfeas1,primfeas2);
        
        dualfeas1 = abs(y'*lam_new)/const_n;
        dualfeas2 = norm(lam_new + projBox(-lam_new,Cd))/(1 + normlam);%norm(lam_new + projBox(-lam_new,par.C))/(1 + normlam);
        dualfeas3 = norm(Lam_new - projspball(Lam_new,par.tau),'fro')/(1 + normLam);
        num_projsp = num_projsp + 1;
        dualfeas = max([dualfeas1,dualfeas2,dualfeas3]);
        
        maxfeas = max(primfeas,dualfeas);
        
        % record history
        runhist.dualfeas(iter) = dualfeas;
        runhist.primfeas(iter) = primfeas;
        runhist.maxfeas(iter) = maxfeas;
        runhist.sigma(iter) = par.sigma;
    end
        
    if ((stop_criterion == 0) && (warm==0) && ( (maxfeas < tol_maxfeas) || (ttime > maxtime) || (iter == maxiter))) %|| (breakyes)
        ATlam = ATzfun1(A,y0,lam_new,par.p,par.q); normATlam = norm(ATlam,'fro'); cntATz = cntATz + 1;
        %-----recover scaling--------
        invd_lam = invd_scale.*lam_new;
        normlam_org = norm(invd_lam);
        ATlam_Lam = ATlam + Lam_new;
        eta_W_org = norm(W_new + ATlam_Lam,'fro')/(1 + normW + normATlam + normLam);
        eta_b_org = dualfeas1;
        eta_v_org = norm(invd_lam + projBox(dv-invd_lam,par.C))/(1 + normlam_org + norm(dv));
        eta_U_org = norm(Lam_new - projspball(U_new + Lam_new, par.tau),'fro')/(1 + normLam + normU);
        num_projsp = num_projsp + 1;
        eta_lam_org =  norm(d_scale.*AW_by_v_en)/const_n;
        eta_Lam_org = primfeas2;
        res_kkt = max([eta_W_org, eta_b_org, eta_v_org, eta_U_org, eta_lam_org, eta_Lam_org]);
        computekkt = 1;
        if res_kkt < tol
            breakyes = 1;
            msg = 'relkkt converged';
            info.termcode = 1;
        end
        rel_comp_org = max([eta_W_org,eta_v_org,eta_U_org]);
    end
    
    if (stop_criterion == 0) && (warm == 1)
        if res_kkt < tol
            breakyes = 1;
            msg = 'relkkt converged';
            info.termcode = 1;
        end
    end
    
    %%--------------------------------------------------------
    %% print results
    %%--------------------------------------------------------
    compute_resgap = 0; comput_primfeas_org = 0;
    if  iter== maxiter || (breakyes) || ((fixsigma == 0) && (rem(iter,sigma_iter) == 0)) %|| rem(iter,print_iter)==0 
        if computekkt == 0
            ATlam = ATzfun1(A,y0,lam_new,par.p,par.q); cntATz  = cntATz + 1;
            ATlam_Lam = ATlam + Lam_new;
        end
        invd_lam = invd_scale.*lam_new;
        if computeobj == 0
            dv = d_scale.*v_new;
            primobj = 0.5*norm(W_new,'fro')^2 + par.tau*sum(svd(full(W_new))) + par.C*sum(max(-d_scale.*AW_by_en,0));
        end
        dualobj = -0.5*norm(ATlam_Lam,'fro')^2 - sum(invd_lam);
        res_gap = abs(primobj - dualobj)/(1 + primobj + abs(dualobj));
        compute_resgap = 1;
        if computekkt == 0
            normlam_org = norm(invd_lam);
            eta_W_org = norm(W_new + ATlam_Lam,'fro')/(1 + normW + norm(ATlam,'fro') + normLam);
            eta_b_org = dualfeas1;
            eta_v_org = norm(invd_lam + projBox(dv-invd_lam,par.C))/(1 + normlam_org + norm(dv));
            eta_U_org = norm(Lam_new - projspball(U_new + Lam_new, par.tau),'fro')/(1 + normLam + normU);
            num_projsp = num_projsp + 1;
            eta_lam_org = norm(d_scale.*(AW_by_v_en))/const_n;
            eta_Lam_org = primfeas2;
            res_kkt = max([eta_W_org, eta_b_org, eta_v_org, eta_U_org, eta_lam_org, eta_Lam_org]);
            computekkt = 1;
            rel_comp_org =  max([eta_W_org,eta_v_org,eta_U_org]);
        end
        eta_W = eta_W_org; eta_U = eta_U_org;
        eta_v = norm(lam_new + projBox(v_new-lam_new,Cd))/(1 + normlam + norm(v_new));
        rel_comp = max([eta_W,eta_v,eta_U]);
        max_dfeas_comp = max(dualfeas,rel_comp);
        feasratio = primfeas/max_dfeas_comp;
        maxfeasratio = max(feasratio,1/feasratio);
        runhist.maxfeasratio(iter) = maxfeasratio;
        runhist.max_dfeas_comp(iter) = max_dfeas_comp;
        comput_primfeas_org = 1;
        %-------------recover primfeas_org and dualfeas_org---------------
        primfeas1_org = eta_lam_org; primfeas2_org = primfeas2;
        dualfeas1_org = dualfeas1; dualfeas3_org = dualfeas3;
        dualfeas2_org = norm(invd_lam + projBox(-invd_lam,par.C))/(1 + normlam_org);
        primfeas_org = max(primfeas1_org,primfeas2_org);
        dualfeas_org = max([dualfeas1_org,dualfeas2_org,dualfeas3_org]);
        %-----------------------------------------------------------------
        
        if (printyes)
            fprintf('\n %5.0d| %3.2e  %3.2e  %3.2e %- 3.2e | %- 5.4e %- 5.4e |',...
                iter,primfeas_org,dualfeas_org,rel_comp_org,res_gap,primobj,dualobj);
            fprintf(' %5.1f | %3.2e | %2.4f  ',ttime, par.sigma,par.steplen);
            if computeW == 0
                fprintf('| %3.1e  [%3.1e %3.0f %3.0d]', tol_sGS, resnrm(end), psqmriter, solve_ok);
            end
            fprintf(' %3.2e',res_kkt);
            %-----------test------------
            if (testsigma == 1)
                fprintf(' [%3.2e   %3.2e]',primfeas,max_dfeas_comp);
                fprintf(' [%3.2e  %3.2e]',feasratio,maxfeasratio);
            end
            if stop_criterion
                fprintf('  %3.2e', relobj);
            end
        end
        
        runhist.primobj(iter) = primobj;
        runhist.dualobj(iter) = dualobj;
        runhist.time(iter) = ttime;
        runhist.relgap(iter) = res_gap;
        runhist.relkkt(iter) = res_kkt;
        runhist.rel_comp(iter) = rel_comp;
        runhist.primfeas_org(iter) = primfeas_org;
        runhist.dualfeas_org(iter) = dualfeas_org;
        runhist.rel_comp_org(iter) = rel_comp_org;
    end
    
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
    if computekkt == 1
        runhist.eta_W(iter) = eta_W_org;
        runhist.eta_b(iter) = eta_b_org;
        runhist.eta_v(iter) = eta_v_org;
        runhist.eta_U(iter) = eta_U_org;
        runhist.eta_lam(iter) = eta_lam_org;
        runhist.eta_Lam(iter) = eta_Lam_org;
        
        runhist.primobj(iter) = primobj;
        runhist.dualobj(iter) = dualobj;
        runhist.time(iter) = ttime;
        runhist.relgap(iter) = res_gap;
        runhist.relkkt(iter) = res_kkt;
    end
    if (breakyes > 0)
        if stop_criterion
            fprintf('\n  breakyes = %3.1f, %s, relobj = %3.2e',breakyes,msg,relobj);
        else
            fprintf('\n  breakyes = %3.1f, %s, relkkt = %3.2e',breakyes,msg,res_kkt);
        end
        break;
    end
    
    %%-----------------------------------------------------------
    %% update penalty parameter sigma
    %%-----------------------------------------------------------
    if (iter < 6800)
        sigma_mul = sigma_mul0;
    elseif (iter < 10000)
        sigma_mul = sigma_mul0;
    else
        sigma_mul = sigma_mul0;
    end
    
    if (fixsigma == 0) && (rem(iter,sigma_iter) == 0) %&& (iter < 6800) 
        if maxfeasratio <= sigma_siter
            sigmascale = sigscale1;
        elseif maxfeasratio > sigma_giter
            sigmascale = sigscale3;
        else
            sigmascale = sigscale2;
        end
        
        if feasratio > sigma_mul
            par.sigma = par.sigma*sigmascale;
        elseif feasratio < 1/sigma_mul
            par.sigma = par.sigma/sigmascale;
        end
        runhist.feasratio(iter) = feasratio;
        runhist.maxfeasratio(iter) = maxfeasratio;
    end
    
end
%------------------------------End dADMM main loop-------------------------
%%-----------------------------------------------------------------
%% recover original variables
%%-----------------------------------------------------------------
v_new = d_scale.*v_new;
lam_new = invd_scale.*lam_new;
normlam = norm(lam_new);

if computekkt == 0
    ATlam = ATzfun1(A0,y0,lam_new,par.p,par.q); cntATz  = cntATz + 1;
    ATlam_Lam = ATlam + Lam_new;
end
if compute_resgap == 0
    primobj = 0.5*norm(W_new,'fro')^2 + par.tau*sum(svd(full(W_new))) + par.C*sum(max(-d_scale.*AW_by_en,0));
    dualobj = -0.5*norm(ATlam_Lam,'fro')^2 - sum(lam_new);
    res_gap = abs(primobj - dualobj)/(1 + primobj + abs(dualobj));
end
if computekkt == 0  
    eta_W_org = norm(W_new + ATlam_Lam,'fro')/(1 + normW + norm(ATlam,'fro') + normLam);
    eta_b_org = dualfeas1;
    eta_v_org = norm(lam_new + projBox(v_new-lam_new,par.C))/(1 + normlam + norm(v_new));
    eta_U_org = norm(Lam_new - projspball(U_new + Lam_new, par.tau),'fro')/(1 + normLam + normU);
    num_projsp = num_projsp + 1;
    eta_lam_org = norm(d_scale.*(AW_by_v_en))/const_n;
    eta_Lam_org = primfeas2;
    res_kkt = max([eta_W_org, eta_b_org, eta_v_org, eta_U_org, eta_lam_org, eta_Lam_org]);
    rel_comp_org = max([eta_W_org,eta_v_org,eta_U_org]);
end

if comput_primfeas_org == 0
    primfeas1_org = eta_lam_org; primfeas2_org = primfeas2;
    dualfeas1_org = dualfeas1; dualfeas3_org = dualfeas3;
    dualfeas2_org = norm(lam_new + projBox(-lam_new,par.C))/(1 + normlam);
    primfeas_org = max(primfeas1_org,primfeas2_org);
    dualfeas_org = max([dualfeas1_org,dualfeas2_org,dualfeas3_org]);
end
ttime = etime(clock,tstart);
ttime_cpu = cputime - tstart_cpu;
if computeW == 1
    ttCG = 0;
else
    ttCG =  sum(runhist.psqmriter);
end

if (testsigma == 1)
    if iter == 1
        info.max_maxfeasratio = inf;
        info.min_maxfeasratio = inf;
    else
        info.max_maxfeasratio = max(runhist.maxfeasratio(2:end));
        info.min_maxfeasratio = min(runhist.maxfeasratio(2:end));
    end
end
if computeW == 0
    if flag_initial0 == 1
        info.cntATz = cntATz + 2*iter + ttCG - 1;
        info.cntAY = cntAY + 2*iter + ttCG - 1;
    else
        info.cntATz = cntATz + 2*iter + ttCG;
        info.cntAY = cntAY + 2*iter + ttCG;
    end
else
    info.cntATz = cntATz;
    info.cntAY = cntAY;
end
if (printyes)
    if ~isempty(msg); fprintf('\n %s',msg); end
    fprintf('\n--------------------------------------------------------------');
    fprintf('------------------');
    fprintf('\n  number iter = %2.0d',iter);
    fprintf('\n  time = %3.2f',ttime);
    fprintf('\n  time per iter = %5.4f',ttime/iter);
    fprintf('\n  cputime = %3.2f', ttime_cpu);
    fprintf('\n  primobj = %9.8e, dualobj = %9.8e, relgap = %3.2e',primobj,dualobj, res_gap);
    fprintf('\n  primfeas_org = %3.2e, dualfeas_org = %3.2e, relcomp_org = %3.2e,  relkkt = %3.2e',...
        primfeas_org, dualfeas_org, rel_comp_org, res_kkt);
    fprintf('\n  Total CG number = %3.0f, CG per iter = %3.1f', ttCG, ttCG/iter); % notice
    fprintf('\n number of ATz = %3.0d, number of AY = %3.0d', info.cntATz,info.cntAY);
    fprintf('\n num_projsp = %2.0d', num_projsp);
    if (testsigma == 1)
        fprintf('\n  max(maxfeasratio) = %3.2e', info.max_maxfeasratio);
        fprintf('\n  min(maxfeasratio) = %3.2e', info.min_maxfeasratio);
    end
    if stop_criterion
        fprintf('\n relative objective residual = %3.2e',relobj);
    end
    fprintf('\n--------------------------------------------------------------');
    fprintf('------------------\n');
end

info.iter = iter;
info.totaltime = ttime;
info.totaltime_cpu = ttime_cpu;
info.primfeas_org = primfeas_org;
info.dualfeas_org = dualfeas_org;
info.rel_comp_org = rel_comp_org;
info.maxfeas = max(primfeas_org,dualfeas_org);
info.relgap = res_gap;
info.primobj = primobj;
info.dualobj = dualobj;
info.relkkt = res_kkt;
info.ttCG = ttCG;
info.num_projsp = num_projsp;
%-----------test---------------
info.primfeas1_org = primfeas1_org;
info.primfeas2_org = primfeas2_org;
info.dualfeas1_org = dualfeas1_org;
info.dualfeas2_org = dualfeas2_org;
info.dualfeas3_org = dualfeas3_org;

info.eta_W_org = eta_W_org;
info.eta_v_org = eta_v_org;
info.eta_U_org = eta_U_org;
%-----------------------------

W = W_new; b = b_new; v = v_new;
U = U_new; lam = lam_new; Lam = Lam_new;
obj = [primobj, dualobj];

info.W = W; info.b = b;
info.v = v; info.U = U;
info.lam = lam; info.Lam = Lam;
if stop_criterion
    info.relobj = relobj;
end
info.sigma_final = par.sigma;
end