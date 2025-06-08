%%******************************************************************************************************
%% ALM-SNCG:
%% Semismooth Newton-CG based augmented Lagrangian method for solving the general support 
%% matrix machine (SMM) model:
%%
%% (P) minimize_{W,b,v,U}  0.5*||W - B||^2_F + (delt/2)*(b-a)^2 + tau*||U||_* + delta^*_S(v)
%%     subject to          AW + b*y + v = en
%%                         W - U = 0
%% where:
%% {X_i,y_i}^n_{i=1} = training samples
%% B in R^{p*q} and a in R = known parameters
%% S := { x in R^n: 0 <= x <= C}
%% W in R^{p*q}, b in R, v in R^n, U in R^{p*q} = unkonwn variables
%% If B = 0 and a = 0, (P) reduces to the standard SMM model:
%%       
%%    minimize_{W,b,v,U}  0.5*||W||^2_F + tau*||U||_* + delta^*_S(v)
%%    subject to          AW + b*y + v = en
%%                        W - U = 0
%%
%% [obj,W,b,runhist,info] = ALMNCG_scaling_relobj_Warm(Ainput,y0,OPTIONS,B,a) 
%% 
%% Input:
%% Ainput = matrix in R^{(p*q)*n} with i-th column X_i(:) for i=1,...,n
%% y0 = class labels in {-1,1}^n
%% OPTIONS.tol = solution accuracy tolerance of (P)
%% OPTIONS.C = parameter C in (P)
%% OPTIONS.tau = parameter tau in (P)
%% OPTIONS.delt = parameter delta in (P)
%% OPTIONS.sigma = initial ALM penalty parameter                  
%% OPTIONS.sigmaiter = sigma update frequency 
%% OPTIONS.sigmascale = scaling factor (>0)
%% (OPTIONS.W0, OPTIONS.b0, OPTIONS.lam0, OPTIONS.Lam0, OPTIONS.lam0,
%% OPTIONS.v0) = initial point for the ALM-SNCG
%% OPTIONS.flag_scaling = 1, scale matrix Ainput
%%                      = 0, do not scale matrix Ainput
%% OPTIONS.optval = objective value from ALM-SNCG with relkkt < 1e-8
%% OPTIONS.warm = 1, use isPADMM for initial point
%%                0, otherwise
%% OPTIONS.sig0_admm = isPADMM initial sigma          
%% OPTIONS.maxiter_admm = isPADMM max iterations
%% OPTIONS.tol_admm = isPADMM accuracy tolerance
%% ALM-SNCG stopping criterion:
%% OPTIONS.stop = 0, use relkkt < tol 
%%              = 1, use relobj < tol
%%              = 2, use relgap < tol
%%
%% Output:
%% obj = [primal objective value, dual objective value]
%% (W, b, info.v, info.U) = output primal solution for (P)
%% (info.lam, info.Lam) =  output dual solution for (P)
%% runhist = a structure containing the run history
%% info.iter = total number of ALM iterations
%% info.numSSNCG = total number of SNCG iterations
%% info.numCG = total number of CG iterations
%% info.res_kkt_final = relative KKT residual
%% info.res_gap_final = relative duality gap
%% info.priminfeas_final_org = relative primal infeasibility
%% info.dualinfeas_final_org = relative dual infeasibility
%% info.relgap = relative residual based on primal and dual infeasibilities, and duality gap
%% info.relobj = relative objective value
%% info.totaltime = total running time
%% 
%% Copyright (c) 2024 by Can Wu, Donghui Li, Defeng Sun
%% For more details, please see Sections 2-3 of the paper: 
%% Support matrix machine: exploring sample sparsity, low rank,
%% and adaptive sieving in high-performance computing
%%******************************************************************************************************

function [obj,W,b,runhist,info] = ALMSNCG(Ainput,y0,OPTIONS,B,a)

%%
%% Input parameters
%%
tol = 1e-6;
C = 1;
tau = 0;
sigma = 1;
maxiter = 500;
maxitersub = 200;
maxtime = 7200;
stagnate_check_psqmr = 20;
maxitpsqmr = 100;
printyes = 1; % print the initial information
printlevel_ALM = 1; % print in ALM
printlevel_SSN = 1; % print in SSNCG
breakyes = 0;
msg = [];

fixsigma = 0;
flag_scaling = 1;
flag_initial = 1; % 0 means the origion as the initial point

sigmaiter = 2;
sigmascale = 1.2;
stop_criterion_ALM = 0;
warm = 0;
sig0_admm = 0.1;
maxiter_admm = 100;
ifrandom = 0;
tol_nSM = -1e-3;
tol_admm = 1e-4;
flag_v0 = 0;

if isfield(OPTIONS,'p'), p = OPTIONS.p; end
if isfield(OPTIONS,'q'), q = OPTIONS.q; end
if nargin < 3
    fprintf('error: the number of inputs should be 3 or 4');
    return;
elseif nargin == 3
    flag_B = 0; B = zeros(p,q); normB = 0; a = 0; delt = 0;
else
    delt = OPTIONS.delt;
    if (norm(B) == 0) && (a == 0) && (delt == 0)
        flag_B = 0; B = zeros(p,q); normB = 0;
    else
        flag_B = 1; warm = 0; normB = norm(B,'fro');
        if isfield(OPTIONS,'delt'), delt = OPTIONS.delt; end
    end
end
if isfield(OPTIONS,'tol'), tol = OPTIONS.tol; end
if isfield(OPTIONS,'C'), C = OPTIONS.C; end
if isfield(OPTIONS,'tau'), tau = OPTIONS.tau; end
if isfield(OPTIONS,'n'), n = OPTIONS.n; else, n = length(y0); end
if isfield(OPTIONS,'sigma'), sigma = OPTIONS.sigma; end
if isfield(OPTIONS,'sigmaiter'), sigmaiter = OPTIONS.sigmaiter; end
if isfield(OPTIONS,'sigmascale'), sigmascale = OPTIONS.sigmascale; end
if isfield(OPTIONS,'W0'), W0 = OPTIONS.W0;else, W0 = zeros(p,q); flag_initial = 0; end
if isfield(OPTIONS,'b0'), b0 = OPTIONS.b0;else, b0 = 0;end
if isfield(OPTIONS,'lam0'), lam0 = OPTIONS.lam0; else, lam0 = zeros(n,1); end
if isfield(OPTIONS,'v0'), v0 = OPTIONS.v0; flag_v0 = 1;end 
if flag_B == 0
    if isfield(OPTIONS,'Lam0'), Lam0 = OPTIONS.Lam0;else, Lam0 = zeros(p,q);end
end
if isfield(OPTIONS,'flag_scaling'), flag_scaling = OPTIONS.flag_scaling; end
if isfield(OPTIONS,'stop'), stop_criterion_ALM = OPTIONS.stop; end
if stop_criterion_ALM == 1, optval = OPTIONS.optval; end
if isfield(OPTIONS,'warm'), warm = OPTIONS.warm; end
if isfield(OPTIONS,'sig0_admm'), sig0_admm = OPTIONS.sig0_admm; end
if isfield(OPTIONS,'maxiter_admm'), maxiter_admm = OPTIONS.maxiter_admm; end
if isfield(OPTIONS,'tol_admm'), tol_admm = OPTIONS.tol_admm; end
if isfield(OPTIONS,'ifrandom'), ifrandom = OPTIONS.ifrandom; end
if isfield(OPTIONS,'printyes'), printyes = OPTIONS.printyes; end
if isfield(OPTIONS,'printlevel_ALM'), printlevel_ALM = OPTIONS.printlevel_ALM; end
if isfield(OPTIONS,'printlevel_SSN'), printlevel_SSN = OPTIONS.printlevel_SSN; end
if isfield(OPTIONS,'tol_nSM'), tol_nSM = OPTIONS.tol_nSM; end
if isfield(OPTIONS,'maxitpsqmr'), maxitpsqmr = OPTIONS.maxitpsqmr; end
if isfield(OPTIONS,'stagnate_check_psqmr')
    stagnate_check_psqmr = OPTIONS.stagnate_check_psqmr;
end
if tau == 0
    flag_tau = 0; % tau = 0  for the SMM model
else
    flag_tau = 1;
end

par.tol = tol;
par.C = C; par.tau = tau;
par.sigma = sigma;
par.n = n; par.p = p; par.q = q;
par.flag_B = flag_B; par.B = B;
par.a = a; par.delt = delt;
par.flag_tau = flag_tau;

tiny = 1e-10;
numSSNCG = 0;
numCG = 0;
cntAY = 0;
cntATz = 0;
num_smallalp = 0;
sum_r_indexJ = 0;
if printyes == 0
    printlevel_SSN = 0; % print in SSNCG
    printlevel_ALM = 0;
end
%%
%% Amap and ATmap
%%
if ifrandom == 1
    flag_scaling = 0;
    A0 = Ainput;
else
    sparsity = 1 - nnz(Ainput)/(n*p*q);
    if sparsity > 0.1  
        A0 = sparse(Ainput);
    else
        A0 = Ainput;
    end
end
%%
%% Generate initial point
%%
tstart = clock;
tstart_cpu = cputime;
en = ones(n,1);
constn = 1 + sqrt(n);
if warm
    OPTIONS_admm.p = p; OPTIONS_admm.q = q;
    OPTIONS_admm.n = n; OPTIONS_admm.tol = tol;
    OPTIONS_admm.fixsigma = 0;
    OPTIONS_admm.stop = 1;
    OPTIONS_admm.flag_scaling = 1;
    OPTIONS_admm.steplen = 1.618;
    OPTIONS_admm.sigma_siter = 50;
    OPTIONS_admm.sigma_giter = 500;
    OPTIONS_admm.sigma = sig0_admm;
    OPTIONS_admm.ifrandom = ifrandom;
    OPTIONS_admm.tol = tol_admm;
    
    OPTIONS_admm.sigma_iter = 5;
    OPTIONS_admm.sigma_mul = 20;
    OPTIONS_admm.sigscale1 = 1.02;
    OPTIONS_admm.sigscale2 = 1.1;
    OPTIONS_admm.sigscale3 = 2;
    
    OPTIONS_admm.maxiter = maxiter_admm;
    OPTIONS_admm.tau = tau;
    OPTIONS_admm.C = C;
    
    OPTIONS_admm.W0 = W0;
    OPTIONS_admm.b0 = b0;
    OPTIONS_admm.lam0 = lam0;
    OPTIONS_admm.Lam0 = Lam0;
    OPTIONS_admm.optval = optval;

    
    if printyes
        fprintf('\n *******************************************************');
        fprintf('******************************************');
        fprintf('\n \t\t  Warm starting: isPADMM  for solving SMM ');
        fprintf('with maxiter = %3.2f and tol = %3.2e', OPTIONS_admm.maxiter, OPTIONS_admm.tol);
        fprintf('\n ******************************************************');
        fprintf('*******************************************\n');
    end
    
    [~,W0,b0,~,info_admm] = isPADMM(Ainput,y0,OPTIONS_admm);

    v0 = -info_admm.AW_by_en;
    lam0 = zeros(OPTIONS_admm.n,1);
    index_v = (info_admm.AW_by_en < -1e-12);
    lam0(index_v) = -OPTIONS_admm.C;
    U0 = info_admm.U; Lam0 = info_admm.Lam;
    AW0 = info_admm.AW_by_en + en - b0*y0; 
    info.relobj_admm = info_admm.relobj;
else
    U0 = W0;
    if flag_initial == 0
        AW0 = zeros(n,1);
        v0 = en;
    else
        AW0 = AYfun(A0,y0,W0); cntAY = cntAY + 1;
        if flag_v0 == 0
            v0 = en - AW0 - b0*y0;
        end
    end
end
%%
%% Scaling A
%%
if flag_scaling == 1
    d_scale = full(max(max(abs(A0))',1));
    invd_scale = 1./d_scale;
    invD = sparse(1:par.n,1:par.n,invd_scale);
    
    A = A0*invD;
    y = invd_scale.*y0;
    en = invd_scale;
    v0 = invd_scale.*v0;
    lam0 = d_scale.*lam0;
    par.Cd = C*d_scale;
    normy = norm(y);
    AW0 = invd_scale.*AW0;
else
    d_scale = ones(par.n,1); invd_scale = d_scale;
    A = A0; y = y0; par.Cd = C;
    normy = sqrt(n);
end

if flag_scaling == 1
    normd_scale = norm(d_scale,'inf');
else
    normd_scale = 0;
end
norm_en = norm(en);
num_svds = 0; num_svd = 0;
%%
%% Print the initial information
%%
if printyes
    fprintf('\n *********************************************************');
    fprintf('*********************************************************');
    fprintf('\n\t\t ALMSNCG for solving SMM with tau = %6.3f and C = %6.3f under scaling = %3.0f and flag_B = %3.0f',...
        par.tau, par.C, flag_scaling, flag_B);
    fprintf('\n *********************************************************');
    fprintf('********************************************************* \n');
    fprintf(' sigmaiter = %3.2f, sigmascale = %3.2f, sigma0 = %3.2f', sigmaiter, sigmascale, par.sigma);
    fprintf('\n *********************************************************');
    fprintf('********************************************************* \n');
    fprintf('\n problem size: p = %3.0f, q = %3.0f, n = %3.0f', par.p, par.q, par.n);
    fprintf('\n initial sigma0 = %g, normd_scale = %3.2e', par.sigma,normd_scale);
    fprintf('\n ----------------------------------------------------------')
    fprintf('\n  iter |  pfeas_org dfeas_org relgap_org |   pobj_org    dobj_org  |');
    fprintf('  time |   sigma  |  relkkt   [primfeas dualfeas]');
end

iter = 0; W_new = W0; b_new = b0; v_new = v0; lam_new = lam0;
normlam = norm(lam_new,'fro');
normW = norm(full(W_new),'fro');
if normW == 0
    AW_new = zeros(par.n,1);
else
    AW_new =AW0;
end
if normlam == 0
    ATlam = zeros(par.p,par.q);
    normATlam = 0;
else
    ATlam = ATzfun(A,y0,lam_new,par.p,par.q);
    normATlam = norm(full(ATlam),'fro');
    cntATz = cntATz + 1;
end

AWbye_new = AW_new + b_new*y - en;
invd_lam = invd_scale.*lam_new; d_v = d_scale.*v_new;
norm_invd_lam = norm(invd_lam);

% Compute the initial relkkt
eta_v_org = norm(invd_lam + projBox(d_v-invd_lam,par.C))/(1 + norm_invd_lam + norm(d_v));
eta_lam_org = norm(d_scale.*(AWbye_new + v_new))/constn;
yTlam = y'*lam_new;

if flag_tau
    U_new = U0; Lam_new = Lam0;
    normLam = norm(full(Lam_new),'fro'); normU = norm(full(U_new),'fro');
    ATlam_Lam = ATlam + Lam_new;
    if flag_B
        eta_W_org = norm(full(W_new + ATlam_Lam + B),'fro')/(1 + normB);
        eta_b_org = abs(yTlam + delt*(b_new - a))/(1+delt*abs(a));
    else
        eta_W_org = norm(full(W_new + ATlam_Lam),'fro')/(1 + normW + normATlam + normLam);
        eta_b_org = abs(yTlam)/constn;
    end
    
    eta_Lam_org = norm(full(W_new - U_new),'fro')/(1 + normW + normU);
    eta_U_org = norm(Lam_new - projspball(U_new + Lam_new,par.tau),'fro')/(1 + normLam + normU); num_svd = num_svd + 1;
    res_kkt_org = max([eta_W_org, eta_b_org, eta_v_org, eta_U_org, eta_lam_org, eta_Lam_org]);
else
    if flag_B
        eta_W_org = norm(full(W_new + ATlam + B),'fro')/(1 + normB);
        eta_b_org = abs(yTlam + delt*(b_new - a))/(1+delt*abs(a));
    else
        eta_W_org = norm(full(W_new + ATlam),'fro')/(1 + normW + normATlam);
        eta_b_org = abs(yTlam)/constn;
    end
    res_kkt_org = max([eta_W_org, eta_b_org, eta_v_org, eta_lam_org]);
end

% Compute the initial primal infeasibility, dual infeasibility and relative duality gap
primfeas1 = norm(AWbye_new + v_new)/(1 + norm_en);

dualfeas1 = (eta_b_org*constn)/(1 + normy);
dualfeas2 = norm(lam_new + projBox(-lam_new,par.C))/(1 + normlam);
dualfeas2_org = norm(invd_lam + projBox(-invd_lam,par.C))/(1 + norm_invd_lam);

if flag_tau
    primfeas2 = eta_Lam_org;
    primfeas = max(primfeas1,primfeas2);
    
    dualfeas3 = norm(Lam_new - projspball(Lam_new,par.tau),'fro')/(1 + normLam); num_svd = num_svd + 1;
    dualfeas = max([dualfeas1,dualfeas2,dualfeas3]);
    dualfeas3_org = dualfeas3;
    primfeas_org = max(eta_lam_org,eta_Lam_org);
    if flag_B
        primobj_org = 0.5*norm(W_new - B,'fro')^2 +(delt/2)*(b_new - a)^2+ par.tau*sum(abs(svd(full(W_new)))) ...
            + par.C*sum(max(-d_scale.*AWbye_new,0));
        dualobj_org = -0.5*norm(full(ATlam_Lam - B),'fro')^2 - lam_new'*en + 0.5*normB^2 ...
            -(0.5/delt)*(yTlam - delt*a)^2 + (delt/2)*a^2;
        dualfeas_org = max([dualfeas3_org,dualfeas2_org]);
    else
        primobj_org = 0.5*norm(W_new,'fro')^2 + par.tau*sum(abs(svd(full(W_new)))) + par.C*sum(max(-d_scale.*AWbye_new,0));
        dualobj_org = -0.5*norm(full(ATlam_Lam),'fro')^2 - lam_new'*en;
        dualfeas_org = max([eta_b_org,dualfeas3_org,dualfeas2_org]);
    end
    X = Lam_new + par.sigma*W_new;
    [ProjX,parPX] = projspball(X,par.tau); num_svd = num_svd + 1;
else
    primfeas = primfeas1;
    dualfeas = max([dualfeas1,dualfeas2]);
    primfeas_org = eta_lam_org;
    
    if flag_B
        primobj_org = 0.5*norm(W_new - B,'fro')^2 + (delt/2)*(b_new - a)^2 + par.C*sum(max(-d_scale.*AWbye_new,0));
        dualobj_org = -0.5*norm(full(ATlam - B),'fro')^2 - lam_new'*en + 0.5*normB^2 -(0.5/delt)*(yTlam - delt*a)^2 + (delt/2)*a^2;
        dualfeas_org = dualfeas2_org;
    else
        primobj_org = 0.5*norm(W_new,'fro')^2 + par.C*sum(max(-d_scale.*AWbye_new,0));
        dualobj_org = -0.5*norm(full(ATlam),'fro')^2 - lam_new'*en;
        dualfeas_org = max([eta_b_org,dualfeas2_org]);
    end
    parPX = [];
end

res_gap_org = abs(primobj_org - dualobj_org)/(1 + primobj_org + abs(dualobj_org));

omega = -lam_new - par.sigma*(AWbye_new);
Projomega = projBox(omega,par.Cd);

ttime = etime(clock,tstart);
if printyes
    fprintf('\n %5.1d |  %3.2e   %3.2e   %- 3.2e| %- 5.4e %-5.4e |',...
        iter, primfeas_org, dualfeas_org, res_gap_org, primobj_org, dualobj_org);
    fprintf(' %5.1f | %3.2e | %3.2e ', ttime, par.sigma, res_kkt_org);
    fprintf(' [%3.2e  %3.2e]', primfeas, dualfeas);
end

%%
%% Main code:ALM
%%
par.flag_svds = 1;
for iter = 1:maxiter
    %%
    %% SNCG --> W_new, b_new
    %%
    omega_snew = omega;  Projomega_sub = Projomega;
    ATlam_sub = ATlam; AWbye_snew = AWbye_new;
    W_snew = W_new; b_snew = b_new; lam_snew = lam_new;
    vsig_snew = omega_snew - Projomega_sub;
    norm_vsig_sqr = norm(vsig_snew)^2;
    v_snew = vsig_snew/par.sigma;
    if flag_tau
        X_snew = X; ProjX_sub = ProjX;
        Usig_snew = X_snew - ProjX_sub;
        norm_Usig_sqr = norm(full(Usig_snew),'fro')^2;
        if flag_B
            gradphiW = -W_new + B + ATzfun(A,y0, Projomega_sub,par.p,par.q) - ProjX_sub; % -gradphiW
            gradphib = y'*Projomega_sub - par.delt*(b_snew - par.a); % -gradphib
            FnormW_Bsqr_old = norm(W_new - par.B,'fro')^2; FnormWsqr_old = norm(W_new,'fro')^2;
            phi_snew_apart = (par.delt/2)*(b_snew - par.a)^2;
        else
            gradphiW = -W_new + ATzfun(A,y0, Projomega_sub,par.p,par.q) - ProjX_sub; % -gradphiW
            gradphib = y'*Projomega_sub; % -gradphib
            FnormW_Bsqr_old = norm(W_new,'fro')^2; FnormWsqr_old = FnormW_Bsqr_old;
            phi_snew_apart = 0;
        end
        phi_snew = -0.5*FnormW_Bsqr_old - (0.5/par.sigma)*(norm(omega_snew)^2 - ...
            norm_vsig_sqr + norm(full(X_snew),'fro')^2 - norm_Usig_sqr) - phi_snew_apart;
        normUsqr = norm_Usig_sqr/(par.sigma^2); Lam_snew = Lam_new;
        normLam_sub = normLam; U_snew = Usig_snew/par.sigma;
    else
        X_snew = [];
        if flag_B
            gradphiW = -W_new + B + ATzfun(A,y0, Projomega_sub,par.p,par.q); % -gradphiW
            gradphib = y'*Projomega_sub - par.delt*(b_snew - par.a); % -gradphib
            FnormW_Bsqr_old = norm(W_new - par.B,'fro')^2; FnormWsqr_old = norm(W_new,'fro')^2;
            phi_snew_apart = (par.delt/2)*(b_snew - par.a)^2;
        else
            gradphiW = -W_new + ATzfun(A,y0, Projomega_sub,par.p,par.q); % -gradphiW
            gradphib = y'*Projomega_sub; % -gradphib
            FnormW_Bsqr_old = norm(W_new,'fro')^2; FnormWsqr_old = FnormW_Bsqr_old;
            phi_snew_apart = 0;
        end
        phi_snew = -0.5*FnormW_Bsqr_old - (0.5/par.sigma)*(norm(omega_snew)^2 - norm_vsig_sqr) - phi_snew_apart;
        normUsqr = 0; Lam_snew = []; normLam_sub = 0; U_snew = [];
    end
    cntATz = cntATz + 1;
    normgradphi = sqrt(max(norm(gradphiW,'fro')^2 + abs(gradphib)^2,0));
    
    parsub = par; subhist.solve_ok = zeros(maxitersub,1); subhist.psqmr = zeros(maxitersub,1);
    normvsqr = norm_vsig_sqr/(par.sigma^2);
    normlam_sub = normlam;  FnormWsqr = FnormWsqr_old; FnormW_Bsqr = FnormW_Bsqr_old;
    
    subhist.primfeas(1) = primfeas;
    subhist.dualfeas(1) = dualfeas;
    
    for itersub = 1:maxitersub
        break_ok = 0;
        subhist.normgradphi(itersub) = normgradphi;
        %% stopping criteria: (A) and (B)
        if flag_tau
            normx_sub = sqrt(FnormWsqr + b_snew^2 +  normvsqr +  normUsqr);
            normz_sub = sqrt(normlam_sub^2 + normLam_sub^2);
            normdeltz_sub = sqrt(norm(lam_snew - lam_new)^2 + norm(full(Lam_snew - Lam_new),'fro')^2);
        else
            normx_sub = sqrt(FnormWsqr + abs(b_snew)^2 +  normvsqr);
            normz_sub = normlam_sub;
            normdeltz_sub = norm(lam_snew - lam_new);
        end
        if flag_B
            normtmp_sub = sqrt(norm(full(W_snew),'fro') + b_snew^2) + normdeltz_sub/parsub.sigma + 1/parsub.sigma;
        else
            normtmp_sub = norm(full(W_snew),'fro') + normdeltz_sub/parsub.sigma + 1/parsub.sigma;
        end
        
        const_stopc = parsub.sigma*(1 + normx_sub + normz_sub)*max(normtmp_sub,1);
        epsilon_sub = 1/(iter^(1.2)); 
        eta_sub = min(1/(iter^(1.2)),0.5); 
        
        tolsub_A = epsilon_sub/const_stopc;
        tolsub_B = eta_sub*normz_sub^2/const_stopc;
        tolsub = min([tolsub_A,tolsub_B,1]);
        if iter == 1
            tolsub = max(tolsub,5e-2);
        else
            tolsub = max(tolsub,primfeas);
        end
        if (itersub > 1)
            if (normgradphi <= tolsub) || (normgradphi < 1e-12)
                if printlevel_SSN
                    msg = 'good termination for ALM';
                    fprintf('\n    %s',msg);
                    fprintf('\n      iters=%2.0d, gradphi=%3.2e, tolsub=%3.2e, const_stopc=%3.2e',...
                        itersub, normgradphi, tolsub, const_stopc);
                end
                break;
            end
        end
        
        if (itersub > 50)
            ratio_gradphi = subhist.normgradphi(itersub-9:itersub)./subhist.normgradphi(itersub-10:itersub-1);
            if (min(ratio_gradphi) > 0.997) && (max(ratio_gradphi) < 1.003)
                if (printlevel_SSN); fprintf('stagnate');end
                break;
            end
        end
       %% ----------------------------------------------------------------
       %% Compute Newton direction by psqmr
        % parameters setting in psqmr
        if (primfeas > 1e-3) || (itersub <= 5)%(itersub <= 5)
            maxitpsqmr = max(maxitpsqmr,200);%max(maxitpsqmr,200);
        elseif (primfeas > 1e-4)
            maxitpsqmr = max(maxitpsqmr,300);
        elseif (primfeas > 1e-5)
            maxitpsqmr = max(maxitpsqmr,400);
        elseif (primfeas > 5e-6)
            maxitpsqmr = max(maxitpsqmr,500);
        end
        parsub.minitpsqmr = 3;
        
        if (primfeas > 1e-4)
            stagnate_check_psqmr = max(stagnate_check_psqmr,20);
        else
            stagnate_check_psqmr = max(stagnate_check_psqmr,50);
        end
        if (itersub > 4 && all(subhist.solve_ok(itersub-[3:-1:1]) <= -1)) ...
                && (dualfeas < 5e-5)
            stagnate_check_psqmr = max(stagnate_check_psqmr,100);
        end
        parsub.stagnate_check_psqmr = stagnate_check_psqmr;
        
        prim_ratio = 0;  dual_ratio = 0;
        if (itersub > 1)
            prim_ratio = primfeas_sub/subhist.primfeas(itersub-1);
            dual_ratio = dualfeas_sub/subhist.dualfeas(itersub-1);
        end
        
        if (iter < 2) && (itersub < 5)
            tolpsqmr = min(1,0.1*normgradphi);
        else
            tolpsqmr = min(1e-1,0.1*normgradphi); % adjusting 1e-1
        end
        if (itersub <= 1)
            const2 = 1;
        else
            const2 = 0.1;
        end
        if (itersub > 1) && (dual_ratio > 0.5 || primfeas_sub > 0.1*subhist.primfeas(1))
            const2 = 0.25*const2;%0.5*const2;
        end
        if (prim_ratio > 1.1)
            const2 = 0.25*const2;
        end
        tolpsqmr =  max(1e-3*tol,const2*tolpsqmr);
        
        %%
        if par.n <= 500
            tau_const1 = 0.01;
        else
            tau_const1 = 1;
        end
        tau_const2 = 0.1; 
        rho = tau_const1*min(tau_const2,normgradphi);
        if (iter == 1) && (parsub.sigma > 20) && (normgradphi > 1e2)
            parsub.rho = (0.01*normgradphi)/(itersub^(1.1));
        else
            parsub.rho = max(rho,par.tol);
        end
        
        parsub = Generatedash_Jacobian(y,omega_snew,parsub,parPX);
        
        if parsub.r_indexJ == 0
            AJTy = sparse(parsub.p,parsub.q);
            if flag_B
                parsub.sigJ_rho = parsub.rho + parsub.delt;
            else
                parsub.sigJ_rho = parsub.rho;
            end
            AJ = []; y0J = []; yJ = [];
        else
            if (itersub > 1) && isequal(parsub.indexJ,parsub.indexJ_old)
                AJ = AJ_old; y0J = y0J_old; yJ = yJ_old; AJTy = AJTy_old;
            else
                AJ = A(:,parsub.indexJ); y0J = y0(parsub.indexJ); yJ = y(parsub.indexJ);
                AJTy = ATzfun(AJ,y0J,yJ,parsub.p,parsub.q,parsub.r_indexJ);
            end
            if flag_B
                parsub.sigJ_rho = parsub.sigma*parsub.yJTyJ + parsub.rho + parsub.delt;
            else
                parsub.sigJ_rho = parsub.sigma*parsub.yJTyJ + parsub.rho;
            end
        end
        sum_r_indexJ = sum_r_indexJ + parsub.r_indexJ;
        
        %% Compute Newton direction --> dW, db by CG
        RhsW = gradphiW - ((parsub.sigma*gradphib)/parsub.sigJ_rho)*AJTy;
        if itersub == 1
            parsub.x0 = sparse(parsub.p,parsub.q);
        else
            parsub.x0 = dW;
        end
        
        if (parsub.k1) || (parsub.r_indexJ > 5000)
            [dW,resnrm,solve_okCG] = mypsqmr(AJ,y0J,yJ,AJTy,parsub,parPX,RhsW,tolpsqmr,maxitpsqmr,0);
        else
            if flag_tau
                if parsub.r_indexJ == 0
                    dW = (1/(1+parsub.sigma))*RhsW;
                else
                    ATJ = y0J'.*AJ;
                    ResW_tmp = (RhsW(:)'*ATJ)';
                    V_tmp = ((parsub.sigma+1)/parsub.sigma)*eye(parsub.r_indexJ) + ((1+parsub.sigma)/parsub.rho)*(yJ*yJ')+ATJ'*ATJ;
                    L_tmp = CholHess(V_tmp);
                    dW_tmp = linsysolvefun(L_tmp,ResW_tmp);
                    dW = (1/(1+parsub.sigma))*(RhsW - reshape(ATJ*dW_tmp,parsub.p,parsub.q));
                end
            else
                if parsub.r_indexJ == 0
                    dW = RhsW;
                else
                    ATJ = y0J'.*AJ;
                    ResW_tmp = (RhsW(:)'*ATJ)';
                    V_tmp = (1/parsub.sigma)*eye(parsub.r_indexJ) + (1/parsub.rho)*(yJ*yJ')+ATJ'*ATJ;
                    L_tmp = CholHess(V_tmp);
                    dW_tmp = linsysolvefun(L_tmp,ResW_tmp);
                    dW = RhsW - reshape(ATJ*dW_tmp,parsub.p,parsub.q);
                end
            end
        end
        
        if parsub.r_indexJ == 0
            db = gradphib/parsub.rho;
            AJdW = [];
        else
            AJdW = AYfun(AJ,y0J,dW,parsub.r_indexJ);
            db = (gradphib - parsub.sigma*y(parsub.indexJ)'*AJdW)/parsub.sigJ_rho;
        end
        if parsub.k1
            psqmriter = length(resnrm) - 1;
            subhist.psqmr(itersub) = psqmriter;
            numCG = numCG + psqmriter;
        end
        
        %% Strongly Wolfe search for stepsize
        steptol = 1e-7;
        [W_snew,b_snew,v_snew,U_snew,lam_snew,Lam_snew,omega_snew,X_snew,Pomega_snew,PX_snew,alpha,iterstep,g0,phi_snew,maxiterfs,AdW,normvsqr,normUsqr,FnormW_Bsqr,parPX] = ...
            myfindstep(A,y0,y,W_snew,b_snew,v_snew,U_snew,lam_snew,Lam_snew,omega_snew,X_snew,dW,db,gradphiW,gradphib,phi_snew,FnormW_Bsqr,steptol,AJdW,parsub);
        
        numSSNCG = numSSNCG + 1;
        cntAY = cntAY + 1;
        if flag_tau
            num_svd = parPX.num_svd + num_svd;
            num_svds = parPX.num_svds + num_svds;
        end
        if alpha < tiny; break_ok = 11; num_smallalp = num_smallalp + 1; end
               
        %% Update primfeas_sub, dualfeas_sub, and gradphi
        ATlam_sub = -ATzfun(A,y0,Pomega_snew,parsub.p,parsub.q);
        cntATz = cntATz + 1;
        if flag_tau
            if flag_B
                gradphiW = -W_snew + B - ATlam_sub - PX_snew;
                gradphib = y'*Pomega_snew - parsub.delt*(b_snew - parsub.a);
            else
                gradphiW = -W_snew - ATlam_sub - PX_snew;
                gradphib = y'*Pomega_snew;
            end
            primfeas2_sub =  norm(full(W_snew - U_snew),'fro')/(1 + sqrt(max(FnormWsqr,0)) + sqrt(max(normUsqr,0)));
            normLam_sub = norm(Lam_snew,'fro');
        else
            if flag_B
                gradphiW = -W_snew + B - ATlam_sub;
                gradphib = y'*Pomega_snew - parsub.delt*(b_snew - parsub.a);
            else
                gradphiW = -W_snew - ATlam_sub;
                gradphib = y'*Pomega_snew;
            end
            primfeas2_sub = 0; normLam_sub = 0;
        end
        normgradphi = sqrt(norm(gradphiW,'fro')^2 + abs(gradphib)^2);
        
        
        AWbye_snew = AWbye_snew + alpha*(AdW + db*y);
        AWbye_v_snew = AWbye_snew + v_snew;
        primfeas1_sub = norm(AWbye_v_snew)/(1 + norm_en);
        primfeas_sub = max(primfeas1_sub,primfeas2_sub);
        
        normlam_sub = norm(lam_snew);
        yTlam_snew = y'*lam_snew;
        dualfeas1_sub = abs(yTlam_snew)/(1 + normy);
        dualfeas_sub = dualfeas1_sub;
        subhist.primfeas(itersub + 1) = primfeas_sub;
        subhist.dualfeas(itersub + 1) = dualfeas_sub;
        if parsub.k1
            subhist.solve_ok(itersub + 1) = solve_okCG;
        end
                
        %% Print results of SSNCG
        subhist.phi_snew(itersub) = -phi_snew;
        if printlevel_SSN
            fprintf('\n\t\t [%2.0d]   [%3.2e  %3.2e] [%3.2e  %3.2e]',...
                itersub, normgradphi, tolsub, primfeas_sub, dualfeas_sub);
            if flag_tau
                fprintf(' rho=%3.2e, g0=%3.2e, r_indexJ=%3.0f [%1.0f %1.0f %1.0f] [%1.0f %1.0f] [%3.2e %2.0f] ',...
                    parsub.rho,-g0, parsub.r_indexJ,parsub.k1,parsub.k2,parsub.p-parsub.k2,parsub.bound_G,parsub.Method_flag,alpha,iterstep);
            else
                fprintf(' rho=%3.2e, g0=%3.2e, r_indexJ=%3.0f [%3.2e %2.0f] ',parsub.rho,-g0, parsub.r_indexJ,alpha,iterstep);
            end
            if itersub > 1
                delphi = subhist.phi_snew(itersub)-subhist.phi_snew(itersub-1);
                fprintf(' delphi=%3.2e ', delphi);
            end
            if parsub.k1
                fprintf('[%3.1e %3.0d %3.0d]', resnrm(end), psqmriter, solve_okCG); 
            end
            if maxiterfs == 1
                fprintf('$');
            end
        end
        if (break_ok > 0)
            break;
        end
        parsub.indexJ_old = parsub.indexJ;
        parsub.r_indexJ_old = parsub.r_indexJ;
        AJ_old = AJ; y0J_old = y0J; yJ_old = yJ; AJTy_old = AJTy;
    end
    %%----------------------- End SSNCG --------------------------
    %%
    %% Update v_new, U_new, lambda_new, Lambda_new
    %%
    runhist.cumuSSN(iter) = numSSNCG;
    runhist.cumuCG(iter) = numCG;
    runhist.cumur_indexJ(iter) =  sum_r_indexJ;
    if iter == 1
        runhist.iterSSN(iter) = numSSNCG;
        runhist.aveCG(iter) = numCG/numSSNCG;
        runhist.aver_indexJ(iter)= sum_r_indexJ/numSSNCG;
    else
        runhist.iterSSN(iter) = numSSNCG - runhist.cumuSSN(iter-1);
        runhist.aveCG(iter) = (numCG - runhist.cumuCG(iter-1))/runhist.iterSSN(iter);
        runhist.aver_indexJ(iter) = (sum_r_indexJ - runhist.cumur_indexJ(iter-1))/runhist.iterSSN(iter);
    end
    
    W_new = W_snew; b_new = b_snew;
    v_new = v_snew;
    lam_new = lam_snew;
    
    AWbye_new = AWbye_snew;
    AWbye_v_new = AWbye_v_snew;
    normW = norm(W_new,'fro');
    %%
    %% Stoping criterion: KKT residual
    %%
    ATlam = ATlam_sub; normATlam = norm(ATlam,'fro');
    normlam = normlam_sub;
    
    primfeas = primfeas_sub; yTlam_new = yTlam_snew;
    dualfeas1 = dualfeas1_sub; dualfeas = dualfeas_sub;
    
    invd_lam = invd_scale.*lam_new; norm_invd_lam = norm(invd_lam);
    d_v = d_scale.*v_new;
    AWbye_v_new_org = d_scale.*AWbye_v_new;
    AWbye_new_org = d_scale.*AWbye_new;
    
    eta_v_org = norm(invd_lam + projBox(d_v-invd_lam,par.C))/(1 + norm_invd_lam + norm(d_v));
    eta_lam_org = norm(AWbye_v_new_org)/constn;
    
    if flag_tau
        U_new = U_snew; Lam_new = Lam_snew; normLam = normLam_sub; ATlam_Lam = ATlam + Lam_new;
        if flag_B
            eta_W_org = norm(W_new + ATlam_Lam - B,'fro')/(1 + normB);
            eta_b_org = abs(yTlam_new + par.delt*(b_new - par.a))/(1+par.delt*abs(par.a));
            primobj_org = 0.5*FnormW_Bsqr + (par.delt/2)*(b_new - par.a)^2 + par.tau*sum(abs(svd(full(W_new)))) + par.C*sum(max(-AWbye_new_org,0));
        else
            eta_W_org = norm(W_new + ATlam_Lam,'fro')/(1 + normW + normATlam + normLam);
            eta_b_org = (dualfeas1*(1 + normy))/constn;
            primobj_org = 0.5*FnormW_Bsqr + par.tau*sum(abs(svd(full(W_new)))) + par.C*sum(max(-AWbye_new_org,0));
        end
        eta_U_org = norm(Lam_new - projspball(U_new + Lam_new,par.tau),'fro')/(1 + normLam + sqrt(max(normUsqr,0)));
        eta_Lam_org = primfeas2_sub; num_svd = num_svd + 1;
        
        res_kkt_org = max([eta_W_org, eta_b_org, eta_v_org, eta_U_org, eta_lam_org, eta_Lam_org]);
    else
        if flag_B
            ATlam_B = ATlam - B;
            eta_W_org = norm(W_new + ATlam_B,'fro')/(1 + normB);
            eta_b_org = abs(yTlam_new + par.delt*(b_new - par.a))/(1+par.delt*abs(par.a));
            primobj_org = 0.5*FnormW_Bsqr + (par.delt/2)*(b_new - par.a)^2 + par.C*sum(max(-AWbye_new_org,0));
        else
            eta_W_org = norm(W_new + ATlam,'fro')/(1 + normW + normATlam);
            eta_b_org = (dualfeas1*(1 + normy))/constn;
            primobj_org = 0.5*FnormW_Bsqr + par.C*sum(max(-AWbye_new_org,0));
        end
        res_kkt_org = max([eta_W_org, eta_b_org, eta_v_org, eta_lam_org]);
    end
    runhist.res_kkt_org(iter) = res_kkt_org;
    runhist.primfeas(iter) = primfeas;
    runhist.dualfeas(iter) = dualfeas;
    
    switch stop_criterion_ALM
        case 0
            if res_kkt_org < tol
                breakyes = 1;
                msg = 'relkkt converged';
                info.termcode = 1;
            end
        case 1
            relobj = abs(primobj_org - optval)/(1 + abs(optval));
            if relobj < tol
                breakyes = 1;
                msg = 'relobj converged';
                info.termcode = 1;
            end
        case 2
            eta_dual_lam = norm(invd_lam + projBox(-invd_lam,par.C))/(1 + norm_invd_lam);
            if flag_tau
                eta_dual_LLam = norm(Lam_new - projspball(Lam_new,par.tau),'fro')/(1 + normLam); num_svd = num_svd + 1;
                eta_prim_org = max(eta_lam_org, eta_Lam_org);
                eta_dual_org = max([eta_b_org,eta_dual_lam, eta_dual_LLam]);
                if flag_B
                    dualobj_org = -0.5*norm(ATlam_Lam - B,'fro')^2 - lam_new'*en + 0.5*normB^2 ...
                        - (0.5/par.delt)*(yTlam_new-par.delt*par.a)^2 + (par.delt/2)*(par.a)^2;
                else
                    dualobj_org = -0.5*norm(ATlam_Lam,'fro')^2 - lam_new'*en;
                end
            else
                eta_prim_org = eta_lam_org;
                eta_dual_org = max(eta_b_org,eta_dual_lam);
                if flag_B
                    dualobj_org = -0.5*norm(ATlam_B,'fro')^2 - lam_new'*en + 0.5*normB^2 ...
                        - (0.5/par.delt)*(yTlam_new-par.delt*par.a)^2 + (par.delt/2)*(par.a)^2;
                else
                    dualobj_org = -0.5*norm(ATlam,'fro')^2 - lam_new'*en;
                end
            end
            
            res_gap_org = abs(primobj_org - dualobj_org)/(1 + primobj_org + abs(dualobj_org));
            relgap = max([eta_prim_org, eta_dual_org,res_gap_org]);
            if relgap < tol
                breakyes = 1;
                msg = 'relgap converged';
                info.termcode = 1;
            end
    end
    %%
    %% Print results of ALM
    %%
    ttime = etime(clock,tstart);
    %if  (iter == maxiter) || (breakyes == 1) || (ttime > maxtime)
    primfeas1_org = eta_lam_org;
    if flag_tau
        primfeas_org = max(primfeas1_org,eta_Lam_org);
        runhist.primfeas2_org(iter) =  eta_Lam_org;
    else
        primfeas_org = primfeas1_org;
    end
    if flag_B
        dualfeas_org = 0;
    else
        dualfeas_org = eta_b_org;
    end

    if stop_criterion_ALM < 2
        if flag_tau
            if flag_B
                dualobj_org = -0.5*norm(ATlam_Lam - B,'fro')^2 - lam_new'*en + 0.5*normB^2 ...
                    - (0.5/par.delt)*(yTlam_new-par.delt*par.a)^2 + (par.delt/2)*(par.a)^2;
            else
                dualobj_org = -0.5*norm(ATlam_Lam,'fro')^2 - lam_new'*en;
            end
        else
            if flag_B
                dualobj_org = -0.5*norm(ATlam_B,'fro')^2 - lam_new'*en + 0.5*normB^2 ...
                    - (0.5/par.delt)*(yTlam_new-par.delt*par.a)^2 + (par.delt/2)*(par.a)^2;
            else
                dualobj_org = -0.5*norm(ATlam,'fro')^2 - lam_new'*en;
            end
        end
        res_gap_org = abs(primobj_org - dualobj_org)/(1 + primobj_org + abs(dualobj_org));
    end

    runhist.primfeas1_org(iter) =  primfeas1_org;
    runhist.dualfeas_org(iter) =  dualfeas_org;
    runhist.res_gap_org(iter) = res_gap_org;
    runhist.primobj_org(iter) = primobj_org;
    runhist.dualobj_org(iter) = dualobj_org;
    runhist.primfeas1_org(iter) =  primfeas1_org;

    if (printlevel_ALM)
        fprintf('\n %5.1d |  %3.2e   %3.2e   %- 3.2e| %- 5.4e %-5.4e |',...
            iter, primfeas_org, dualfeas_org, res_gap_org, primobj_org, dualobj_org);
        fprintf(' %5.1f | %3.2e | %3.2e ', ttime, par.sigma, res_kkt_org);
        fprintf(' [%3.2e  %3.2e]', primfeas, dualfeas);
    end
    %end

    %%
    %% Update sigma
    %%
    if (fixsigma == 0) && (mod(iter,sigmaiter)) == 0
        par.sigma = min(par.sigma*sigmascale,1e6);
    end
    %%
    %% Termination
    %%
    if (iter == maxiter)
        msg = 'maximum iteration reached';
        breakyes = 10;
        info.termcode = 2;
    elseif (ttime > maxtime)
        msg = 'maximum time reached';
        breakyes = 100;
        info.termcode = 3;
    end
    
    if (breakyes > 0)
        if printyes
            fprintf('\n breakyes = %3.1f, %s, res_kkt = %3.2e', breakyes, msg, res_kkt_org);
        end
        if flag_tau
            W = W_new; b = b_new; lam = lam_new; Lam = Lam_new; v = d_v;
        else
            W = W_new; b = b_new; lam = lam_new; v = d_v;
        end
        break;
    end
    
    omega = -lam_new - par.sigma*(AWbye_new);
    Projomega = projBox(omega,par.Cd);
    if flag_tau
        X = Lam_new + par.sigma*W_new;
        [ProjX,parPX] = projspball(X,par.tau);
        num_svd = num_svd + 1;
        if (res_kkt_org > 0.1) && min(par.p, par.q) >= 500
            par.flag_svds = 1;
        else
            par.flag_svds = 0;
        end
    else
        parPX = [];
    end
    
end
%%------------------------------- End ALM ---------------------------------

ttime = etime(clock,tstart);
ttime_cpu = cputime - tstart_cpu;
obj = [primobj_org, dualobj_org];

%%
%% Print results
%%
xi_AS = v - AWbye_v_new_org;
Index0 = (xi_AS > tol_nSM);
nSMM = sum(Index0);
if (printyes)
    if ~isempty(msg); fprintf('\n %s',msg); end
    fprintf('\n----------------------------------------------------------');
    fprintf('------------------------------');
    fprintf('\n number iter = %2.0f',iter);
    fprintf('\n number iter of SSN = %2.0f',numSSNCG);
    fprintf('\n SSN  per iter = %3.0f',numSSNCG/iter);
    fprintf('\n number iter of CG = %3.0f',numCG);
    fprintf('\n PSQMR per iter = %3.1f',numCG/iter);
    fprintf('\n PSQMR per itersub = %3.1f',numCG/numSSNCG);
    fprintf('\n time = %3.2f',ttime);
    fprintf('\n time per iter = %5.4f',ttime/iter);
    fprintf('\n cputime = %3.2f',ttime_cpu);
    fprintf('\n cntAY = %2.0d, cntATz = %2.0d', cntAY, cntATz);
    if flag_tau
        fprintf('\n cntSVD = %2.0d, cntSVDS = %2.0d', num_svd, num_svds);
    end
    fprintf('\n primobj_org = %9.8e, dualobj_org = %9.8e, relgap_org = %3.2e',primobj_org,dualobj_org,res_gap_org);
    fprintf('\n priminfeas_org = %3.2e, dualinfeas_org = %3.2e',primfeas_org, dualfeas_org);
    fprintf('\n priminfeas = %3.2e, dualinfeas = %3.2e',primfeas, dualfeas);
    fprintf('\n relative KKT residual = %3.2e',res_kkt_org);
    fprintf('\n number small alpha = %2.0f',num_smallalp);
    fprintf('\n number of support matrices = %2.0f',nSMM);
    if stop_criterion_ALM == 1
        fprintf('\n relative objective residual = %3.2e',relobj);
    elseif stop_criterion_ALM == 2
        fprintf('\n relative gap residual = %3.2e',relgap);
    end
    fprintf('\n----------------------------------------------------------');
    fprintf('------------------------------');
end

%%
%% Record history
%%
info.W = W;
info.b = b;
info.lam = invd_scale.*lam;
info.v = v;

if flag_tau
    info.Lam = Lam; info.U = U_new;
    info.eta_U_org = eta_U_org;
    info.eta_Lam_org = eta_Lam_org;
    info.normLam = normLam;
    info.num_svd = num_svd;
    info.num_svds = num_svds;
end

info.xi_AS = xi_AS;
info.Index0 = Index0;
info.iter = iter;
info.numSSNCG = numSSNCG;
info.numCG = numCG;
info.aveSSNCG = numSSNCG/iter;
info.aveCG_ALM = numCG/iter;
info.aveCG_SSN = numCG/numSSNCG;
info.res_kkt_final = res_kkt_org;
info.res_gap_final = res_gap_org;
info.priminfeas_final_org = primfeas_org;
info.dualinfeas_final_org = dualfeas_org;
info.priminfeas_final = primfeas;
info.dualinfeas_final = dualfeas;
info.totaltime = ttime;
info.totaltime_cpu = ttime_cpu;
info.sigma = par.sigma;
info.breakyes = breakyes;
info.cntAY = cntAY;
info.cntATz = cntATz;

info.r_indexJ = parsub.r_indexJ;
runhist.indexJ = parsub.indexJ;
info.num_smallalp = num_smallalp;
runhist.iter = iter;
if stop_criterion_ALM == 1
    info.relobj = relobj;
elseif stop_criterion_ALM == 2
    info.relgap = relgap;
end
info.r_indexJ = parsub.r_indexJ;
info.eta_W_org = eta_W_org;
info.eta_b_org = eta_b_org;
info.eta_lam_org = eta_lam_org;
info.eta_v_org = eta_v_org;
info.nSMM = nSMM;
info.normW = normW;
info.k1 = parsub.k1;
info.AWbye_new_org = AWbye_new_org;

end
