%%******************************************************************************************************
%% AS+ALM:
%% An adaptive sieving strategy combined with ALM-SNCG for solving SMM models with a sequence of C
%%
%%    minimize_{W,b,v,U}  0.5*||W||^2_F + tau*||U||_* + delta^*_S(v)
%%    subject to          AW + b*y + v = en
%%                        W - U = 0
%% where:
%% {X_i,y_i}^n_{i=1} = training samples
%% S := { x in R^n: 0 <= x <= C}
%% W in R^{p*q}, b in R, v in R^n, U in R^{p*q} = unkonwn variables
%%
%% [objopt,Wsol,bsol,runhist,info] = AS_ALM(A,y,OPTIONS)
%% 
%% Input:
%% A = matrix in R^{(p*q)*n} with i-th column as the vector X_i(:) for i=1,...,n
%% y = class label vector in R^n
%% OPTIONS.tol = accuracy tolerance for the solution of (P)
%% OPTIONS.C_vec = parameter vector C for (P)
%% OPTIONS.tau = parameter tau in (P)
%% OPTIONS.sigma = initial sigma value in the ALM                     
%% OPTIONS.sigmaiter = frequency of sigma updates
%% OPTIONS.sigmascale = positive scaling factor for updating sigma
%% OPTIONS.flag_scaling = scaling option for matrix A:
%%                        1, scale A
%%                        0, do not scale A   
%% OPTIONS.Test_sigma = sigma adjustment testing option:
%%                      1, test sigma adjustment
%%                      0, skip testing
%% OPTIONS.tol_nSM = nonnegative scalar \hat{varepsilon} in Algorithm 3
%% OPTIONS.ifrandom = Data option:
%%                    1, using random data
%%                    0, using real data
%% Output:
%% objopt = [vector of primal objective values, vector of dual objective values]
%% Wsol = solution path of primal variable W 
%% bsol = solution path of primal variable b
%% runhist = a structure containing the run history
%% runhist.nSMM_path = vector of the number of support matrices at each C
%% runhist.sigma = vector of initial penalty parameters (sigma) in ALM-SNCG at each C
%% runhist.nI_mean = vector of average sample sizes in reduced subproblems at each C  
%% runhist.r_indexJ_ALM = vector of cardinalities of the index set J_1 at each C
%% runhist.relkkt = vector of relative KKT residuals at each C
%% runhist.relgap = vector of relative duality gaps at each C
%% runhist.iterAS = vector of AS iteration counts at each C
%% runhist.iter_ALM = vector of ALM iteration counts at each C
%% runhist.iter_SSN = vector of SNCG iteration counts at each C
%% runhist.time = vector of runtime per C
%% runhist.ttime_path = vector of cumulative runtime per C
%% info.termcode = termination code:
%%                 1, AS converges
%%                 3, maximum iteration reached
%% info.r_indexJ = cardinality of index set J_1;
%% info.num_AindexI = number of submatrices extracted from A
%% 
%% Copyright (c) 2024 by Can Wu, Donghui Li, Defeng Sun
%% For more details, please see Section 5 of the paper: 
%% Support matrix machine: exploring sample sparsity, low rank,
%% and adaptive sieving in high-performance computing
%%******************************************************************************************************

function [objopt,Wsol,bsol,runhist,info] = AS_ALM(A,y,OPTIONS)

%%
%% Input parameters
%%
rng('default');
tol = 1e-4;
tau = 1;
maxiterAS = 20;
sigmaiter = 2;
sigmascale = 1.2;

printyes = 1;
printyes_intALM =0;
ifrandom = 0;
flag_scaling = 0;
tol_nSM = -1e-12;

if isfield(OPTIONS,'tol'), tol = OPTIONS.tol; end
if isfield(OPTIONS,'tau'), tau = OPTIONS.tau; end
if isfield(OPTIONS,'sigma'), sigma_vec = OPTIONS.sigma; end
if isfield(OPTIONS,'n'), n = OPTIONS.n; else, n = length(y0); end
if isfield(OPTIONS,'p'), p = OPTIONS.p; end
if isfield(OPTIONS,'q'), q = OPTIONS.q; end
if isfield(OPTIONS,'C_vec'), C_vec = OPTIONS.C_vec; end
if isfield(OPTIONS,'maxiterAS'), maxiterAS = OPTIONS.maxiterAS; end
if isfield(OPTIONS,'W0'), W0 = OPTIONS.W0;else, W0 = zeros(p,q); end
if isfield(OPTIONS,'b0'), b0 = OPTIONS.b0;else, b0 = 0;end
if isfield(OPTIONS,'lam0'), lam0 = OPTIONS.lam0;else, lam0 = zeros(n,1);end
if isfield(OPTIONS,'Lam0'), Lam0 = OPTIONS.Lam0;else, Lam0 = -W0;end
if isfield(OPTIONS,'v0'), v0 = OPTIONS.v0;else, v0 = ones(n,1);end
if isfield(OPTIONS,'sigmaiter'), sigmaiter = OPTIONS.sigmaiter; end
if isfield(OPTIONS,'sigmascale'), sigmascale = OPTIONS.sigmascale; end
if isfield(OPTIONS,'printyes'), printyes_intALM = OPTIONS.printyes; end
if isfield(OPTIONS,'maxiterAS'), maxiterAS = OPTIONS.maxiterAS; end
if isfield(OPTIONS,'ifrandom'), ifrandom = OPTIONS.ifrandom; end
if isfield(OPTIONS,'flag_scaling'), flag_scaling = OPTIONS.flag_scaling; end
if isfield(OPTIONS,'tol_nSM'), tol_nSM = OPTIONS.tol_nSM; end
lencc = length(C_vec);
Time = zeros(lencc,1);
objopt = zeros(lencc,2);

optionsAS.tau = tau;
optionsAS.tol = tol;
optionsAS.stop = 0; % 0, if relkkt <= tol; 1, if relobj <= tol; 2, if relgap <= tol
optionsAS.sigmaiter = sigmaiter;
optionsAS.sigmascale = sigmascale;
optionsAS.p = p;
optionsAS.q = q;
optionsAS.printyes = printyes_intALM;
if ifrandom
    optionsAS.flag_scaling = 0;
    if n <= 1e5
        hat_epsion = 5e-2;
    else
        hat_epsion = 1e-1;
    end
else
    optionsAS.flag_scaling = flag_scaling;
    hat_epsion = 0.4;
end
%%
%% Amap and ATmap
%%
tstart = clock;
msg = [];
if ifrandom == 0
    sparsity = 1 - nnz(A)/(n*p*q);
    if sparsity > 0.1  % adjusting
        A = sparse(A);
    end
end
%%
%% Initial point
%%
en = ones(n,1);

optionsAS.C = C_vec(1);
optionsAS.sigma = sigma_vec(1);
optionsAS.W0 = W0;
optionsAS.b0 = b0;
optionsAS.U0 = W0;
optionsAS.Lam0 = Lam0;
optionsAS.v0 = v0;
optionsAS.n = n;
optionsAS.lam0 = lam0;

if (printyes)
    fprintf('\n ======================================================================');
    fprintf('\n Prob = %3.1d, C = %6.3f, n = %6.0d, sigma0 = %3.2e', ...
        0, optionsAS.C, n, optionsAS.sigma);
    fprintf('\n ======================================================================');
    fprintf('\n  iter|priminfeas  dualinfeas   relgap |    pobj       dobj      |');
    fprintf(' time |   sigma  | res_kkt  | numALM  numSNCG |   n     r_J    nSM  |  numJ  numJ_hat');
end

[obj_AS, W0, b0, ~, infoAS] = ALMSNCG(A,y,optionsAS);

v0 = infoAS.v; U0 = infoAS.U;
lam0 = infoAS.lam; Lam0 = infoAS.Lam;
xi_AS = en - AYfun(A,y,W0) - b0*y;
nSMM_AS = sum(xi_AS > -tol_nSM);
ttime = etime(clock,tstart);
if (printyes)
    fprintf('\n %5.1d|%3.2e    %3.2e   %- 3.2e| %- 5.4e %- 5.4e |',...
        0,infoAS.priminfeas_final_org,infoAS.dualinfeas_final_org,infoAS.res_gap_final,...
        obj_AS(1),obj_AS(2));
    fprintf('%5.1f | %3.2e |',ttime, optionsAS.sigma);
    fprintf(' %3.2e | %5.1d  %5.1d    | %5.1d %5.1d  %5.1d  |%5.1d   %5.1d',infoAS.res_kkt_final,...
        infoAS.iter, infoAS.numSSNCG, n, infoAS.r_indexJ, nSMM_AS, 0, 0);
end

runhist.iter_ALM(1) = infoAS.iter;
runhist.iter_SSN(1) = infoAS.numSSNCG;
runhist.r_indexJ_ALM(1) = infoAS.r_indexJ;
runhist.sigma(1) = optionsAS.sigma;
runhist.eta_W_AS(1) = infoAS.eta_W_org;
runhist.eta_b_AS(1) = infoAS.eta_b_org;
runhist.eta_v_AS(1) = infoAS.eta_v_org;
runhist.eta_U_AS(1) = infoAS.eta_U_org;
runhist.eta_lam_AS(1) = infoAS.eta_lam_org;
runhist.eta_Lam_AS(1) = infoAS.eta_Lam_org;
Time(1) = ttime;
runhist.time(1) = ttime;
runhist.ttime_path(1) = ttime;
runhist.nSMM_path(1) = nSMM_AS;
runhist.relkkt(1) = infoAS.res_kkt_final;
runhist.relgap(1) = infoAS.res_gap_final;
runhist.iterAS(1) = 0;
runhist.nI_mean(1) = n;
objopt(1,:) = [obj_AS(1),obj_AS(2)];

optionsAS.tol_nSM = tol_nSM;
%---------------test----------------
%% Choose percent*n number of samples
Index0 = (xi_AS > -hat_epsion);
n_I = sum(Index0);
%----------------------------------
W = W0;
b = b0;
lam = lam0;
Lam = Lam0;
v = -v0;
U = U0;

clear W0 b0 lam0 Lam0 v0 U0;

Wsol = zeros(p,q,lencc);
bsol = zeros(lencc,1);
ttime = etime(clock,tstart);
%%
%% Print the initial information
%%
if printyes
    fprintf('\n *********************************************************');
    fprintf('*********************************************************');
    fprintf('\n\t\t Adaptive sketching for solving SMM with tau = %6.3f', tau);
    fprintf('\n *********************************************************');
    fprintf('********************************************************* \n');
    fprintf('\n problem size: p = %3.0f, q = %3.0f, n = %3.0f, init time = %3.2f \n', p, q, n,ttime);
end

%%
%% Main code: AS
%%
num_AindexI = 0;
for cc = 2:lencc
    optionsAS.C = C_vec(cc);
    optionsAS.sigma = sigma_vec(cc);
   
    W_AS = W; b_AS = b;
    v_sub = v;
    U_AS = U;
    lam_sub = lam;
    Lam_AS = Lam;
    
    indexI_AS = Index0;
    n_sub = n_I;
    if (printyes)
        fprintf('\n ======================================================================');
        fprintf('\n Prob = %3.0d, C = %6.3f, n = %6.0d, sigma0 = %3.2e', ...
            cc, optionsAS.C, n_sub,optionsAS.sigma);
        fprintf('\n ======================================================================');
        fprintf('\n  iter|priminfeas  dualinfeas   relgap |    pobj       dobj      |');
        fprintf(' time |   sigma  | res_kkt  | numALM  numSNCG |   n     r_J   nSMM  |  numJ  numJ_hat');
    end
    runhistAS.nI_path = []; runhistAS.iter_ALM = []; runhistAS.iter_SSN = [];
    
    optionsAS.W0 = W_AS;
    optionsAS.b0 = b_AS;
    optionsAS.U0 = U_AS;
    optionsAS.Lam0 = Lam_AS;
    
    n_Isub = n_I;
    %% Begin the inner loop of the AS
    for iterAS = 1:maxiterAS
        
        runhistAS.nI_path(iterAS) = n_Isub;
        
        % Compute Asub and ysub
        % Test: obtain Asub (indexI_AS) from A_old (indexI_AS_old)
        if n <= 1e5
            if (cc >= 3) && (iterAS == 1) && all((indexI_AS & indexI_AS_old) == indexI_AS)
                indexI_AS_sub = indexI_AS(indexI_AS_old);
                Asub = Asub(:,indexI_AS_sub);
                num_AindexI = num_AindexI + 1;
            else
                Asub = A(:,indexI_AS);
            end
            ysub = y(indexI_AS);
        end
        
        % Compute the terms of the reduced SMM problem
        lam_AS = lam_sub(indexI_AS);
        v_AS = v_sub(indexI_AS);
        ComindexI_AS = ~indexI_AS; %n_I = sum(indexI_AS);
        
        optionsAS.n = n_Isub;
        optionsAS.v0 = v_AS;
        optionsAS.lam0 = lam_AS;
        if n <= 1e5
            [obj_AS,W_AS,b_AS,~,infoAS] = ALMSNCG(Asub,ysub,optionsAS);
        else
            [obj_AS,W_AS,b_AS,~,infoAS] = ALMSNCG(A(:,indexI_AS),y(indexI_AS),optionsAS);
        end
        
        lam_AS = infoAS.lam;
        v_AS = infoAS.v;
        Lam_AS = infoAS.Lam;
        U_AS = infoAS.U;

        optionsAS.W0 = infoAS.W;
        optionsAS.b0 = infoAS.b;
        optionsAS.U0 = infoAS.U;
        optionsAS.Lam0 = infoAS.Lam;
        
        % update lam_sub and v_sub
        lam_sub(indexI_AS) = lam_AS;
        v_sub(indexI_AS) = v_AS;
        xi_AS = en - AYfun(A,y,W_AS) - b_AS*y;
        xi_comI_AS = xi_AS(ComindexI_AS);
        v_sub(ComindexI_AS) = xi_comI_AS;
        
        IndexJ = (v_sub > tol_nSM) & ComindexI_AS;%(v_sub > -1e-12) & ComindexI_AS;
        numJ = sum(IndexJ);
        
        lam_sub(IndexJ) = -optionsAS.C;
        lam_sub(~IndexJ & ComindexI_AS) = 0;
        
        Index0 = (xi_AS > tol_nSM);
        nSMM_AS = sum(Index0);
        indexI_AS_old = indexI_AS;
        
        
        % Compute the index set IndexJ_hat
        if numJ > 0
            if numJ <= 500
                indexJ_hat_AS = IndexJ;
            else
                dnum_AS = min([numJ, 500]);% adjusting
                Res_tmp = sort(v_sub(IndexJ),'descend');
                cnum_Res = Res_tmp(dnum_AS);
                
                indexJ_hat_AS = (v_sub >= max(cnum_Res,tol_nSM));%(v_sub >= max(cnum_Res,-1e-12)); tol_nSM
                indexJ_hat_AS = (indexJ_hat_AS & IndexJ);
            end
            numJ_hat_AS = sum(indexJ_hat_AS);
            indexI_AS = (indexI_AS | indexJ_hat_AS);
        else
            numJ_hat_AS = 0;
        end
        
        % Print result of the AS
        if (printyes)
            fprintf('\n %5.1d|%3.2e    %3.2e   %- 3.2e| %- 5.4e %- 5.4e |',...
                iterAS,infoAS.priminfeas_final_org,infoAS.dualinfeas_final_org,infoAS.res_gap_final,...
                obj_AS(1),obj_AS(2));
            fprintf('%5.1f | %3.2e |',infoAS.totaltime, optionsAS.sigma);
            fprintf(' %3.2e | %5.1d  %5.1d    | %5.1d %5.1d  %5.1d  |%5.1d   %5.1d',infoAS.res_kkt_final,...
                infoAS.iter, infoAS.numSSNCG, n_Isub, infoAS.r_indexJ, nSMM_AS, numJ, numJ_hat_AS);
        end
        
        n_Isub = sum(indexI_AS);
        runhistAS.iter_ALM(iterAS) =  infoAS.iter;
        runhistAS.iter_SSN(iterAS) = infoAS.numSSNCG;
        runhistAS.sigma_vec(iterAS) = infoAS.sigma; % Test
        
        % Stopping criterion for the AS
        if (iterAS == maxiterAS) || (numJ == 0)
            if  (numJ == 0)
                msg = 'AS converges';
                info.termcode = 1;
            else
                msg = ' maximum iteration reached';
                info.termcode = 3;
            end
            break;
        end
    end
    
    %% End the Loop for inner AS
    
    runhist.iter_ALM(cc) = sum(runhistAS.iter_ALM);
    runhist.iter_SSN(cc) = sum(runhistAS.iter_SSN);
    runhist.r_indexJ_ALM(cc) = infoAS.r_indexJ;
    runhist.sigma(cc) = optionsAS.sigma;
    
    % ------------------------ Compute the relkkt of original problem ------------------------
    eta_W_AS = infoAS.eta_W_org;
    eta_b_AS = infoAS.eta_b_org*(1+sqrt(optionsAS.n))/(1+sqrt(n));
    eta_U_AS = infoAS.eta_U_org;
    eta_Lam_AS = infoAS.eta_Lam_org;
    
    normlam = norm(lam_sub); normv = norm(v);
    eta_v_AS = norm(lam_sub + projBox(v_sub - lam_sub,optionsAS.C))/(1+normlam+normv);
    eta_lam_AS = infoAS.eta_lam_org*(1+sqrt(optionsAS.n))/(1+sqrt(n)); %optionsAS.n
    
    
    relkkt_AS = max([eta_W_AS,eta_b_AS,eta_v_AS,eta_U_AS,eta_lam_AS,eta_Lam_AS]);
    
    % ---------------- Compute priminfeas and dualinfeas  ----------------
    primfeas_AS = max(eta_lam_AS,eta_Lam_AS);
    dualfeas_AS = eta_b_AS;
    % --------------- Compute primobj, dualobj and relgap  ---------------
    primobj_AS = obj_AS(1) + optionsAS.C*sum(max(xi_comI_AS,0));
    dualobj_AS = obj_AS(2);
    relgap_AS = abs(primobj_AS - dualobj_AS)/(1+primobj_AS+abs(dualobj_AS));
    
    
    runhist.eta_W_AS(cc) = eta_W_AS;
    runhist.eta_b_AS(cc) = eta_b_AS;
    runhist.eta_v_AS(cc) = eta_v_AS;
    runhist.eta_U_AS(cc) = eta_U_AS;
    runhist.eta_lam_AS(cc) = eta_lam_AS;
    runhist.eta_Lam_AS(cc) = eta_Lam_AS;
    
    ttime = etime(clock,tstart);
    Time(cc) = ttime;
    runhist.time(cc) = Time(cc)-Time(cc-1);
    
    %% Print results for the AS
    if (printyes)
        if ~isempty(msg); fprintf('\n %s',msg); end
        fprintf('\n--------------------------------------------------------------');
        fprintf('------------------');
        fprintf('\n  time = %3.2f',runhist.time(cc));
        fprintf('\n  number iter of AS = %2.0d',iterAS);
        fprintf('\n  number iter of ALM = %2.0d',runhist.iter_ALM(cc));
        fprintf('\n  number iter of SNCG = %2.0d',runhist.iter_SSN(cc));
        fprintf('\n  primobj = %9.8e, dualobj = %9.8e, relgap = %3.2e',primobj_AS,dualobj_AS,relgap_AS);
        fprintf('\n  priminfeas    = %3.2e, dualinfeas    = %3.2e',...
            primfeas_AS, dualfeas_AS);
        fprintf('\n  relative KKT residual = %3.2e',relkkt_AS);
        fprintf('\n  number of support matrices =%2.0d', nSMM_AS);
        fprintf('\n--------------------------------------------------------------');
        fprintf('------------------\n');
    end
    
    Wsol(:,:,cc) = W_AS; bsol(cc) = b_AS;
    Index0 = (xi_AS > -hat_epsion);
    n_I = sum(Index0);
    
    
    W = W_AS; b = b_AS; v = v_sub; U = U_AS;
    lam = lam_sub; Lam = Lam_AS;
    
    objopt(cc,:) = [primobj_AS,dualobj_AS];
    runhist.ttime_path(cc) = ttime;
    runhist.nSMM_path(cc) = nSMM_AS;
    
    runhist.relkkt(cc) = relkkt_AS;
    runhist.relgap(cc) = relgap_AS;
    runhist.iterAS(cc) = iterAS;
    runhist.nI_mean(cc) = mean(runhistAS.nI_path);
    
    info.r_indexJ = infoAS.r_indexJ;
    info.num_AindexI = num_AindexI;
end


