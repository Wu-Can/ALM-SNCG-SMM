%%=================================================================================================
%% myfindstep:
%% find the step length alpha of SNCG using the strong Wolfe line search 
%%
%% [W_snew,b_snew,v_snew,U_snew,lam_snew,Lam_snew,omega_new,X_new,Pomega_new,
%%  PX_new,alpha,iter,g0,Ly,maxiterfs,AdW,normvsqr,normUsqr,FnormW_Bsqr_new,parPX] =
%%    myfindstep(A,y0,y,W_sub,b_sub,v_sub,U_sub,lam_sub,
%% Lam_sub,omega_sub,X_sub,dW,db,gradphiW,gradphib,Ly0,FnormW_Bsqr,tol,AJdW,parsub)
%%
%% Input:
%% A = matrix in R^{(p*q)*n}
%% y0 = class label vector in R^n
%% y = vector in R^n
%% (W_sub,b_sub,v_sub,U_sub,lam_sub,Lam_sub) = current iteration point in SNCG
%% omega_sub = omega_k(W_sub,b_sub) defined in (2.9) of the paper
%% X_sub = X_k(W_sub) defined in (2.9) of the paper
%% (dW,db) = Newton direction
%% (gradphiW,gradphib) = right-hand term in (2.21) of the paper
%% Ly0 = negative objective value of subproblem (2.8) at (W_sub,b_sub) in the paper
%% FnormW_Bsqr = norm(W_sub,'fro')^2
%% tol = a small positive number
%% AJdW = y0J.*(dW(:)'*AJ)'
%% parsub.sigma = parameter sigma in ALM 
%% parsub.tau = parameter tau in the SMM model
%% parsub.delt = parameter tau in the general SMM model
%% parsub.Cd = C*d_scale
%% parsub.n = sample size
%% parsub.B = matrix in R^{p*q} in the general SMM model
%% parsub.a = constant in the general SMM model
%% parsub.flag_B = 0, if B = 0
%%               = 1, otherwise
%% parsub.flag_tau = 0, if tau = 0
%%                 = 1, otherwise
%% parsub.r_indexJ = cardinality of index set J
%% parsub.flag_svds = 1, use SVDS for X_new
%%                  = 0, otherwise
%% Output:
%% (W_snew,b_snew,v_snew,U_snew,lam_snew,Lam_snew) = the latest iteration point
%% omega_new = omega_k(W_snew,b_snew) defined in (2.9) of the paper
%% X_new = X_k(W_snew) defined in (2.9) of the paper
%% Pomega_new = projection of omega_new onto the box [0,U]
%% PX_new = projection of X_new onto the spectral ball with radius tau
%% alpha = step length
%% iter = number of strong Wolfe line search iterations 
%% g0 = <gradphiW,dW> + gradphib*db
%% Ly = negative objective value of subproblem (2.8) at (W_snew,b_snew)
%% maxiterfs = 1, if iter = maximum iterations
%%           = 0, otherwise
%%
%% AdW = y0.*(dW(:)'*A)'
%% normvsqr = norm(v_snew)^2
%% normUsqr = norm(U_snew,'fro')^2
%% FnormW_Bsqr_new = norm(W_snew-B,'fro')^2
%% parPX.num_svd = total number of SVD calls
%% parPX.num_svds = total number of SVDS calls
%% Copyright (c) 2024 by Can Wu, Donghui Li, Defeng Sun
%% For more details, please see Algorithm 2.2 of the paper: 
%% Support matrix machine: exploring sample sparsity, low rank,
%% and adaptive sieving in high-performance computing
%%=================================================================================================
function [W_snew,b_snew,v_snew,U_snew,lam_snew,Lam_snew,omega_new,X_new,Pomega_new,PX_new,alpha,iter,g0,Ly,maxiterfs,AdW,normvsqr,normUsqr,FnormW_Bsqr_new,parPX] = ...
    myfindstep(A,y0,y,W_sub,b_sub,v_sub,U_sub,lam_sub,Lam_sub,omega_sub,X_sub,dW,db,gradphiW,gradphib,Ly0,FnormW_Bsqr,tol,AJdW,parsub)
%%
printlevel = 0;
maxit = ceil(log(1/(tol+eps))/log(2));
c1 = 1e-4; c2 = 0.9;
sigma = parsub.sigma; tau = parsub.tau;
Cd = parsub.Cd; n = parsub.n;
B = parsub.B; flag_B = parsub.flag_B;
flag_tau = parsub.flag_tau;
num_svd = 0;
num_svds = 0;
%%
g0 = sum(sum(gradphiW.*dW)) + gradphib*db;
if (parsub.r_indexJ == n)
    AdW = AJdW;
else
    AdW = AYfun(A,y0,dW);
end
sigAdW_dby = sigma*(AdW + db*y);
sigdW = sigma*dW;
if flag_B
    innerW_dW = sum(sum((W_sub-B).*dW));
else
    innerW_dW = sum(sum(W_sub.*dW));
end
normdWsqr = norm(dW,'fro')^2;
Ly = [];
maxiterfs = 0;
k1_vec = zeros(maxit,1);

if (g0 <= 0)
    alpha = 0; iter = 0;
    if (printlevel)
        fprintf('\n Need an ascent direction, %2.1e  ',g0);
    end
    W_snew = W_sub; b_snew = b_sub;
    v_snew = v_sub; U_snew = U_sub;
    lam_snew = lam_sub; Lam_snew = Lam_sub;
    omega_new = omega_sub; X_new = X_sub;
    Ly = Ly0;
    if flag_tau
        normUsqr = norm(U_sub,'fro')^2;
        [PX_new,parPX] = projspball(X_sub,tau); 
    else
        normUsqr = []; PX_new = []; parPX = [];
    end
    FnormW_Bsqr_new = FnormW_Bsqr;%norm(W_snew,'fro')^2;
    Pomega_new = projBox(omega_sub,Cd);

    num_svd = num_svd + 1;
    normvsqr = norm(v_snew)^2;
    parPX.num_svd = num_svd;
    parPX.num_svds = num_svds;
    return;
end
%%
alpha = 1; alpconst = 0.5;
for iter = 1:maxit
    if (iter == 1)
        alpha = 1; LB = 0; UB = 1;
    else
        alpha = alpconst*(LB+UB);
    end
    
    % update phi_new and gradphi_new
    omega_new = omega_sub - alpha*sigAdW_dby;
    Pomega_new =  projBox(omega_new,Cd);
    v_snew = (omega_new - Pomega_new)/sigma;
    
    if flag_tau
        X_new = X_sub + alpha*sigdW;
        % using SVD or SVDS
        if iter > 1
            k1_vec(iter-1) = parPX.k1;
        end
        if ((parsub.flag_svds == 1) && (iter > 1)) || ((iter >= 5) && all(k1_vec(iter-4:iter-2)) == k1_vec(iter-1))
            [PX_new,parPX] = projspball(X_new,tau,k1_vec(iter-1));
            num_svds = num_svds + 1;
        else
            [PX_new,parPX] = projspball(X_new,tau);
            num_svd = num_svd + 1;
        end
        U_snew = (X_new - PX_new)/sigma;
        if flag_B
            galp = -innerW_dW - alpha*normdWsqr + Pomega_new'*AdW - sum(sum(PX_new.*dW))+...
                db*(y'*Pomega_new) - parsub.delt*(b_sub + alpha*db - parsub.a)*db;
        else
            galp = -innerW_dW - alpha*normdWsqr + Pomega_new'*AdW - sum(sum(PX_new.*dW))+...
                db*(y'*Pomega_new);
        end
    else
        X_new = []; PX_new = []; U_snew = []; parPX = [];
        if flag_B
            galp = -innerW_dW - alpha*normdWsqr + Pomega_new'*AdW + db*(y'*Pomega_new) - parsub.delt*(b_sub + alpha*db - parsub.a)*db;
        else
            galp = -innerW_dW - alpha*normdWsqr + Pomega_new'*AdW - sum(sum(PX_new.*dW))+...
                db*(y'*Pomega_new);
        end
    end
    
    if (iter == 1)
        gLB = g0; gUB = galp;
        if (sign(gLB)*sign(gUB) > 0)
            if (printlevel); fprintf('|'); end
            normvsqr = norm(v_snew)^2; 
            FnormW_Bsqr_new = FnormW_Bsqr + 2*alpha*innerW_dW + alpha^2*normdWsqr;
            if flag_B
                Ly_apart = (parsub.delt/2)*(b_sub + alpha*db - parsub.a)^2;
            else
                Ly_apart = 0;
            end
            if flag_tau
                normUsqr = norm(U_snew,'fro')^2;
                Ly = -0.5*FnormW_Bsqr_new - Ly_apart -(0.5/sigma)*(norm(omega_new)^2 + norm(X_new,'fro')^2) + (sigma/2)*(normvsqr + normUsqr);
                Lam_snew = PX_new;
                parPX.num_svd = num_svd; parPX.num_svds = num_svds;
            else
                Ly = -0.5*FnormW_Bsqr_new - Ly_apart -(0.5/sigma)*norm(omega_new)^2 + (sigma/2)*normvsqr;
                normUsqr = []; Lam_snew = []; parPX = [];
            end
            W_snew = W_sub + alpha*dW; b_snew = b_sub + alpha*db;
            lam_snew = -Pomega_new;
            return;
        end
    end
    
    if (abs(galp) < c2*abs(g0))
        normvsqr = norm(v_snew)^2; 
        FnormW_Bsqr_new = FnormW_Bsqr + 2*alpha*innerW_dW + alpha^2*normdWsqr;
        if flag_B
            Ly_apart = (parsub.delt/2)*(b_sub + alpha*db - parsub.a)^2;
        else
            Ly_apart = 0;
        end
        if flag_tau
            normUsqr = norm(U_snew,'fro')^2;
            Ly = -0.5*FnormW_Bsqr_new - Ly_apart -(0.5/sigma)*(norm(omega_new)^2 + norm(X_new,'fro')^2) + ...
                (sigma/2)*(normvsqr + normUsqr);
        else
            Ly = -0.5*FnormW_Bsqr_new - Ly_apart -(0.5/sigma)*norm(omega_new)^2 + (sigma/2)*normvsqr;
            normUsqr = [];
        end
        if (Ly-Ly0-c1*alpha*g0 > -1e-8/max(1,abs(Ly0)))
            if (printlevel); fprintf(':'); end
            W_snew = W_sub + alpha*dW; b_snew = b_sub + alpha*db;
            lam_snew = -Pomega_new;
            if flag_tau
                Lam_snew = PX_new;
                parPX.num_svd = num_svd; parPX.num_svds = num_svds;
            else
                Lam_snew = []; parPX = [];
            end
            return;
        end
    end
    
    if (sign(galp)*sign(gUB) < 0)
        LB = alpha; gLB = galp;
    elseif (sign(galp)*sign(gLB) < 0)
        UB = alpha; gUB = galp;
    end
    
end

if (iter == maxit)
    maxiterfs = 1;
    W_snew = W_sub + alpha*dW; b_snew = b_sub + alpha*db;
    lam_snew = -Pomega_new; Lam_snew = PX_new;
end

if (printlevel); fprintf('m'); end
if isempty(Ly)
    normvsqr = norm(v_snew)^2;
    FnormW_Bsqr_new = FnormW_Bsqr + 2*alpha*innerW_dW + alpha^2*normdWsqr;
    if flag_B
        Ly_apart = (parsub.delt/2)*(b_sub + alpha*db - parsub.a)^2;
    else
        Ly_apart = 0;
    end
    if flag_tau
        normUsqr = norm(U_snew,'fro')^2;
        Ly = -0.5*FnormW_Bsqr_new - Ly_apart -(0.5/sigma)*(norm(omega_new)^2 + norm(X_new,'fro')^2) + ...
            (sigma/2)*(normvsqr + normUsqr);
    else
        Ly = -0.5*FnormW_Bsqr_new - Ly_apart -(0.5/sigma)*norm(omega_new)^2 + (sigma/2)*normvsqr;
        normUsqr = [];
    end
end
if flag_tau
    parPX.num_svd = num_svd;
    parPX.num_svds = num_svds;
end








