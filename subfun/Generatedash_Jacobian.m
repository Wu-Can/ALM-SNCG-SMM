%%==========================================================================================
%% Generatedash_Jacobian:
%% Compute key components of matrix M and linear operator G.
%%
%% M in R^{n*n} is an element of the Clarke generalized Jacobian of the projection 
%% onto box S at some omega in R^n;
%% G:R^{p*q} -> R^{p*q} is an element of the Clarke generalized Jacobian of the
%% projection onto the spectral norm ball at some matrix X in R^{p*q}.
%% 
%% options = Generatedash_Jacobian(y,omega,options,parPX)
%%
%% Input:
%% y = vector in R^n
%% omega = vector in R^n
%% options.p,options.q = dimensions of the feature matrix
%% options.Cd = vector in R^n, i.e., C*d_scale
%% options.tau =  parameter tau in the SMM model
%% options.flag_tau = 0, if tau = 0;
%%                    1, otherwise
%% parPX.nu = vector of sigular velues of X_k(W)
%% parPX.U = left singular vectors of X_k(W)
%% parPX.V = right singular vectors of X_k(W)
%%
%% Output:
%% options.indexJ = index set J_1
%% options.r_indexJ = cardinality of J_1
%% options.yJTyJ = yJ^T*yJ;
%% options.k1 = cardinality of index set alpha
%% options.k2 = sum of the cardinalities of index sets alpha and beta_1
%% options.d_vec = vector in R^{k1}
%% options.flag_k1k2 = 1, if k2 > k1
%%                   = 0, otherwise
%% options.Method_flag = 1, utilize the low-rank property in G(dW)
%%                     = 2, otherwise
%% Copyright (c) 2024 by Can Wu, Donghui Li, Defeng Sun
%% See Subsection 3.1 of the paper for more details: 
%% Support matrix machine: exploring sample sparsity, low rank,
%% and adaptive sieving in high-performance computing
%%=========================================================================================
function options = Generatedash_Jacobian(y,omega,options,parPX)
p = options.p;
q = options.q;
Cd = options.Cd;
tau = options.tau;
flag_tau = options.flag_tau;
tol = 1e-12;

%% Compute key components of H(dW)
indexJ = (omega > tol) & (omega < Cd - tol);
r_indexJ = sum(indexJ);
yJ = y(indexJ);
yJTyJ = yJ'*yJ;

options.indexJ = indexJ;
options.r_indexJ = r_indexJ;
options.yJTyJ = yJTyJ;
options.k1 = 0;

%% Compute key components of G(dW)
if flag_tau
    nu = parPX.nu;
    U = parPX.U;
    V = parPX.V;
    len_nu = length(nu);
    index_alp = (nu > tau + tol);
    index_bet1 = (abs(nu - tau) <= tol);
    index_bet2 = (nu < tau - tol);

    k1 = sum(index_alp);
    k2 = k1 + sum(index_bet1);
    
    omega_num = max(q/p, p/q);
    if omega_num == 1
        bound_G = min(p,q)/2;
    else
        bound_G = (5*omega_num+1)*min(p,q)/(7*omega_num+3);
    end
    
    if k1 <= bound_G
        Method_flag = 1; % utilize the low-rank property
    else
        Method_flag = 2; % do not utilize the low-rank property
    end
    
    XI1_alpbet2 = zeros(k1,len_nu-k2);
    XI2_alpbet2 = zeros(k1,len_nu-k2);
    XI2_alpalp = zeros(k1,k1);
    
    U_alp = U(:,index_alp);  U_bet2 = U(:,index_bet2);
    V_alp = V(:,index_alp);  V_bet2 = V(:,index_bet2);
    
    if k2 > k1
        XI2_alpbet1 = zeros(k1,k2-k1);
        flag_k1k2 = 1;
        options.V_bet1 = V(:,index_bet1); options.U_bet1 = U(:,index_bet1);
    else
        XI2_alpbet1 = []; flag_k1k2 = 0;
        options.V_bet1 = []; options.U_bet1 = [];
    end
    
    if Method_flag == 1
        for i = 1:k1
            for j = 1:k1
                XI2_alpalp(i,j) = 1 - 2*tau/(nu(i) + nu(j));
            end
            if flag_k1k2 == 1
                for jj = 1:k2-k1
                    t = k1 + jj;
                    XI2_alpbet1(i,jj) = (nu(i) - tau)/(nu(i) + nu(t));
                end
            end
            for jjj = 1:(len_nu-k2)
                tt = k2 + jjj;
                XI1_alpbet2(i,jjj) = (nu(i) - tau)/(nu(i) - nu(tt));
                XI2_alpbet2(i,jjj) = (nu(i) - tau)/(nu(i) + nu(tt));
            end
        end
    else
        for i = 1:k1
            for j = 1:k1
                XI2_alpalp(i,j) = 2*tau/(nu(i) + nu(j));
            end
            if flag_k1k2 == 1
                for jj = 1:k2-k1
                    t = k1 + jj;
                    XI2_alpbet1(i,jj) = (tau + nu(t))/(nu(i) + nu(t));
                end
            end
            for jjj = 1:(len_nu-k2)
                tt = k2 + jjj;
                XI1_alpbet2(i,jjj) = (tau - nu(tt))/(nu(i) - nu(tt));
                XI2_alpbet2(i,jjj) = (tau + nu(tt))/(nu(i) + nu(tt));
            end
        end
        
    end

    if (k1 > 0) && (p ~= q)
        if Method_flag == 1
            d_vec = 1 - tau./nu(1:k1);
        else
            d_vec = tau./nu(1:k1);
        end
        U_alpDiagd = U_alp.*d_vec';
        DiagdV_alpT = d_vec.*V_alp';
    else
        U_alpDiagd = []; d_vec = []; DiagdV_alpT = [];
    end
    
    options.XI1_alpbet2 = XI1_alpbet2;
    options.XI2_alpalp = XI2_alpalp;
    options.XI2_alpbet1 = XI2_alpbet1;
    options.XI2_alpbet2 = XI2_alpbet2;
    options.d_vec = d_vec;
    options.index_alp = index_alp;
    options.index_bet1 = index_bet1;
    options.index_bet2 = index_bet2;
    options.k1 = k1;
    options.k2 = k2;
    options.flag_k1k2 = flag_k1k2;
    options.U_alp = U_alp;
    options.U_bet2 = U_bet2;
    options.V_alp = V_alp;
    options.V_bet2 = V_bet2;
    options.U_alpDiagd = U_alpDiagd;
    options.Method_flag = Method_flag;
    options.DiagdV_alpT = DiagdV_alpT;
    options.bound_G = bound_G;
end





