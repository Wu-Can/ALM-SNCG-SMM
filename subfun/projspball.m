%%==============================================================================
%% projspball: compute projection onto the spectral ball of radius tau
%%
%% [PX, par] = projspball(X,tau,k1)
%%
%% Input:
%% X = matrix in R^{p*q}
%% tau = parameter tau in the SMM model
%% k1 = positive number between 1 and min(p,q)
%% Output:
%% PX = projection onto the spectral ball of radius tau at X
%% par.U = left singular vectors of X
%% par.V = right singular vectors of X
%% par.nu = vector of sigular values of X
%% par.k1 = number of sigular values of X greater than tau
%% par.Pk = 1, if k1 = 0
%%        = 0, if k1 > 0
%% Copyright (c) 2024 by Can Wu, Donghui Li, Defeng Sun
%% See the paper for more details:
%% Support matrix machine: exploring sample sparsity, low rank,and 
%% adaptive sieving in high-performance computing
%%==============================================================================
function [PX, par] = projspball(X,tau,k1)
tiny = 1e-14;
if issparse(X)
    X = full(X);
end
if (nargin == 2)
    [U,S,V] = svd(X,'econ');
    nu = diag(S); len_nu = length(nu);
    index_nu = (nu > tau + tiny);
    k1 = sum(index_nu);
    
    Pk = (k1 == 0);
    if Pk
        PX = X;
    else
        if k1 > 0.3*len_nu
            g_nu = min(nu,tau-tiny);
            PX = U*sparse(1:len_nu,1:len_nu,g_nu)*V';
        else
            nu_part = nu(1:k1) - tau;
            U_part = U(:,1:k1);
            V_part = V(:,1:k1);
            
            PX = X - U_part*sparse(1:k1,1:k1,nu_part)*V_part';
        end
    end
elseif (nargin == 3)
    [U,S,V] = svds(X, k1,'largest','MaxIterations', 500);
    nu = diag(S);
    index_nu = (nu > tau + tiny);
    lenu_part = sum(index_nu);
    
    Pk = (lenu_part == 0);
    if Pk
        PX = X;
    else
        nu_part = nu(1:lenu_part) - tau;
        U_part = U(:,1:lenu_part);
        V_part = V(:,1:lenu_part);
        
        PX = X - U_part*sparse(1:lenu_part,1:lenu_part,nu_part)*V_part';
    end
end

par.U = U;
par.V = V;
par.nu = nu;
par.Pk = Pk;
par.k1 = k1;






