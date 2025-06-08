%%===========================================================================================
%% matvec_VdW:
%% Compute the value of the linear operator V(dW):
%% VdW = dW + sigma*G(dW) + sigma*AJ^T(AJ(dW)) - tmp_num*AJ^T*yJ,
%% where tmp_num = sigma^2*yJ^T*AJ(dW)/(sigma*|J|+rho).
%% 
%% [VdW,options] = matvec_VdW(AJ,y0J,yJ,dW,AJTy,options,parPX)
%%
%% Input:
%% AJ = A(:,indexJ)
%% y0J = y0(indexJ)
%% yJ = y(indexJ)
%% dW = matrix in R^{p*q}
%% AJTy = reshape(AJ*(y0J.*yJ),p,q)
%% options.sigma = parameter sigma in ALM
%% options.r_indexJ = cardinality of index set J
%% options.p, options.q = dimensions of the feature matrix
%% options.sigJ_rho = sigma*r_indexJ + rho
%% options.flag_tau = 0, if tau = 0;
%%                    1, otherwise
%% options = a structure containing all subparts and index sets 
%%           needed by G(dW)
%% parPX.U = left singular vectors of X_k(W)
%% parPX.V = right singular vectors of X_k(W)
%%
%% Output:
%% VdW = (1+tidy)*dW + sigATJAJdW - tmp_num*AJTy
%% Copyright (c) 2024 by Can Wu, Donghui Li, Defeng Sun
%% See Subsection 3.1 of the paper for more details: 
%% Support matrix machine: exploring sample sparsity, low rank,
%% and adaptive sieving in high-performance computing
%%===========================================================================================
function [VdW,options] = matvec_VdW(AJ,y0J,yJ,dW,AJTy,options,parPX)
sigma = options.sigma;
r_indexJ = options.r_indexJ;
p = options.p; q = options.q;
sigJ_rho = options.sigJ_rho;
flag_tau = options.flag_tau;

if r_indexJ == 0
    yTAJdW = 0;
    tmp_num = 0;
    sigATJAJdW = sparse(p,q);
else
    AJdW = AYfun(AJ,y0J,dW,r_indexJ); % AY = y(indexJ).*(Y(:)'*A(:,indexJ))';  
    yTAJdW = yJ'*AJdW; %y(indexJ)'*AJdW;
    tmp_num = yTAJdW*sigma^2/sigJ_rho;
    sigATJAJdW = sigma*ATzfun(AJ,y0J,AJdW,p,q,r_indexJ); %ATz = reshape(A(:,indexJ)*(y(indexJ).*z),p,q);
end

tidy = 1e-12;
if flag_tau
    GdW = Generate_GdW(dW,options,parPX);
    VdW = (1+tidy)*dW + sigma*GdW + sigATJAJdW - tmp_num*AJTy;
    options.GdW = GdW;
else
    VdW = (1+tidy)*dW + sigATJAJdW - tmp_num*AJTy;
end
options.yTAJdW = yTAJdW;
options.sigATJAJdW = sigATJAJdW;










