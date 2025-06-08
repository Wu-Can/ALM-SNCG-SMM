%%=========================================================================
%% AYfun:
%% compute matrix*vector: y.*(Y(:)'*A)'
%% 
%% AY = AYfun(A,y,Y,r_indexJ)
%%
%% Input:
%% A = a (p*q)-by-n matrix
%% y = a vector in R^n
%% Y = a p-by-q matrix
%% r_indexJ = a positive number between 0 and n
%% Output:
%% AY = y.*(Y(:)'*A)'
%% Copyright (c) 2024 by Can Wu, Donghui Li, Defeng Sun
%% For more details, please see the paper: 
%% Support matrix machine: exploring sample sparsity, low rank,
%% and adaptive sieving in high-performance computing
%%=========================================================================
function AY = AYfun(A,y,Y,r_indexJ)
if  (nargin == 3) || ((nargin == 4) && (r_indexJ > 0))
    AY = y.*(Y(:)'*A)';
else
    n = length(y);
    AY = zeros(n,1);
end



