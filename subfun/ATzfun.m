%%=========================================================================
%% ATzfun:
%% compute matrix-vector product and reshape into a p-by-q matrix: 
%%                  reshape(A*(y.*z),p,q)
%% 
%% ATz = ATzfun(A,y,z,p,q,r_indexJ)
%%
%% Input:
%% A = a (p*q)-by-n matrix
%% y = a vector in R^n
%% p,q = dimensions of the feature matrix
%% r_indexJ = a positive integer between 0 and n
%% Output:
%% ATz = reshape(A*(y.*z),p,q)
%% Copyright (c) 2024 by Can Wu, Donghui Li, Defeng Sun
%% For more details, please see the paper: 
%% Support matrix machine: exploring sample sparsity, low rank,
%% and adaptive sieving in high-performance computing
%%=========================================================================
function ATz = ATzfun(A,y,z,p,q,r_indexJ)
if (norm(z) == 0)
    ATz = sparse(p,q);
    return;
end

if nargin == 5
    ATz = reshape(A*(y.*z),p,q);
elseif nargin == 6
    if r_indexJ == 0
        ATz = sparse(p,q);
    else
        ATz = reshape(A*(y.*z),p,q); 
    end
end

end


