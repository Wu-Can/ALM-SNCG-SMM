%%=========================================================================
%% ATzfun1:
%% compute matrix-vector product and reshape into a p-by-q matrix: 
%%                  reshape(A*(y.*z),p,q)
%% 
%% ATz = ATzfun1(A,y,z,p,q,indexJ,r_indexJ)
%%
%% Input:
%% A = a (p*q)-by-r_indexJ matrix
%% y = a vector in R^n
%% p,q = dimensions of the feature matrix
%% indexJ = index set J_1
%% r_indexJ = cardinality of indexJ
%% Output:
%% ATz = reshape(A*(y.*z),p,q)
%% Copyright (c) 2024 by Can Wu, Donghui Li, Defeng Sun
%% For more details, please see the paper: 
%% Support matrix machine: exploring sample sparsity, low rank,
%% and adaptive sieving in high-performance computing
%%=========================================================================
function ATz = ATzfun1(A,y,z,p,q,indexJ,r_indexJ)
if (norm(z) == 0)
    ATz = sparse(p,q);
    return;
end

if nargin == 5
    ATz = reshape(A*(y.*z),p,q);  %ATz = reshape(mexAx(A,y.*z,0),p,q);
elseif nargin == 7
    if r_indexJ == 0
        ATz = sparse(p,q);%zeros(p,q);
    else
        n = length(y); n1 = length(z);
        if n1 == n
            normzy = norm(z - y);
            if normzy == 0
                ATz =  reshape(A(:,indexJ)*ones(r_indexJ,1),p,q); %reshape((ones(1,r_indexJ)*A(indexJ,:))',p,q);
            else
                ATz = reshape(A(:,indexJ)*(y(indexJ).*z(indexJ)),p,q); %reshape(((y(indexJ).*z(indexJ))'*A(indexJ,:))',p,q); 
            end
        else
            ATz = reshape(A(:,indexJ)*(y(indexJ).*z),p,q); %reshape(((y(indexJ).*z)'*A(indexJ,:))',p,q); 
        end
    end
end

end
