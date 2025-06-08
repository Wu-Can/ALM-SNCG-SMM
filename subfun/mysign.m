%%=========================================================================
%% mysign:signum function 
%%
%% y = mysign(x)
%%
%% Input:
%% x = vector in R^n
%% Output
%% y = 1,  if x >= 0
%%   = -1, if x < 0
%% Copyright (c) 2024 by Can Wu, Donghui Li, Defeng Sun
%% See the paper for more details: 
%% Support matrix machine: exploring sample sparsity, low rank,
%% and adaptive sieving in high-performance computing
%%=========================================================================
function y = mysign(x)
index_1 = (x >= -eps);
index_neg1 = (x < -eps);
y = x;
y(index_1) = 1;
y(index_neg1) = -1;
end




