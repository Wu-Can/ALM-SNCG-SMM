%%=========================================================================
%% projBox: compute projection onto the box [0,U]
%%
%% [Pomega,Pk] = projBox(omega,U)
%%
%% Input:
%% omega = vector in R^n
%% Output:
%% Pomega = min(max(Pomega,0),U)
%% Pk = set of indices i where Pomega(i)= omega(i)
%% Copyright (c) 2024 by Can Wu, Donghui Li, Defeng Sun
%% See the paper for more details: 
%% Support matrix machine: exploring sample sparsity, low rank,
%% and adaptive sieving in high-performance computing
%%=========================================================================
function [Pomega,Pk] = projBox(omega,U)
Pomega = max(omega,0);
if ~isempty(U)
    Pomega = min(Pomega,U);
end
if nargout == 2
    Pk = abs(Pomega - omega) < 1e-16;
end