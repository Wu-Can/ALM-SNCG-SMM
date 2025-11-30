function [D, nuc, rk, time_svd] = shrinkage_svd(X, tau)
tic;
[U, S, V] = svd(X);
time_svd = toc;
s = max(0, S-tau);
nuc = sum(diag(s));
D = U *  s * V';
rk = sum(diag(s>0));
end