%%=========================================================================
%% Generate_random_data:
%% randomly generate training and test data sets:
%%
%% [Xy_train,Xy_test] = Generate_random_data(n_train,p,q,flag_trans)
%%
%% Input:
%% n_train = sample size for training set
%% p,q = dimensions of the feature matrix
%% flag_trans = 0, if Xy_train.X is a n_train-by-(p*q) matrix
%%              1, if Xy_train.X is a (p*q)-by-n_train matrix
%% flag_test = 1,  if the test dadaset Xy_test is generated
%%             0,  otherwise
%% Output:
%% Xy_train.X = n_train-by-(p*q) matrix
%% Xy_train.y = n_train-dimensional label vector
%% Xy_train.n = n_train
%% Xy_train.p, Xy_train.q = dimensions of the feature matrix
%% Xy_test.X = (n_train/4)-by-(p*q) matrix
%% Xy_test.y = (n_train/4)-dimensional label vector
%% Xy_test.n = n_train/4
%% Xy_test.p, Xy_test.q = dimensions of the feature matrix
%% Copyright (c) 2024 by Can Wu, Donghui Li, Defeng Sun
%% For more details, please see Subsection 5.1 of the paper:
%% Support matrix machine: exploring sample sparsity, low rank,
%% and adaptive sieving in high-performance computing
%%=========================================================================
function [Xy_train,Xy_test] = Generate_random_data(n_train,p,q,flag_trans,flag_test,delta)

if nargin == 3
    flag_trans = 1; flag_test = 1;
end
V = 20;
if nargin < 6
    delta = 2e-4;
end
n_total = ceil(1.25*n_train);
rng(10);
d = p*q;
%% Generate p-by-q weighted matrix W with rank k
V = min([V,p,q]); % rank(W)
[W_ture,flag_rank] = generate_W(p,q,V);

if flag_rank
    fprintf('error: the rank of W is not k');
end

%% Generate n_total-by-d matrix X
V_e = orth(randn(n_total,V));
if ~(size(V_e,2) == V)
    fprintf('error: V_e is not the n*V orthogonal matix');
end
ii_vec = zeros(1,d);
vec_ellr_over_q = ceil(([1:q].*V)./q);
for iii = 1:d
    ii_vec(iii) = vec_ellr_over_q(ceil(iii/p));
end
Coef_V = sparse(ii_vec,[1:d],ones(1,d),V,d,d);
X = V_e*Coef_V;
Epsilon = delta*randn(n_total,d);

X = X + Epsilon;
clear Epsilon V_e Coef_V

%% Generate n_total-dimensional label vector y
y = mysign(X*W_ture(:));

%% Generate Xy_train and Xy_test
if flag_test == 1
    if flag_trans
        Xy_test.X = (X((n_train+1):end,:))';
        X((n_train+1):end,:) = [];
        Xy_train.X = X';
    else
        Xy_test.X = X((n_train+1):end,:);
        X((n_train+1):end,:) = [];
        Xy_train.X = X;
    end
    Xy_train.y = y(1:n_train);
    y(1:n_train) = [];
    Xy_test.y = y;

    Xy_train.n = n_train;
    Xy_train.p = p;
    Xy_train.q = q;
    Xy_test.n = length(y);
    Xy_test.p = p;
    Xy_test.q = q;
else
    Xy_test = [];
    if flag_trans
        X((n_train+1):end,:) = [];
        Xy_train.X = X';
    else
        X((n_train+1):end,:) = [];
        Xy_train.X = X;
    end
    Xy_train.y = y(1:n_train);
    Xy_train.n = n_train;
    Xy_train.p = p;
    Xy_train.q = q;
end

end


function [W,flag_rank] = generate_W(m,n,k)
rng(1)
P = orth(randn(m,k));
Q = orth(randn(n,k));
W = P*Q';
flag_rank = rank(W)-k;
end


