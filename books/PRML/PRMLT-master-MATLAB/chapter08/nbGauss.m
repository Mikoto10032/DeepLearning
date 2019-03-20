function model = nbGauss(X, t)
% Naive bayes classifier with indepenet Gaussian, each dimension of data is
% assumed from a 1d Gaussian distribution with independent mean and variance.
% Input:
%   X: d x n data matrix
%   t: 1 x n label (1~k)
% Output:
%   model: trained model structure
% Written by Mo Chen (sth4nth@gmail.com).
n = size(X,2);
k = max(t);
E = sparse(t,1:n,1,k,n,n);
nk = full(sum(E,2));
w = nk/n;
R = E'*spdiags(1./nk,0,k,k);
mu = X*R;  
var = X.^2*R-mu.^2;

model.mu = mu;      % d x k means 
model.var = var;  % d x k variances
model.w = w;