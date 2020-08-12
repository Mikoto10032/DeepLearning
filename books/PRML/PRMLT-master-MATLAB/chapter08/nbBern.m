function model = nbBern(X, t)
% Naive bayes classifier with indepenet Bernoulli.
% Input:
%   X: d x n data matrix
%   t: 1 x n label (1~k)
% Output:
%   model: trained model structure
% Written by Mo Chen (sth4nth@gmail.com).
n = size(X,2);
E = sparse(1:n,t,1);
nk = sum(E,1);
w = full(nk)/n;
mu = X*(E./nk);  

model.mu = mu;      % d x k means 
model.w = w;