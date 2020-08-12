function [X, model] = ppcaRnd(m, d, n)
% Generate data from probabilistic PCA model
% Input:
%   m: dimension of latent space
%   d: dimension of data
%   n: number of data
% Output:
%   X: d x n data matrix
%   model: model structure
% Written by Mo Chen (sth4nth@gmail.com).
beta = randg;
Z = randn(m,n);
W = randn(d,m); 
mu = randn(d,1);
X = bsxfun(@times,W*Z,mu)+randn(d,n)/sqrt(beta);

model.W = W;
model.mu = mu;
model.beta = beta;