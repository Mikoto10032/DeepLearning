function [X, z, mu] = kmeansRnd(d, k, n)
% Generate samples from a Gaussian mixture distribution with common variances (kmeans model).
% Input:
%   d: dimension of data
%   k: number of components
%   n: number of data
% Output:
%   X: d x n data matrix
%   z: 1 x n response variable
%   mu: d x k centers of clusters
% Written by Mo Chen (sth4nth@gmail.com).
alpha = 1;
beta = nthroot(k,d); % in volume x^d there is k points: x^d=k

X = randn(d,n);
w = dirichletRnd(alpha,ones(1,k)/k);
z = discreteRnd(w,n);
E = full(sparse(z,1:n,1,k,n,n));
mu = randn(d,k)*beta;
X = X+mu*E;