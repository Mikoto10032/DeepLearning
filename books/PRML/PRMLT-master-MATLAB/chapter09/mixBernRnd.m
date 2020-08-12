function [X, z, mu] = mixBernRnd(d, k, n)
% Generate samples from a Bernoulli mixture distribution.
% Input:
%   d: dimension of data
%   k: number of components
%   n: number of data
% Output:
%   X: d x n data matrix
%   z: 1 x n response variable
%   mu: d x k parameters of each Bernoulli component
% Written by Mo Chen (sth4nth@gmail.com).

% w = dirichletRnd(1,ones(1,k)/k);
w = ones(1,k)/k;
z = discreteRnd(w,n);
mu = rand(d,k);
X = zeros(d,n);
for i = 1:k
    idx = z==i;
    X(:,idx) = bsxfun(@le,rand(d,sum(idx)), mu(:,i));
end
