function [X, y, W ] = mixLinRnd(d, k, n)
% Generate data from mixture of linear model
% Input:
%   d: dimension of data
%   k: number of components
%   n: number of data
% Output:
%   X: d x n data matrix
%   y: 1 x n response variable
%   W: d+1 x k weight matrix 
% Written by Mo Chen (sth4nth@gmail.com).
W = randn(d+1,k);
[X, z] = kmeansRnd(d, k, n);
y = zeros(1,n);
for j = 1:k
    idx = (z == j);
    y(idx) = W(1:(end-1),j)'*X(:,idx)+W(end,j);
end


