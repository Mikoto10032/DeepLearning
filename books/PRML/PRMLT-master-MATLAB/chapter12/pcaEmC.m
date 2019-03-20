function [W, Z, mu, mse] = pcaEmC(X, m)
% Perform Constrained EM like algorithm for PCA.
% Input:
%   X: d x n data matrix
%   m: dimension of target space
% Output:
%   W: d x m weight matrix
%   Z: m x n projected data matrix
%   mu: d x 1 mean vector
%   mse: mean square error
% Reference: 
%   A Constrained EM Algorithm for Principal Component Analysis by Jong-Hoon Ahn & Jong-Hoon Oh
% Written by Mo Chen (sth4nth@gmail.com).

d = size(X,1);
mu = mean(X,2);
X = bsxfun(@minus,X,mu);
W = rand(d,m); 

tol = 1e-6;
mse = inf;
maxIter = 200;
for iter = 1:maxIter
    Z = tril(W'*W)\(W'*X);
    W = (X*Z')/triu(Z*Z');

    last = mse;
    E = X-W*Z;
    mse = mean(dot(E(:),E(:)));
    if abs(last-mse)<mse*tol; break; end;
end
fprintf('Converged in %d steps.\n',iter);