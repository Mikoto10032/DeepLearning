function [W, Z, mu, mse] = pcaEm(X, m)
% Perform EM-like algorithm for PCA (by Sam Roweis).
% Input:
%   X: d x n data matrix
%   m: dimension of target space
% Output:
%   W: d x m weight matrix
%   Z: m x n projected data matrix
%   mu: d x 1 mean vector
%   mse: mean square error
% Reference: 
%   Pattern Recognition and Machine Learning by Christopher M. Bishop 
%   EM algorithms for PCA and SPCA by Sam Roweis 
% Written by Mo Chen (sth4nth@gmail.com).
d = size(X,1);
mu = mean(X,2);
X = bsxfun(@minus,X,mu);
W = rand(d,m); 

tol = 1e-6;
mse = inf;
maxIter = 200;
for iter = 1:maxIter
    Z = (W'*W)\(W'*X);             % 12.58
    W = (X*Z')/(Z*Z');              % 12.59

    last = mse;
    E = X-W*Z;
    mse = mean(dot(E(:),E(:)));
    if abs(last-mse)<mse*tol; break; end;
end
fprintf('Converged in %d steps.\n',iter);

