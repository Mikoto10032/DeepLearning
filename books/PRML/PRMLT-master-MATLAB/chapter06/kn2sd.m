function D = kn2sd(K)
% Transform a kernel matrix (or inner product matrix) to a squared distance matrix
% Input:
%   K: n x n kernel matrix
% Ouput:
%   D: n x n squared distance matrix
% Written by Mo Chen (sth4nth@gmail.com).
d = diag(K);
D = -2*K+bsxfun(@plus,d,d');
