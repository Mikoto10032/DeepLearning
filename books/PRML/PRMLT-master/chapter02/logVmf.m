function y = logVmf(X, mu, kappa)
% Compute log pdf of a von Mises-Fisher distribution.
% Input:
%   X: d x n data matrix
%   mu: d x k mean
%   kappa: 1 x k variance
% Output:
%   y: k x n probability density in logrithm scale y=log p(x)
% Written by Mo Chen (sth4nth@gmail.com).
d = size(X,1);
c = (d/2-1)*log(kappa)-(d/2)*log(2*pi)-logbesseli(d/2-1,kappa);
q = bsxfun(@times,mu,kappa)'*X;
y = bsxfun(@plus,q,c');
