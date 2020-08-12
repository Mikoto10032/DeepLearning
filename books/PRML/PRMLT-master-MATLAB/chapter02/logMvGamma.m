function y = logMvGamma(x, d)
% Compute logarithm multivariate Gamma function 
% which is used in the probability density function of the Wishart and inverse Wishart distributions.
% Gamma_d(x) = pi^(d(d-1)/4) \prod_(j=1)^d Gamma(x+(1-j)/2)
% log(Gamma_d(x)) = d(d-1)/4 log(pi) + \sum_(j=1)^d log(Gamma(x+(1-j)/2))
% Input:
%   x: m x n data matrix
%   d: dimension
% Output:
%   y: m x n logarithm multivariate Gamma
% Written by Michael Chen (sth4nth@gmail.com).
y = d*(d-1)/4*log(pi)+sum(gammaln(x(:)+(1-(1:d))/2),2);
y = reshape(y,size(x));