function y = logWishart(Sigma, W, v)
% Compute log pdf of a Wishart distribution.
% Input:
%   Sigma: d x d covariance matrix
%   W: d x d covariance parameter
%   v: degree of freedom
% Output:
%   y: probability density in logrithm scale y=log p(Sigma)
% Written by Mo Chen (sth4nth@gmail.com).
d = length(Sigma);
B = -0.5*v*logdet(W)-0.5*v*d*log(2)-logmvgamma(0.5*v,d);
y = B+0.5*(v-d-1)*logdet(Sigma)-0.5*trace(W\Sigma);