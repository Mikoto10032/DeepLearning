function x = gaussRnd(mu, Sigma, n)
% Generate samples from a Gaussian distribution.
% Input:
%   mu: d x 1 mean vector
%   Sigma: d x d covariance matrix
%   n: number of samples
% Outpet:
%   x: d x n generated sample x~Gauss(mu,Sigma)
% Written by Mo Chen (sth4nth@gmail.com).
if nargin == 2
    n = 1;
end
[V,err] = chol(Sigma);
if err ~= 0
    error('ERROR: sigma must be a symmetric positive definite matrix.');
end
x = V'*randn(size(V,1),n)+repmat(mu,1,n);