function y = logGauss(X, mu, sigma)
% Compute log pdf of a Gaussian distribution.
% Input:
%   X: d x n data matrix
%   mu: d x 1 mean vector of Gaussian
%   sigma: d x d covariance matrix of Gaussian
% Output:
%   y: 1 x n probability density in logrithm scale y=log p(x)
% Written by Mo Chen (sth4nth@gmail.com).
d = size(X,1);
X = X-mu;
[U,p]= chol(sigma);
if p ~= 0
    error('ERROR: sigma is not PD.');
end
Q = U'\X;
q = dot(Q,Q,1);  % quadratic term (M distance)
c = d*log(2*pi)+2*sum(log(diag(U)));   % normalization constant
y = -(c+q)/2;
