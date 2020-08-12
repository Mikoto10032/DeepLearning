function [label, Theta, w, llh] = mixGaussGb(X, opt)
% Collapsed Gibbs sampling for Dirichlet process (infinite) Gaussian mixture model (a.k.a. DPGM). 
% This is a wrapper function which calls underlying Dirichlet process mixture model.
% Input: 
%   X: d x n data matrix
%   opt(optional): prior parameters
% Output:
%   label: 1 x n cluster label
%   Theta: 1 x k structure of trained Gaussian components
%   w: 1 x k component weight vector
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
[d,n] = size(X);
mu = mean(X,2);
Xo = bsxfun(@minus,X,mu);
s = sum(Xo(:).^2)/(d*n);
if nargin == 1
    kappa0 = 1;
    m0 = mean(X,2);
    nu0 = d;
    S0 = s*eye(d);
    alpha0 = 1;
else
    kappa0 = opt.kappa;
    m0 = opt.m;
    nu0 = opt.nu;
    S0 = opt.S;
    alpha0 = opt.alpha;
end
prior = GaussWishart(kappa0,m0,nu0,S0);
[label, Theta, w, llh] = mixDpGb(X,alpha0,prior);