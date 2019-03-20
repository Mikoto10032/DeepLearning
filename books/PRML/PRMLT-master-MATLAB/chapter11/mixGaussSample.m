function [X, z] = mixGaussSample(Theta, w, n )
% Genarate samples form a Gaussian mixture model with GaussianWishart prior.
% Input:
%   Theta: cell of GaussianWishart priors of components
%   w: weight of components
%   n: number of data
% Output:
%   X: d x n data matrix
%   z: 1 x n response variable
% Written by Mo Chen (sth4nth@gmail.com).
z = discreteRnd(w,n);
d = Theta{1}.dim();
X = zeros(d,n);
for i = 1:numel(w)
    idx = z==i;
    [mu,Sigma] = Theta{i}.sample(); % invpd(wishrnd(W0,v0));
    X(:,idx) = gaussRnd(mu,Sigma,sum(idx));
end
