function [gamma, alpha, beta, c] = hmmSmoother(model, x)
% HMM smoothing alogrithm (normalized forward-backward or normalized alpha-beta algorithm).
% The alpha and beta returned by this function are the normalized version.
% Input:
%   x: 1 x n integer vector which is the sequence of observations
%   model: model structure which contains
%       model.s: k x 1 start probability vector
%       model.A: k x k transition matrix
%       model.E: k x d emission matrix
% Output:
%   gamma: k x n matrix of posterior gamma(t)=p(z_t,x_{1:T})
%   alpha: k x n matrix of posterior alpha(t)=p(z_t|x_{1:T})
%   beta: k x n matrix of posterior beta(t)=gamma(t)/alpha(t)
%   c: 1 x n normalization constant vector
% Written by Mo Chen (sth4nth@gmail.com).
s = model.s;
A = model.A;
E = model.E;

n = size(x,2);
X = sparse(x,1:n,1);
M = E*X;

[K,T] = size(M);
At = A';
c = zeros(1,T); % normalization constant
alpha = zeros(K,T);
[alpha(:,1),c(1)] = normalize(s.*M(:,1),1);
for t = 2:T
    [alpha(:,t),c(t)] = normalize((At*alpha(:,t-1)).*M(:,t),1);  % 13.59
end
beta = ones(K,T);
for t = T-1:-1:1
    beta(:,t) = A*(beta(:,t+1).*M(:,t+1))/c(t+1);   % 13.62
end
gamma = alpha.*beta;                  % 13.64

