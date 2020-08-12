function [alpha, llh] = hmmFilter(model, x)
% HMM forward filtering algorithm. 
% The alpha returned by this function is the normalized version (posterior): alpha(t)=p(z_t|x_{1:t})
% Unnormalized version (joint distribution): alpha(t)=p(z_t,x_{1:t}) is numerical unstable.
% Input:
%   x: 1 x n integer vector which is the sequence of observations
%   model: model structure which contains
%       model.s: k x 1 start probability vector
%       model.A: k x k transition matrix
%       model.E: k x d emission matrix
% Output:
%   alpha: k x n matrix of posterior alpha(t)=p(z_t|x_{1:t})
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
s = model.s;
A = model.A;
E = model.E;

n = size(x,2);
X = sparse(x,1:n,1);
M = E*X;

[K,T] = size(M);
At = A';
llh = zeros(1,T);
alpha = zeros(K,T);
[alpha(:,1),llh(1)] = normalize(s.*M(:,1),1);
for t = 2:T
    [alpha(:,t),llh(t)] = normalize((At*alpha(:,t-1)).*M(:,t),1);    % 13.59
end
llh = sum(log(llh(llh>0)));