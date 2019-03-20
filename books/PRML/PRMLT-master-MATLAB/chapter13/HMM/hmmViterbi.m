function [z, llh] = hmmViterbi(model, x)
% Viterbi algorithm (calculated in log scale to improve numerical stability).
% Input:
%   x: 1 x n integer vector which is the sequence of observations
%   model: model structure which contains
%       model.s: k x 1 start probability vector
%       model.A: k x k transition matrix
%       model.E: k x d emission matrix
% Output:
%   z: 1 x n latent state
%   llh:  loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
n = size(x,2);
X = sparse(x,1:n,1);
s = log(model.s);
A = log(model.A);
M = log(model.E*X);

k = numel(s);
Z = zeros(k,n);
Z(:,1) = 1:k;
v = s(:)+M(:,1);
for t = 2:n
    [v,idx] = max(bsxfun(@plus,A,v),[],1);    % 13.68
    v = v(:)+M(:,t);
    Z = Z(idx,:);
    Z(:,t) = 1:k;
end
[llh,idx] = max(v);
z = Z(idx,:);

