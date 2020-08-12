function [Y,s] = softmax(X, dim)
% Softmax function
%   By default dim = 1 (columns).
% Written by Mo Chen (sth4nth@gmail.com).
if nargin == 1
    dim = find(size(X)~=1,1);
    if isempty(dim), dim = 1; end
end
s = logsumexp(X,dim);
Y = exp(X-s);
