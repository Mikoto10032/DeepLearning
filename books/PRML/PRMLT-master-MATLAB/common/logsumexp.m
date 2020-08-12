function s = logsumexp(X, dim)
% Compute log(sum(exp(X),dim)) while avoiding numerical underflow.
%   By default dim = 1 (columns).
% Written by Mo Chen (sth4nth@gmail.com).
if nargin == 1
    dim = find(size(X)~=1,1);
    if isempty(dim), dim = 1; end
end
a = max(X,[],dim);
s = a+log(sum(exp(X-a),dim));   % TODO: use log1p
i = isinf(a);
s(i) = a(i);