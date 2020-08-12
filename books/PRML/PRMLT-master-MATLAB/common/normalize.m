function [Y, s] = normalize(X, dim)
% Normalize the vectors to be summing to one
%   By default dim = 1 (columns).
% Written by Michael Chen (sth4nth@gmail.com).
if nargin == 1
    % Determine which dimension sum will use
    dim = find(size(X)~=1,1);
    if isempty(dim), dim = 1; end
end
s = sum(X,dim);
Y = X./s;