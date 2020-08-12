function [Y, s] = unitize(X, dim)
% Unitize the vectors to be unit length
%   By default dim = 1 (columns).
% Written by Mo Chen (sth4nth@gmail.com).
if nargin == 1
    % Determine which dimension sum will use
    dim = find(size(X)~=1,1);
    if isempty(dim), dim = 1; end
end
s = sqrt(dot(X,X,dim));
Y = bsxfun(@times,X,1./s);