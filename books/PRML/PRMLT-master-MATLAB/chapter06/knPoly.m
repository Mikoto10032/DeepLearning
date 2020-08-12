function K = knPoly(X, Y, o, c)
% Polynomial kernel k(x,y)=(x'y+c)^o
% Input:
%   X: d x nx data matrix
%   Y: d x ny data matrix
%   o: order of polynomial
%   c: constant
% Ouput:
%   K: nx x ny kernel matrix
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 4
    c = 0;
end

if nargin < 3
    o = 3;
end

if nargin < 2 || isempty(Y)  
    K = (dot(X,X,1)+c).^o;            % norm in kernel space
else
    K = (X'*Y+c).^o;
end

