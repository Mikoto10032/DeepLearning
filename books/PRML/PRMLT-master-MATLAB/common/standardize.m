function [Y, s] = standardize(X)
% Unitize the vectors to be unit length
%   By default dim = 1 (columns).
% Written by Mo Chen (sth4nth@gmail.com).
X = bsxfun(@minux,X,mean(X,2));
s = sqrt(mean(sum(X.^2,1)));
Y = X/s;