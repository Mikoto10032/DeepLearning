function z = logMn(x, p)
% Compute log pdf of a multinomial distribution.
% Input:
%   x: d x 1 integer vector 
%   p: d x 1 probability
% Output:
%   z: probability density in logrithm scale z=log p(x)
% Written by Mo Chen (sth4nth@gmail.com).    
z = gammaln(sum(x)+1)-sum(gammaln(x+1))+dot(x,log(p));
