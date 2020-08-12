function x = discreteRnd(p, n)
% Generate samples from a discrete distribution (multinomial).
% Input:
%   p: k dimensional probability vector
%   n: number of samples
% Ouput:
%   x: k x n generated samples x~Mul(p)
% Written by Mo Chen (sth4nth@gmail.com).
if nargin == 1
    n = 1;
end
r = rand(1,n);
p = cumsum(p(:));
[~,x] = histc(r,[0;p/p(end)]);
