function x = dirichletRnd(a, m)
% Generate samples from a Dirichlet distribution.
% Input:
%   a: k dimensional vector
%   m: k dimensional mean vector
% Outpet:
%   x: generated sample x~Dir(a,m)
% Written by Mo Chen (sth4nth@gmail.com).
if nargin == 2
    a = a*m;
end
x = gamrnd(a,1);
x = x/sum(x);
