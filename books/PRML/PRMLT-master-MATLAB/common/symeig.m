function [V,A,flag] = symeig(S,d,m)
% Compute eigenvalues and eigenvectors of symmetric matrix
%   m == 's' smallest (default)
%   m == 'l' largest
% Written by Mo Chen (sth4nth@gmail.com).
if nargin == 2
    m = 's';
end
opt.disp = 0;
opt.issym = 1;
opt.isreal = 1;
if any(m == 'ls')
    [V,A,flag] = eigs(S,d,[m,'a'],opt);
else
    error('The third parameter must be l or s.');
end
