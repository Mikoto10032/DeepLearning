function z = entropy(x)
% Compute entropy z=H(x) of a discrete variable x.
% Input:
%   x: a integer vectors  
% Output:
%   z: entropy z=H(x)
% Written by Mo Chen (sth4nth@gmail.com).
n = numel(x);
[~,~,x] = unique(x);
Px = accumarray(x, 1)/n;
Hx = -dot(Px,log2(Px));
z = max(0,Hx);