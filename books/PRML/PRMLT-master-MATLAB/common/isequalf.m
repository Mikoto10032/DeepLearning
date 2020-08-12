function z = isequalf(x, y, tol)
% Determine whether two float number x and y are equal up to precision tol
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 3
    tol = 1e-8;
end
assert(all(size(x)==size(y)));
z = max(abs(x(:)-y(:)))<tol;

