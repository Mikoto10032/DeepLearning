function z = maxdiff(x, y)
% Written by Mo Chen (sth4nth@gmail.com).
assert(all(size(x)==size(y)));
z = max(abs(x(:)-y(:)));

