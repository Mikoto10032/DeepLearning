function W = invpd(M)
% Compute A\B where A is a positive definite matrix
% Input:
%   M: a positive difinie matrix
% Written by Michael Chen (sth4nth@gmail.com).
[U,p] = chol(M);
if p > 0
    error('ERROR: the matrix is not positive definite.');
end
V = inv(U);
W = V*V';