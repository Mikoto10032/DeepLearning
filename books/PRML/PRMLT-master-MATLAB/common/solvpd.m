function V = solvpd(A,B)
% Compute A\B where A is a positive definite matrix
%   A: a positive difinie matrix
% Written by Mo Chen (sth4nth@gmail.com).
[U,p] = chol(A);
if p > 0
    error('ERROR: the matrix is not positive definite.');
end
V = U\(U'\B);