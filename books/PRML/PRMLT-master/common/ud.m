function [U, D] = ud(X)
% UD factorization U'*D*U=X'*X;
% Written by Mo Chen (sth4nth@gmail.com).
[~,R] = qr(X,0);
d = diag(R);
D = d.^2;
U = bsxfun(@times,R,1./d);