function y = logdet(A)
% Compute log(det(A)) where A is positive definite.
% Written by Michael Chen (sth4nth@gmail.com).
[U,p] = chol(A);
if p > 0
    y = -inf;
else
    y = 2*sum(log(diag(U)));
end