function [Q, R] = gson(X)
% Gram-Schmidt orthonormalization which produces the same result as [Q,R]=qr(X,0)
% Written by Mo Chen (sth4nth@gmail.com).
[d,n] = size(X);
m = min(d,n);
R = zeros(m,n);
Q = zeros(d,0);
for i = 1:m
    R(1:i-1,i) = Q'*X(:,i);
    v = X(:,i)-Q*R(1:i-1,i);
    R(i,i) = norm(v);
    Q(:,i) = v/R(i,i);
end
R(:,m+1:n) = Q'*X(:,m+1:n);