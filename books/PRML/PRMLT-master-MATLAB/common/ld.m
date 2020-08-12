% function [L, D] = ld(X)
% % LD factorization produces LDL'=X*X' which is the same as [L,D] = ldl(X*X');
% % the underlying algorithm is Gram-Schmidt orthogonalization
% [d,n] = size(X);
% m = min(d,n);
% L = eye(d,m);
% Q = zeros(m,n);
% D = zeros(m,1);
% for i = 1:m
%     L(i,1:i-1) = X(i,:)*bsxfun(@times,Q(1:i-1,:),1./D(1:i-1))';
%     Q(i,:) = X(i,:)-L(i,1:i-1)*Q(1:i-1,:);
%     D(i) = dot(Q(i,:),Q(i,:));
% end
% L(m+1:d,:) = X(m+1:d,:)*bsxfun(@times,Q,1./D)';

function [L, D] = ld(X)
% LD factorization produces LDL'=X*X' which is the same as [L,D] = ldl(X*X');
% the underlying algorithm is modified Gram-Schmidt orthogonalization
[d,n] = size(X);
m = min(d,n);
L = eye(d,m);
Q = zeros(m,n);
D = zeros(m,1);
for i = 1:m
    v = X(i,:);
    for j = 1:i-1
        L(i,j) = v*Q(j,:)'/D(j);
        v = v-L(i,j)*Q(j,:);
    end
    Q(i,:) = v;
    D(i) = dot(Q(i,:),Q(i,:));
end
L(m+1:d,:) = X(m+1:d,:)*bsxfun(@times,Q,1./D)';