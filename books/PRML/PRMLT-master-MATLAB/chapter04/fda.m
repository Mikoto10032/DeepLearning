function U = fda(X, t, q)
% Fisher (linear) discriminant analysis
% Input:
%   X: d x n data matrix
%   t: 1 x n class label
%   d: target dimension
% Output:
%   U: projection matrix y=U'*x
% Written by Mo Chen (sth4nth@gmail.com).
n = size(X,2);
k = max(t);

E = sparse(1:n,t,true,n,k,n);  % transform label into indicator matrix
nk = full(sum(E));

m = mean(X,2);
Xo = bsxfun(@minus,X,m);
St = (Xo*Xo')/n;                   % 4.43

mk = bsxfun(@times,X*E,1./nk);
mo = bsxfun(@minus,mk,m);
mo = bsxfun(@times,mo,sqrt(nk/n));
Sb = mo*mo';                       % 4.46
% Sw = St-Sb;                        % 4.45

[U,A] = eig(Sb,St,'chol');        
[~,idx] = sort(diag(A),'descend');
U = U(:,idx(1:q));
