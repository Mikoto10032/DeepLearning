function y = logDirichlet(X, a)
% Compute log pdf of a Dirichlet distribution.
% Input:
%   X: d x n data matrix, each column sums to one (sum(X,1)==ones(1,n) && X>=0)
%   a: d x k parameter of Dirichlet
%   y: k x n probability density
% Output:
%   y: k x n probability density in logrithm scale y=log p(x)
% Written by Mo Chen (sth4nth@gmail.com).
X = bsxfun(@times,X,1./sum(X,1));
if size(a,1) == 1
    a = repmat(a,size(X,1),1);
end
c = gammaln(sum(a,1))-sum(gammaln(a),1);
g = (a-1)'*log(X);
y = bsxfun(@plus,g,c');
