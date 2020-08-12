function z = logKde (X, Y, sigma)
% Compute log pdf of kernel density estimator.
% Input:
%   X: d x n data matrix to be evaluate
%   Y: d x k data matrix served as database
% Output:
%   z: probability density in logrithm scale z=log p(x|y)
% Written by Mo Chen (sth4nth@gmail.com).
D = dot(X,X,1)+dot(Y,Y,1)'-2*(Y'*X);
z = logsumexp(D/(-2*sigma^2),1)-0.5*log(2*pi)-log(sigma*size(Y,2),1);
