function y = logSt(X, mu, sigma, v)
% Compute log pdf of a Student's t distribution.
% Input:
%   X: d x n data matrix
%   mu: mean
%   sigma: variance
%   v: degree of freedom
% Output:
%   y: probability density in logrithm scale y=log p(x)
% Written by mo Chen (sth4nth@gmail.com).
[d,k] = size(mu);

if size(sigma,1)==d && size(sigma,2)==d && k==1
    [R,p]= chol(sigma);
    if p ~= 0
        error('ERROR: sigma is not SPD.');
    end
    X = bsxfun(@minus,X,mu);
    Q = R'\X;
    q = dot(Q,Q,1);  % quadratic term (M distance)
    o = -log(1+q/v)*((v+d)/2);
    c = gammaln((v+d)/2)-gammaln(v/2)-(d*log(v*pi)+2*sum(log(diag(R))))/2;
    y = c+o;
elseif size(sigma,1)==d && size(sigma,2)==k
    lambda = 1./sigma;
    ml = mu.*lambda;
    q = bsxfun(@plus,X'.^2*lambda-2*X'*ml,dot(mu,ml,1)); % M distance
    o = bsxfun(@times,log(1+bsxfun(@times,q,1./v)),-(v+d)/2);
    c = gammaln((v+d)/2)-gammaln(v/2)-(d*log(pi*v)+sum(log(sigma),1))/2;
    y = bsxfun(@plus,o,c);
elseif size(sigma,1)==1 && size(sigma,2)==k
    X2 = repmat(dot(X,X,1)',1,k);
    D = bsxfun(@plus,X2-2*X'*mu,dot(mu,mu,1));
    q = bsxfun(@times,D,1./sigma);  % M distance
    o = bsxfun(@times,log(1+bsxfun(@times,q,1./v)),-(v+d)/2);
    c = gammaln((v+d)/2)-gammaln(v/2)-d*log(pi*v.*sigma)/2;
    y = bsxfun(@plus,o,c);
else
    error('Parameters are mismatched.');
end
