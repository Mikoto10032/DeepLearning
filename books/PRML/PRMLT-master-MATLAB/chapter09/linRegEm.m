function [model, llh] = linRegEm(X, t, alpha, beta)
% Fit empirical Bayesian linear regression model with EM (p.448 chapter 9.3.4)
% Input:
%   X: d x n data
%   t: 1 x n response
%   alpha: prior parameter
%   beta: prior parameter
% Output:
%   model: trained model structure
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 3
    alpha = 0.02;
    beta = 0.5;
end
[d,n] = size(X);
I = eye(d);
xbar = mean(X,2);
tbar = mean(t,2);

X = bsxfun(@minus,X,xbar);
t = bsxfun(@minus,t,tbar);

XX = X*X';
Xt = X*t';

tol = 1e-4;
maxiter = 100;
llh = -inf(1,maxiter+1);
for iter = 2:maxiter
    A = beta*XX+alpha*eye(d);
    U = chol(A);
    
    m = beta*(U\(U'\Xt));
    m2 = dot(m,m);
    e2 = sum((t-m'*X).^2);
    
    logdetA = 2*sum(log(diag(U)));    
    llh(iter) = 0.5*(d*log(alpha)+n*log(beta)-alpha*m2-beta*e2-logdetA-n*log(2*pi));  % 3.86
    if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter-1)); break; end
    
    invU = U'\I;
    trS = dot(invU(:),invU(:));    % A=inv(S)
    alpha = d/(m2+trS);   % 9.63
    
    invUX = U'\X;
    trXSX = dot(invUX(:),invUX(:));
    beta = n/(e2+trXSX);  % 9.68 is wrong
end
w0 = tbar-dot(m,xbar);

llh = llh(2:iter);
model.w0 = w0;
model.w = m;
%% optional for bayesian probabilistic inference purpose
model.alpha = alpha;
model.beta = beta;
model.xbar = xbar;
model.U = U;
