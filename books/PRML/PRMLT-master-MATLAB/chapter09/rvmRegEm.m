function [model, llh] = rvmRegEm(X, t, alpha, beta)
% Relevance Vector Machine (ARD sparse prior) for regression
% trained by empirical bayesian (type II ML) using EM
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
xbar = mean(X,2);
tbar = mean(t,2);
X = bsxfun(@minus,X,xbar);
t = bsxfun(@minus,t,tbar);
XX = X*X';
Xt = X*t';

tol = 1e-3;
maxiter = 500;
llh = -inf(1,maxiter+1);
index = 1:d;
alpha = alpha*ones(d,1);
for iter = 2 : maxiter
    nz = 1./alpha > tol ;   % nonzeros
    index = index(nz);
    alpha = alpha(nz);
    XX = XX(nz,nz);
    Xt = Xt(nz);
    X = X(nz,:);
    % E-step
    U = chol(beta*(XX)+diag(alpha));        % 7.83
    m = beta*(U\(U'\(X*t')));   % E[m]     % 7.82
    m2 = m.^2;       
    e2 = sum((t-m'*X).^2);

    logdetS = 2*sum(log(diag(U)));    
    llh(iter) = 0.5*(sum(log(alpha))+n*log(beta)-beta*e2-logdetS-dot(alpha,m2)-n*log(2*pi));  % 3.86
    if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter-1)); break; end
    % M-step
    V = inv(U);
    dgS = dot(V,V,2);
    alpha = 1./(m2+dgS);    % 9.67
    UX = U'\X;
    trXSX = dot(UX(:),UX(:));
    beta = n/(e2+trXSX);    % 9.68 is wrong
end
llh = llh(2:iter);

model.index = index;
model.w0 = tbar-dot(m,xbar(nz));
model.w = m;
model.alpha = alpha;
model.beta = beta;
%% optional for bayesian probabilistic prediction purpose
model.xbar = xbar;
model.U = U;