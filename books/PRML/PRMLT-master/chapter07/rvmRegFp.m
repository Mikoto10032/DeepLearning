function [model, llh] = rvmRegFp(X, t, alpha, beta)
% Relevance Vector Machine (ARD sparse prior) for regression
% training by empirical bayesian (type II ML) using Mackay fix point update.
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
llh = -inf(1,maxiter);
index = 1:d;
alpha = alpha*ones(d,1);
for iter = 2:maxiter
    % remove zeros
    nz = 1./alpha > tol;    % nonzeros
    index = index(nz);
    alpha = alpha(nz);
    XX = XX(nz,nz);
    Xt = Xt(nz);
    X = X(nz,:);
    
    U = chol(beta*XX+diag(alpha));      % 7.83
    m = beta*(U\(U'\Xt));               % 7.82    
    m2 = m.^2;
    e = sum((t-m'*X).^2);
    
    logdetS = 2*sum(log(diag(U)));    
    llh(iter) = 0.5*(sum(log(alpha))+n*log(beta)-beta*e-logdetS-dot(alpha,m2)-n*log(2*pi)); % 3.86
    if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter-1)); break; end

    V = inv(U);    
    dgSigma = dot(V,V,2);
    gamma = 1-alpha.*dgSigma;   % 7.89
    alpha = gamma./m2;           % 7.87
    beta = (n-sum(gamma))/e;    % 7.88
end
llh = llh(2:iter);

model.index = index;
model.w0 = tbar-dot(m,xbar(nz));
model.w = m;
model.alpha = alpha;
model.beta = beta;
%% optional for bayesian probabilistic prediction purpose
model.xbar = xbar(index);
model.U = U;