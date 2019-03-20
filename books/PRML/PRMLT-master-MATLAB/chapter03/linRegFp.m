function [model, llh] = linRegFp(X, t, alpha, beta)
% Fit empirical Bayesian linear model with Mackay fixed point method (p.168)
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


tol = 1e-4;
maxiter = 200;
llh = -inf(1,maxiter);
for iter = 2:maxiter
    A = beta*XX+diag(alpha);  % 3.81 3.54
    U = chol(A);

    m = beta*(U\(U'\Xt));  % 3.84
    m2 = dot(m,m);
    e = sum((t-m'*X).^2);   
    
    logdetA = 2*sum(log(diag(U)));    
    llh(iter) = 0.5*(d*log(alpha)+n*log(beta)-alpha*m2-beta*e-logdetA-n*log(2*pi)); % 3.86
    if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter-1)); break; end

    V = inv(U);               % A=inv(S)
    trS = dot(V(:),V(:));  
    gamma = d-alpha*trS;  % 3.91 9.64
    alpha = gamma/m2;    % 3.92
    beta = (n-gamma)/e;   % 3.95

end
w0 = tbar-dot(m,xbar);

llh = llh(2:iter);
model.w0 = w0;
model.w = m;
%% optional for bayesian probabilistic prediction purpose
model.alpha = alpha;
model.beta = beta;
model.xbar = xbar;
model.U = U;