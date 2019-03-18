function [model, llh] = rvmBinFp(X, t, alpha)
% Relevance Vector Machine (ARD sparse prior) for binary classification.
% trained by empirical bayesian (type II ML) using Mackay fix point update.
% Input:
%   X: d x n data matrix
%   t: 1 x n label (0/1)
%   alpha: prior parameter
% Output:
%   model: trained model structure
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 3
    alpha = 1;
end
n = size(X,2);
X = [X;ones(1,n)];
d = size(X,1);
alpha = alpha*ones(d,1);
m = zeros(d,1);

tol = 1e-4;
maxiter = 100;
llh = -inf(1,maxiter);
index = 1:d;
for iter = 2:maxiter
    % remove zeros
    nz = 1./alpha > tol;    % nonzeros
    index = index(nz);
    alpha = alpha(nz);
    X = X(nz,:);
    m = m(nz); 
    
    [m,e,U] = logitBin(X,t,alpha,m);            % 7.110 ~ 7.113
    
    m2 = m.^2;
    llh(iter) = e(end)+0.5*(sum(log(alpha))-2*sum(log(diag(U)))-dot(alpha,m2)-n*log(2*pi)); % 7.114  & 7.118
    if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter-1)); break; end
    
    V = inv(U);
    dgS = dot(V,V,2);
    alpha = (1-alpha.*dgS)./m2;       % 7.89 & 7.87 & 7.116
end
llh = llh(2:iter);

model.index = index;
model.w = m;                  
model.alpha = alpha;


function [w, llh, U] = logitBin(X, t, lambda, w)
% Logistic regression
[d,n] = size(X);
tol = 1e-4;
maxiter = 100;
llh = -inf(1,maxiter);
idx = (1:d)';
dg = sub2ind([d,d],idx,idx);
h = ones(1,n);
h(t==0) = -1;
a = w'*X;
for iter = 2:maxiter
    y = sigmoid(a);                     % 4.87
    r = y.*(1-y);                       % 4.98
    Xw = bsxfun(@times, X, sqrt(r));
    H = Xw*Xw';                         % 4.97
    H(dg) = H(dg)+lambda;
    U = chol(H);
    g = X*(y-t)'+lambda.*w;             % 4.96
    p = -U\(U'\g);
    wo = w;                             % 4.92
    w = wo+p;   
    a = w'*X;   
    llh(iter) = -sum(log1pexp(-h.*a))-0.5*sum(lambda.*w.^2);  % 4.89
    incr = llh(iter)-llh(iter-1);
    while incr < 0      % line search
        p = p/2;
        w = wo+p;
        a = w'*X;   
        llh(iter) = -sum(log1pexp(-h.*a))-0.5*sum(lambda.*w.^2);
        incr = llh(iter)-llh(iter-1);
    end
    if incr < tol; break; end
end
llh = llh(2:iter);



