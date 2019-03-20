function [label, model, llh] = mixLinReg(X, y, k, lambda)
% Mixture of linear regression
% input:
%   X: d x n data matrix
%   y: 1 x n responding vector
%   k: number of mixture component
%   lambda: regularization parameter
% output:
%   label: 1 x n cluster label
%   model: trained model structure
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 4
    lambda = 1;
end
n = size(X,2);
X = [X;ones(1,n)]; % adding the bias term
d = size(X,1);
label = ceil(k*rand(1,n));  % random initialization
R = full(sparse(label,1:n,1,k,n,n));
tol = 1e-6;
maxiter = 500;
llh = -inf(1,maxiter);
Lambda = lambda*eye(d);
W = zeros(d,k);
Xy = bsxfun(@times,X,y);
beta = 1;
for iter = 2:maxiter
    % maximization
    nk = sum(R,2);
    alpha = nk/n;
    for j = 1:k
        Xw = bsxfun(@times,X,sqrt(R(j,:)));
        U = chol(Xw*Xw'+Lambda);
        W(:,j) = U\(U'\(Xy*R(j,:)'));  % 3.15 & 3.28
    end
    D = bsxfun(@minus,W'*X,y).^2;
    % expectation
    logRho = (-0.5)*beta*D;
    logRho = bsxfun(@plus,logRho,log(alpha));
    T = logsumexp(logRho,1);
    logR = bsxfun(@minus,logRho,T);
    R = exp(logR);
    llh(iter) = sum(T)/n;
    if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter)); break; end
end
llh = llh(2:iter);
model.alpha = alpha; % mixing coefficient
model.beta = beta; % mixture component precision
model.W = W;  % linear model coefficent
[~,label] = max(R,[],1);
model.label = label;
