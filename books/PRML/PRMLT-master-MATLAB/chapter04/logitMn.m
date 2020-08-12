function [model, llh] = logitMn(X, t, lambda)
% Multinomial regression for multiclass problem (Multinomial likelihood)
% Input:
%   X: d x n data matrix
%   t: 1 x n label (1~k)
%   lambda: regularization parameter
% Output:
%   model: trained model structure
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 3
    lambda = 1e-4;
end
X = [X; ones(1,size(X,2))];
[W, llh] = newtonRaphson(X, t, lambda);
% [W, llh] = newtonBlock(X, t, lambda);
model.W = W;

function [W, llh] = newtonRaphson(X, t, lambda)
[d,n] = size(X);
k = max(t);
tol = 1e-4;
maxiter = 100;
llh = -inf(1,maxiter);
dk = d*k;
idx = (1:dk)';
dg = sub2ind([dk,dk],idx,idx);
T = sparse(t,1:n,1,k,n,n);
W = zeros(d,k);
HT = zeros(d,k,d,k);
for iter = 2:maxiter
    A = W'*X;                                        % 4.105
    logY = bsxfun(@minus,A,logsumexp(A,1));            % 4.104
    llh(iter) = dot(T(:),logY(:))-0.5*lambda*dot(W(:),W(:));  % 4.108
    if abs(llh(iter)-llh(iter-1)) < tol; break; end
    Y = exp(logY);
    for i = 1:k
         for j = 1:k
            r = Y(i,:).*((i==j)-Y(j,:));  % r has negative value, so cannot use sqrt
            HT(:,i,:,j) = bsxfun(@times,X,r)*X';     % 4.110
        end
    end
    G = X*(Y-T)'+lambda*W;      % 4.96
    H = reshape(HT,dk,dk);
    H(dg) = H(dg)+lambda;
    W(:) = W(:)-H\G(:);    % 4.92
end
llh = llh(2:iter);

function [W, llh] = newtonBlock(X, t, lambda)
[d,n] = size(X);
k = max(t);
idx = (1:d)';
dg = sub2ind([d,d],idx,idx);
tol = 1e-4;
maxiter = 100;
llh = -inf(1,maxiter);
T = sparse(t,1:n,1,k,n,n);
W = zeros(d,k);
A = W'*X;
logY = bsxfun(@minus,A,logsumexp(A,1));
for iter = 2:maxiter
    for j = 1:k
        Y = exp(logY);
        Xw =  bsxfun(@times,X,sqrt(Y(j,:).*(1-Y(j,:))));
        H = Xw*Xw';
        H(dg) = H(dg)+lambda;
        g = X*(Y(j,:)-T(j,:))'+lambda*W(:,j);
        W(:,j) = W(:,j)-H\g;
        A(j,:) = W(:,j)'*X;
        logY = bsxfun(@minus,A,logsumexp(A,1));  % must be here to renormalize
    end
    llh(iter) = dot(T(:),logY(:))-0.5*lambda*dot(W(:),W(:));
    if abs(llh(iter)-llh(iter-1)) < tol; break; end
end
llh = llh(2:iter);
