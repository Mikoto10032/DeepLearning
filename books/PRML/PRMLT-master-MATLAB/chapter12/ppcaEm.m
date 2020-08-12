function [W, mu, beta, llh] = ppcaEm(X, m)
% Perform EM algorithm to maiximize likelihood of probabilistic PCA model.
% Input:
%   X: d x n data matrix
%   m: dimension of target space
% Output:
%   W: d x m weight matrix
%   mu: d x 1 mean vector
%   beta: precition vector (inverse of variance
%   llh: loglikelihood
% Reference: 
%   Pattern Recognition and Machine Learning by Christopher M. Bishop 
%   Probabilistic Principal Component Analysis by Michael E. Tipping & Christopher M. Bishop
% Written by Mo Chen (sth4nth@gmail.com).
[d,n] = size(X);
mu = mean(X,2);
X = bsxfun(@minus,X,mu);

tol = 1e-4;
maxiter = 500;
llh = -inf(1,maxiter);
I = eye(m);
r = dot(X(:),X(:)); % total norm of X
W = randn(d,m); 
s = 1/randg;
for iter = 2:maxiter
    M = W'*W+s*I;
    U = chol(M);
    WX = W'*X;
    
    % likelihood
    logdetC = 2*sum(log(diag(U)))+(d-m)*log(s);
    T = U'\WX;
    trInvCS = (r-dot(T(:),T(:)))/(s*n);
    llh(iter) = -n*(d*log(2*pi)+logdetC+trInvCS)/2;                     % 12.43 12.44
    if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter-1)); break; end   % check likelihood for convergence
    
    % E step
    Ez = M\WX;                                     % 12.54
    V = inv(U);                                % inv(M) = V*V'
    Ezz = n*s*(V*V')+Ez*Ez'; % n*s because we are dealing with all n E[zi*zi']    % 12. 55
    
    % M step
    U = chol(Ezz);                                           
    W = ((X*Ez')/U)/U';                                 % 12.56
    WR = W*U';
    s = (r-2*dot(Ez(:),WX(:))+dot(WR(:),WR(:)))/(n*d);         % 12.57
end
llh = llh(2:iter);
beta = 1/s;