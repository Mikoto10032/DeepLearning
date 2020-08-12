function [model, llh] = ldsEm(X, init)
% EM algorithm for parameter estimation of linear dynamic system.
% NOTE: This is the exact implementation of the EM algorithm in PRML.
% However, this algorithm is not practical. It is numerical unstable and 
% there is too much redundant degree of freedom. 
% Input:
%   X: d x n data matrix
%   model: prior model structure
% Output:
%   model: trained model structure
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
d = size(X,1);
if isstruct(init)   % init with a model
    model = init;
elseif numel(init) == 1  % random init with latent k
    k = init;
    model.A = randn(k,k);
    model.G = iwishrnd(eye(k),k);
    model.C = randn(d,k);
    model.S = iwishrnd(eye(d),d);
    model.mu0 = randn(k,1);
    model.P0 = iwishrnd(eye(k),k);
end
tol = 1e-2;
maxIter = 100;
llh = -inf(1,maxIter);
for iter = 2:maxIter
%     E-step
    [nu, U, Ezz, Ezy, llh(iter)] = kalmanSmoother(model,X);
    if llh(iter)-llh(iter-1) < tol*abs(llh(iter-1)); break; end   % check likelihood for convergence
%     M-step 
    model = maximization(X, nu, U, Ezz, Ezy);
end
llh = llh(2:iter);

function model = maximization(X ,nu, U, Ezz, Ezy)
n = size(X,2);
mu0 = nu(:,1);
P0 = U(:,:,1);

Ezzn = sum(Ezz,3);
Ezz1 = Ezzn-Ezz(:,:,n);
Ezz2 = Ezzn-Ezz(:,:,1);
Ezy = sum(Ezy,3);

A = Ezy/Ezz1;                                           % 13.113
EzyA = Ezy*A';
G = (Ezz2-(EzyA+EzyA')+A*Ezz1*A')/(n-1);                % 13.114
Xnu = X*nu';
C = Xnu/Ezzn;                                           % 13.115
XnuC = Xnu*C';
S = (X*X'-(XnuC+XnuC')+C*Ezzn*C')/n;                    % 13.116

model.A = A;
model.G = G;
model.C = C;
model.S = S;
model.mu0 = mu0;
model.P0 = P0;
