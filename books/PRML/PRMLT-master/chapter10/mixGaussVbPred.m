function [z, R] = mixGaussVbPred(model, X)
% Predict label and responsibility for Gaussian mixture model trained by VB.
% Input:
%   X: d x n data matrix
%   model: trained model structure outputed by the EM algirthm
% Output:
%   label: 1 x n cluster label
%   R: k x n responsibility
% Written by Mo Chen (sth4nth@gmail.com).
alpha = model.alpha; % Dirichlet
kappa = model.kappa;   % Gaussian
m = model.m;         % Gasusian
v = model.v;         % Whishart
U = model.U;         % Whishart 
logW = model.logW;
n = size(X,2);
[d,k] = size(m);

EQ = zeros(n,k);
for i = 1:k
    Q = (U(:,:,i)'\bsxfun(@minus,X,m(:,i)));
    EQ(:,i) = d/kappa(i)+v(i)*dot(Q,Q,1);    % 10.64
end
ElogLambda = sum(psi(0,0.5*bsxfun(@minus,v+1,(1:d)')),1)+d*log(2)+logW; % 10.65
Elogpi = psi(0,alpha)-psi(0,sum(alpha)); % 10.66
logRho = -0.5*bsxfun(@minus,EQ,ElogLambda-d*log(2*pi)); % 10.46
logRho = bsxfun(@plus,logRho,Elogpi);   % 10.46
logR = bsxfun(@minus,logRho,logsumexp(logRho,2)); % 10.49
R = exp(logR);
z = zeros(1,n);
[~,z(:)] = max(R,[],2);
[~,~,z(:)] = unique(z);

