function [label, R] = mixGaussPred(model, X)
% Predict label and responsibility for Gaussian mixture model.
% Input:
%   X: d x n data matrix
%   model: trained model structure outputed by the EM algirthm
% Output:
%   label: 1 x n cluster label
%   R: k x n responsibility
% Written by Mo Chen (sth4nth@gmail.com).
mu = model.mu;
Sigma = model.Sigma;
w = model.w;

n = size(X,2);
k = size(mu,2);
logRho = zeros(n,k);

for i = 1:k
    logRho(:,i) = loggausspdf(X,mu(:,i),Sigma(:,:,i));
end
logRho = bsxfun(@plus,logRho,log(w));
T = logsumexp(logRho,2);
logR = bsxfun(@minus,logRho,T);
R = exp(logR);
[~,label(1,:)] = max(R,[],2);

function y = loggausspdf(X, mu, Sigma)
d = size(X,1);
X = bsxfun(@minus,X,mu);
[U,p]= chol(Sigma);
if p ~= 0
    error('ERROR: Sigma is not PD.');
end
Q = U'\X;
q = dot(Q,Q,1);  % quadratic term (M distance)
c = d*log(2*pi)+2*sum(log(diag(U)));   % normalization constant
y = -(c+q)/2;