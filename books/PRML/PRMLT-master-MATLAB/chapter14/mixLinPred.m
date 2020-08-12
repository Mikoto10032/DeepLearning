function [y, z, p] = mixLinPred(model, X, t)
% Prediction function for mxiture of linear regression
% input:
%   model: trained model structure
%   X: d x n data matrix
%   t:(optional) 1 x n responding vector
% output:
%   y: 1 x n prediction 
%   z: 1 x n cluster label
%   p: 1 x n predict probability for t
% Written by Mo Chen (sth4nth@gmail.com).
W = model.W;
alpha = model.alpha;
beta = model.beta;

X = [X;ones(1,size(X,2))]; % adding the bias term
y = W'*X;
D = bsxfun(@minus,y,t).^2;
logRho = (-0.5)*beta*D;
logRho = bsxfun(@plus,logRho,log(alpha));
T = logsumexp(logRho,1);
p = exp(T);
logR = bsxfun(@minus,logRho,T);
R = exp(logR);
z = max(R,[],1);
