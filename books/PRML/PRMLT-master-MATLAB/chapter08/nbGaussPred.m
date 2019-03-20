function y = nbGaussPred(model, X)
% Prediction of naive Bayes classifier with independent Gaussian.
% input:
%   model: trained model structure
%   X: d x n data matrix
% output:
%   y: 1 x n predicted class label
% Written by Mo Chen (sth4nth@gmail.com).
mu = model.mu;
var = model.var;
w = model.w;
assert(all(size(mu)==size(var)));
d = size(mu,1);

lambda = 1./var;
ml = mu.*lambda;
M = bsxfun(@plus,lambda'*X.^2-2*ml'*X,dot(mu,ml,1)'); % M distance
c = d*log(2*pi)+2*sum(log(var),1)'; % normalization constant
R = -0.5*bsxfun(@plus,M,c);
[~,y] = max(bsxfun(@times,exp(R),w),[],1);
