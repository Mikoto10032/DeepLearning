function y = nbBernPred(model, X)
% Prediction of naive Bayes classifier with independent Bernoulli.
% input:
%   model: trained model structure
%   X: d x n data matrix
% output:
%   y: 1 x n predicted class label
% Written by Mo Chen (sth4nth@gmail.com).
mu = model.mu;
w = model.w;
[~,y] = max(log(mu)'*X+log(1-mu)'*(1-X)+log(w(:)),[],1);

