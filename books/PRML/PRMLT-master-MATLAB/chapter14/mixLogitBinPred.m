function t = mixLogitBinPred(model, X)
% Prediction function for mixture of logistic regression
% input:
%   model: trained model structure
%   X: d x n data matrix
% output:
%   t: 1 x n cluster label
% Written by Mo Chen (sth4nth@gmail.com).
alpha = model.alpha; % mixing coefficient
W = model.W ;  % logistic model coefficentalpha
n = size(X,2);
X = [X; ones(1,n)];
t = round(alpha*sigmoid(W'*X));

