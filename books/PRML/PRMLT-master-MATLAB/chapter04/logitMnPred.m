function [y, P] = logitMnPred(model, X)
% Prediction of multiclass (multinomial) logistic regression model
% Input:
%   model: trained model structure
%   X: d x n testing data
% Output:
%   y: 1 x n predict label (1~k)
%   P: k x n predict probability for each class
% Written by Mo Chen (sth4nth@gmail.com).
W = model.W;
X = [X; ones(1,size(X,2))];
P = softmax(W'*X);
[~, y] = max(P,[],1);