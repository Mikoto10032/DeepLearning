function [y, p] = logitBinPred(model, X)
% Prediction of binary logistic regression model
% Input:
%   model: trained model structure
%   X: d x n testing data
% Output:
%   y: 1 x n predict label (0/1)
%   p: 1 x n predict probability [0,1]
% Written by Mo Chen (sth4nth@gmail.com).
X = [X;ones(1,size(X,2))];
w = model.w;
p = sigmoid(w'*X);
y = round(p);

