function [y, p] = rvmBinPred(model, X)
% Prodict the label for binary logistic regression model
% Input:
%   model: trained model structure
%   X: d x n testing data
% Output:
%   y: 1 x n predict label (0/1)
%   p: 1 x n predict probability [0,1]
% Written by Mo Chen (sth4nth@gmail.com).
index = model.index;
X = [X;ones(1,size(X,2))];
X = X(index,:);
w = model.w;
p = sigmoid(w'*X); 
y = round(p);
