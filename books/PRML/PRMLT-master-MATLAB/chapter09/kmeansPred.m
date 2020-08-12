function [label, energy] = kmeansPred(mu, X)
% Prediction for kmeans clusterng
% Input:
%   model: dx k cluster center matrix
%   X: d x n testing data
% Output:
%   label: 1 x n cluster label
%   energy: optimization target value
% Written by Mo Chen (sth4nth@gmail.com).
[val,label] = min(dot(X,X,1)+dot(mu,mu,1)'-2*mu'*X,[],1); % assign labels
energy = sum(val);