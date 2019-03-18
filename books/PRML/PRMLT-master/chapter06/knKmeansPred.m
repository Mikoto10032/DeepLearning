function [label, energy] = knKmeansPred(model, Xt)
% Prediction for kernel kmeans clusterng
% Input:
%   model: trained model structure
%   Xt: d x n testing data
% Ouput:
%   label: 1 x n predict label
%   engery: optimization target value
% Written by Mo Chen (sth4nth@gmail.com).
X = model.X;
t = model.label;
kn = model.kn;

n = size(X,2);
k = max(t);
E = sparse(t,1:n,1,k,n,n);
E = E./sum(E,2);
Z = E*kn(X,Xt)-dot(E*kn(X,X),E,2)/2;
[val, label] = max(Z,[],1);
energy = sum(kn(Xt))-2*sum(val);
