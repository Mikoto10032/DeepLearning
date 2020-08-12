function [X, Z, model] = ldsRnd(d, k, n)
% Generate a data sequence from linear dynamic system.
% Input:
%   d: dimension of data
%   k: dimension of latent variable
%   n: number of data
% Output:
%   X: d x n data matrix
%   model: model structure
% Written by Mo Chen (sth4nth@gmail.com).
A = randn(k,k);
G = iwishrnd(eye(k),k);
C = randn(d,k);
S = iwishrnd(eye(d),d);
mu0 = randn(k,1);
P0 = iwishrnd(eye(k),k);

X = zeros(d,n);
Z = zeros(k,n);
Z(:,1) = gaussRnd(mu0,P0);              % 13.80
X(:,1) = gaussRnd(C*Z(:,1),S);
for i = 2:n
    Z(:,i) = gaussRnd(A*Z(:,i-1),G);           % 13.75, 13.78
    X(:,i) = gaussRnd(C*Z(:,i),S);      % 13.76, 13.79
end

model.A = A; % transition matrix 
model.G = G; % transition covariance
model.C = C; % emission matrix
model.S = S;  % emision covariance
model.mu0 = mu0; % prior mean
model.P0 = P0;  % prior covairance
