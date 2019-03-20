function [model, mse] = mlp(X, T, h)
% Train a multilayer perceptron neural network
% Input:
%   X: d x n data matrix
%   T: p x n response matrix
%   h: L x 1 vector specify number of hidden nodes in each layer l
% Ouput:
%   model: model structure
%   mse: mean square error
% Written by Mo Chen (sth4nth@gmail.com).
eta = 1/size(X,2);
h = [size(X,1);h(:);size(T,1)];
L = numel(h);
W = cell(L-1,1);
for l = 1:L-1
    W{l} = randn(h(l),h(l+1));
end
Z = cell(L,1);
Z{1} = X;
maxiter = 200;
mse = zeros(1,maxiter);
for iter = 1:maxiter
%     forward
    for l = 2:L
        Z{l} = sigmoid(W{l-1}'*Z{l-1});   % 5.10, 5.49
    end
%     backward
    E = T-Z{L};
    mse(iter) =  mean(dot(E,E),1);
    for l = L-1:-1:1
        df = Z{l+1}.*(1-Z{l+1});
        dG = df.*E;
        dW = Z{l}*dG';
        W{l} = W{l}+eta*dW;
        E = W{l}*dG;
    end
end
mse = mse(1:iter);
model.W = W;