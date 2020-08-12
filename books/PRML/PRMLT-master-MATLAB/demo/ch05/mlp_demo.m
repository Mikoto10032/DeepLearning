clear; close all;
h = [4,5];
X = [0 0 1 1;0 1 0 1];
T = [0 1 1 0];
[model,mse] = mlp(X,T,h);
plot(mse);
disp(['T = [' num2str(T) ']']);
Y = mlpPred(model,X);
disp(['Y = [' num2str(Y) ']']);